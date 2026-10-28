"""Runner-owned real SONIC/MuJoCo fixture with camera publishing (GPU environment)."""
import argparse
import importlib.util
import signal
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SUSPENSION = 0.10  # m: feet clear of the floor; one band step (key 7) sets them down with the band loaded
ROPE_FADE = 1.0  # s: key 9 first fades the band's orientation/horizontal hold...
RELEASE = 2.0  # s: ...then its vertical support (an instant release spiked the ankles to 18.5 rad/s)
LOWER_SPEED = 0.1  # m/s: keys 7/8 move the band anchor at this speed instead of jumping 0.1 m
# Motor armature SONIC was trained with (gear_sonic/envs/manager_env/robots/g1.py); the stock scene uses 0.01.
A5020, A7520_14, A7520_22, A4010 = 0.003609725, 0.010177520, 0.025101925, 0.00425
ARMATURE = {"hip_pitch": A7520_22, "hip_roll": A7520_22, "hip_yaw": A7520_14, "knee": A7520_22,
            "ankle_pitch": 2 * A5020, "ankle_roll": 2 * A5020, "waist_yaw": A7520_14,
            "waist_roll": 2 * A5020, "waist_pitch": 2 * A5020, "shoulder_pitch": A5020,
            "shoulder_roll": A5020, "shoulder_yaw": A5020, "elbow": A5020, "wrist_roll": A5020,
            "wrist_pitch": A4010, "wrist_yaw": A4010}


def standing_command_received(motors):
    """Do not release the initial simulated pose for a damping-only command."""
    return any(m.kp > 0 for m in motors[:29])


def main():
    import mujoco
    from sonic_laptop.model import load_model
    from sonic_laptop.motions import read_clip
    from gpt_astra.recording import SimulationRecorder
    from gear_sonic.scripts.run_sim_loop import ArgsConfig, SimWrapper, instantiate_g1_robot_model
    from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory
    from cyclonedds.core import Policy, Qos
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--idle", type=Path, required=True)
    p.add_argument("--task-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--camera-port", type=int, default=5555)
    a = p.parse_args()
    robot = load_model(a.upstream)
    arrays, q = read_clip(a.idle, robot)
    config = ArgsConfig().load_wbc_yaml()
    # astra_ws: ego camera at 10 Hz (GPT samples ~every 2 s); the 30 Hz default plus the viewer kept
    # the sim at 0.7x real time, and SONIC oscillated once the band released.
    config.update(INTERFACE="lo", DOMAIN_ID=0, ENABLE_OFFSCREEN=True, IMAGE_DT=0.1)
    task_module = None
    task_script = a.task_dir / "sim.py"
    if task_script.is_file():
        spec = importlib.util.spec_from_file_location("gpt_astra_task_sim", task_script)
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)
        source_scene = a.upstream / config["ROBOT_SCENE"]
        config["ROBOT_SCENE"] = str(task_module.prepare_scene(source_scene, a.run_dir).resolve())
    # The upstream minimal simulator only accepts the 'default' environment.
    sim = SimWrapper(instantiate_g1_robot_model(), "default", config,
                     onscreen=False, offscreen=True, enable_image_publish=True).sim
    env = sim.sim_env
    recorder = SimulationRecorder(env.mj_model, env.mj_data, a.run_dir,
                                  a.upstream / config["ROBOT_SCENE"])
    task_sim = (task_module.TaskSimulation(env.mj_model, env.mj_data, a.run_dir / "task-start")
                if task_module else None)
    addresses = [env.mj_model.jnt_qposadr[mujoco.mj_name2id(
        env.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in robot["names"]]
    # astra_ws sim fixes (ported from the G1-Zero-Shot session's scripts/laptop/sim.py, 2026-10-28).
    for name in robot["names"]:
        joint = env.mj_model.joint(name)
        env.mj_model.dof_armature[joint.dofadr[0]] = ARMATURE[
            name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")]
    robot_joint_ids = list(env.body_joint_index) + list(env.left_hand_index) + list(env.right_hand_index)
    robot_nv = max(int(env.mj_model.jnt_dofadr[i]) for i in robot_joint_ids) + 1
    def reset_idle():
        env.mj_data.qpos[addresses] = q[0]
        env.mj_data.qpos[:3] = arrays["body_pos"][0, :3] + [0, 0, SUSPENSION]
        env.mj_data.qpos[3:7] = arrays["body_quat"][0, :4]
        env.mj_data.qvel[:robot_nv] = 0
        mujoco.mj_forward(env.mj_model, env.mj_data)
    reset_idle()
    # Balance the band, loaded with the robot's weight, at the held height; the stock 1 m anchor
    # yanks the robot up when the hold ends and swings the arms into the clearance guard.
    band = env.elastic_band
    weight = -env.mj_model.opt.gravity[2] * env.mj_model.body_subtreemass[0]
    band.length = env.mj_data.xpos[env.band_attached_link][2] - band.point[2] + weight / band.kp_pos
    # Decode only the newest body command each step; the SDK listener decodes every 500 Hz
    # command in Python and holds the simulator below real time (guard: state stale >100 ms).
    bridge = sim.unitree_bridge
    bridge.low_cmd_suber.Close()
    participant = DomainParticipant(config["DOMAIN_ID"])
    commands = DataReader(participant, Topic(participant, "rt/lowcmd", LowCmd_), Qos(Policy.History.KeepLast(1)))
    gains = (band.kp_pos, band.kd_pos, band.kp_ang, band.kd_ang)
    advance = band.Advance
    hold = [1.0]  # scale of the band's horizontal force and orientation torque
    def gantry_advance(pose):
        wrench = advance(pose)
        wrench[:2] *= hold[0]
        wrench[3:] *= hold[0]
        return wrench
    band.Advance = gantry_advance
    releasing, was_enabled = None, band.enable
    length = target = band.length
    original, started = env.sim_step, False
    # Run-folder triggers do what keys 7 and 9 do, for drivers that cannot send keys to the viewer
    # (synthetic X key events to an off-screen GLFW window are ignored).
    triggers = {"band_lower": False, "band_release": False}
    def step():
        nonlocal started, releasing, was_enabled, length, target
        if env.mj_data.time % 0.1 < sim.sim_dt:
            for name, seen in triggers.items():
                if not seen and (a.run_dir / name).exists():
                    triggers[name] = True
                    print(f"[sim] trigger {name}", flush=True)
                    if name == "band_lower":
                        band.length -= 0.1
                    else:
                        band.enable = False
        # Keys 7/8 step band.length by 0.1 m; follow at LOWER_SPEED instead of jumping.
        target += band.length - length
        length += max(-LOWER_SPEED * sim.sim_dt, min(LOWER_SPEED * sim.sim_dt, target - length))
        band.length = length
        # Key 9 slackens the band like a gantry: orientation/horizontal hold fades first, then the weight.
        if was_enabled and not band.enable and releasing is None:
            band.enable, releasing = True, env.mj_data.time
        if releasing is not None:
            elapsed = env.mj_data.time - releasing
            hold[0] = max(0.0, 1 - elapsed / ROPE_FADE)
            scale = min(1.0, max(0.0, 1 - (elapsed - ROPE_FADE) / RELEASE))
            band.kp_pos, band.kd_pos = gains[0] * scale, gains[1] * scale
            if scale == 0:
                band.enable, releasing, hold[0] = False, None, 1.0
                band.kp_pos, band.kd_pos, band.kp_ang, band.kd_ang = gains
        was_enabled = band.enable
        for command in commands.take(1):
            bridge.low_cmd = command
        # Damping-only messages have kp=0 and do not provide a standing controller.
        started = started or standing_command_received(sim.unitree_bridge.low_cmd.motor_cmd)
        if task_sim:
            task_sim.step()
        original()
        if not started:
            reset_idle()
        recorder.capture()
    env.sim_step = step
    # MuJoCo renders RGB; upstream's OpenCV JPEG publisher expects BGR.
    # Correct the color order at the publisher boundary, without touching upstream.
    original_publisher_start = sim.start_image_publish_subprocess
    def start_publisher(*args, **kwargs):
        original_publisher_start(*args, **kwargs)
        publisher = env.image_publish_process
        publish = publisher.update_shared_memory
        publisher.update_shared_memory = lambda images: publish(
            {name: image[:, :, ::-1].copy() for name, image in images.items()})
    sim.start_image_publish_subprocess = start_publisher
    def stop_simulator(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_simulator)
    try:
        SimulatorFactory.start_simulator(sim, as_thread=False, enable_image_publish=True,
                                         camera_port=a.camera_port)
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
