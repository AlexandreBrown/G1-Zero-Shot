"""Runner-owned real SONIC/MuJoCo fixture with camera publishing (GPU environment)."""
import argparse
import importlib.util
import signal
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
    config.update(INTERFACE="lo", DOMAIN_ID=0, ENABLE_OFFSCREEN=True)
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
                     onscreen=True, offscreen=True, enable_image_publish=True).sim
    env = sim.sim_env
    recorder = SimulationRecorder(env.mj_model, env.mj_data, a.run_dir,
                                  a.upstream / config["ROBOT_SCENE"])
    task_sim = (task_module.TaskSimulation(env.mj_model, env.mj_data, a.run_dir / "task-start")
                if task_module else None)
    addresses = [env.mj_model.jnt_qposadr[mujoco.mj_name2id(
        env.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in robot["names"]]
    robot_joint_ids = list(env.body_joint_index) + list(env.left_hand_index) + list(env.right_hand_index)
    robot_nv = max(int(env.mj_model.jnt_dofadr[i]) for i in robot_joint_ids) + 1
    def reset_idle():
        env.mj_data.qpos[addresses] = q[0]
        env.mj_data.qpos[:3] = arrays["body_pos"][0, :3]
        env.mj_data.qpos[3:7] = arrays["body_quat"][0, :4]
        env.mj_data.qvel[:robot_nv] = 0
        mujoco.mj_forward(env.mj_model, env.mj_data)
    reset_idle()
    original, started = env.sim_step, False
    def step():
        nonlocal started
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
