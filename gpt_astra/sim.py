"""Runner-owned real SONIC/MuJoCo fixture with camera publishing (GPU environment)."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import mujoco
    from sonic_laptop.model import load_model
    from sonic_laptop.motions import read_clip
    from gear_sonic.scripts.run_sim_loop import ArgsConfig, SimWrapper, instantiate_g1_robot_model
    from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--idle", type=Path, required=True)
    p.add_argument("--camera-port", type=int, default=5555)
    a = p.parse_args()
    robot = load_model(a.upstream)
    arrays, q = read_clip(a.idle, robot)
    config = ArgsConfig().load_wbc_yaml()
    config.update(INTERFACE="lo", DOMAIN_ID=0, ENABLE_OFFSCREEN=True)
    sim = SimWrapper(instantiate_g1_robot_model(), "astra_sonic", config,
                     onscreen=True, offscreen=True, enable_image_publish=True).sim
    env = sim.sim_env
    addresses = [env.mj_model.jnt_qposadr[mujoco.mj_name2id(
        env.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in robot["names"]]
    def reset_idle():
        env.mj_data.qpos[addresses] = q[0]
        env.mj_data.qpos[:3] = arrays["body_pos"][0, :3]
        env.mj_data.qpos[3:7] = arrays["body_quat"][0, :4]
        env.mj_data.qvel[:] = 0
        mujoco.mj_forward(env.mj_model, env.mj_data)
    reset_idle()
    original, started = env.sim_step, False
    def step():
        nonlocal started
        started = started or any(m.kp or m.kd or m.tau for m in sim.unitree_bridge.low_cmd.motor_cmd[:29])
        original()
        if not started:
            reset_idle()
    env.sim_step = step
    SimulatorFactory.start_simulator(sim, as_thread=False, enable_image_publish=True,
                                     camera_port=a.camera_port)


if __name__ == "__main__":
    main()
