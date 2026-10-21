"""Run SONIC's MuJoCo simulator without a window; press 9 to toggle suspension."""

import select
import sys
import termios
import tty

from gear_sonic.scripts.run_sim_loop import ArgsConfig, SimWrapper, instantiate_g1_robot_model


def main():
    config = ArgsConfig(enable_onscreen=False, enable_offscreen=False)
    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = config.env_name
    sim = SimWrapper(
        robot_model=instantiate_g1_robot_model(),
        env_name=config.env_name,
        config=wbc_config,
        onscreen=False,
        offscreen=False,
    ).sim
    original = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        sim.start_as_thread()
        print("Headless simulator ready. Press 9 to toggle suspension; q to quit.", flush=True)
        while sim.sim_thread.is_alive():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if not key or key == "q":
                    break
                if key == "9":
                    sim.handle_keyboard_button(key)
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()
        if sim.sim_thread is not None:
            sim.sim_thread.join(timeout=5)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original)


if __name__ == "__main__":
    main()
