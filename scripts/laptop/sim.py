#!/usr/bin/env python3
"""LOCAL simulation only: start upstream MuJoCo in the supplied clear idle pose."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco
from sonic_laptop.model import load_model
from sonic_laptop.motions import read_clip
from gear_sonic.scripts.run_sim_loop import ArgsConfig, SimWrapper, instantiate_g1_robot_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--idle',type=Path,required=True)
    p.add_argument('--upstream',type=Path,default=Path(__file__).resolve().parents[3]/'GR00T-WholeBodyControl')
    args=p.parse_args()
    model=load_model(args.upstream); arrays, q=read_clip(args.idle,model)
    config=ArgsConfig().load_wbc_yaml(); config['INTERFACE']='lo';config['DOMAIN_ID']=0
    wrapper=SimWrapper(instantiate_g1_robot_model(),'guarded_sonic_sim',config,onscreen=True,offscreen=False)
    sim=wrapper.sim;env=sim.sim_env
    addresses=[env.mj_model.jnt_qposadr[mujoco.mj_name2id(env.mj_model,mujoco.mjtObj.mjOBJ_JOINT,name)] for name in model['names']]
    def reset_idle():
        env.mj_data.qpos[addresses]=q[0]
        env.mj_data.qpos[:3]=arrays['body_pos'][0,:3]
        env.mj_data.qpos[3:7]=arrays['body_quat'][0,:4]
        env.mj_data.qvel[:]=0
        mujoco.mj_forward(env.mj_model,env.mj_data)
    reset_idle()
    original=env.sim_step
    started=False
    def step():
        nonlocal started
        started=started or any(m.kp or m.kd or m.tau for m in sim.unitree_bridge.low_cmd.motor_cmd[:29])
        original()
        # Simulation fixture only, removed permanently once a body command arrives.
        if not started:
            reset_idle()
    env.sim_step=step
    try: sim.start()
    finally: sim.close()

if __name__=='__main__':main()
