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
from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

SUSPENSION=0.10  # m: feet clear of the floor; one band step (key 7, -0.1 m) sets them down with the band still loaded
# Key 9 slackens the band like a gantry instead of dropping the robot: first the orientation and horizontal hold fade
# (ROPE_FADE) while the band still carries the weight, then the vertical support fades (RELEASE).
ROPE_FADE=1.0
RELEASE=2.0
LOWER_SPEED=0.1  # m/s: keys 7/8 move the band anchor at this speed instead of jumping 0.1 m
# Motor armature SONIC was trained with (gear_sonic/envs/manager_env/robots/g1.py); its PD gains are
# armature*(2*pi*10 Hz)^2. The stock scene gives every joint 0.01, which detunes each joint's response.
A5020,A7520_14,A7520_22,A4010=0.003609725,0.010177520,0.025101925,0.00425
ARMATURE={'hip_pitch':A7520_22,'hip_roll':A7520_22,'hip_yaw':A7520_14,'knee':A7520_22,
           'ankle_pitch':2*A5020,'ankle_roll':2*A5020,'waist_yaw':A7520_14,'waist_roll':2*A5020,'waist_pitch':2*A5020,
           'shoulder_pitch':A5020,'shoulder_roll':A5020,'shoulder_yaw':A5020,'elbow':A5020,'wrist_roll':A5020,
           'wrist_pitch':A4010,'wrist_yaw':A4010}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--idle',type=Path,help='Optional SONIC idle directory; otherwise use bundled Kimodo idle')
    p.add_argument('--upstream',type=Path,default=Path(__file__).resolve().parents[3]/'GR00T-WholeBodyControl')
    args=p.parse_args()
    import tempfile
    from sonic_laptop.idle import resolve_idle
    from sonic_laptop.limits import load as load_limits
    temporary=tempfile.TemporaryDirectory(prefix='sonic-sim-idle-')
    limits=load_limits()
    idle=resolve_idle(args.idle,Path(temporary.name),args.upstream,limits)
    model=load_model(args.upstream); arrays, q=read_clip(idle,model,limits)
    config=ArgsConfig().load_wbc_yaml(); config['INTERFACE']='lo';config['DOMAIN_ID']=0
    wrapper=SimWrapper(instantiate_g1_robot_model(),'default',config,onscreen=True,offscreen=False)
    sim=wrapper.sim;env=sim.sim_env
    addresses=[env.mj_model.jnt_qposadr[mujoco.mj_name2id(env.mj_model,mujoco.mjtObj.mjOBJ_JOINT,name)] for name in model['names']]
    for name in model['names']:
        joint=env.mj_model.joint(name)
        env.mj_model.dof_armature[joint.dofadr[0]]=ARMATURE[name.removeprefix('left_').removeprefix('right_').removesuffix('_joint')]
    def reset_idle():
        env.mj_data.qpos[addresses]=q[0]
        env.mj_data.qpos[:3]=arrays['body_pos'][0,:3]+[0,0,SUSPENSION]
        env.mj_data.qpos[3:7]=arrays['body_quat'][0,:4]
        env.mj_data.qvel[:]=0
        mujoco.mj_forward(env.mj_model,env.mj_data)
    reset_idle()
    # Balance the band, loaded with the robot's weight, at the held height. Its default 1 m anchor
    # otherwise yanks the robot up when the hold ends, swinging the arms into the clearance guard.
    band=env.elastic_band
    weight=-env.mj_model.opt.gravity[2]*env.mj_model.body_subtreemass[0]
    band.length=env.mj_data.xpos[env.band_attached_link][2]-band.point[2]+weight/band.kp_pos
    # Decode only the newest body command each step. The SDK listener decodes every 500 Hz command
    # in Python, which holds the simulator below real time once a controller streams.
    bridge=sim.unitree_bridge
    bridge.low_cmd_suber.Close()
    participant=DomainParticipant(config['DOMAIN_ID'])
    commands=DataReader(participant,Topic(participant,'rt/lowcmd',LowCmd_),Qos(Policy.History.KeepLast(1)))
    gains=(band.kp_pos,band.kd_pos,band.kp_ang,band.kd_ang)
    advance=band.Advance
    hold=[1.0]  # scale of the band's horizontal force and orientation torque
    def gantry_advance(pose):
        wrench=advance(pose)
        wrench[:2]*=hold[0];wrench[3:]*=hold[0]
        return wrench
    band.Advance=gantry_advance
    releasing=None
    was_enabled=band.enable
    length=target=band.length
    original=env.sim_step
    started=False
    def step():
        nonlocal started,releasing,was_enabled,length,target
        target+=band.length-length  # keys 7/8 step band.length by 0.1 m; follow at LOWER_SPEED instead
        length+=max(-LOWER_SPEED*sim.sim_dt,min(LOWER_SPEED*sim.sim_dt,target-length))
        band.length=length
        if was_enabled and not band.enable and releasing is None:
            band.enable=True;releasing=env.mj_data.time
        if releasing is not None:
            elapsed=env.mj_data.time-releasing
            hold[0]=max(0.0,1-elapsed/ROPE_FADE)
            scale=min(1.0,max(0.0,1-(elapsed-ROPE_FADE)/RELEASE))
            band.kp_pos,band.kd_pos=gains[0]*scale,gains[1]*scale
            if scale==0:
                band.enable=False;releasing=None;hold[0]=1.0
                band.kp_pos,band.kd_pos,band.kp_ang,band.kd_ang=gains
        was_enabled=band.enable
        for command in commands.take(1):
            bridge.low_cmd=command
        started=started or any(m.kp or m.kd or m.tau for m in sim.unitree_bridge.low_cmd.motor_cmd[:29])
        original()
        # Simulation fixture only, removed permanently once a body command arrives.
        if not started:
            reset_idle()
    env.sim_step=step
    try: sim.start()
    finally: sim.close()

if __name__=='__main__':main()
