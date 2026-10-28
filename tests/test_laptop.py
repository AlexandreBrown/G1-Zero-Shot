import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from sonic_laptop.model import load_model, clearance
from sonic_laptop.motions import stage_library
from sonic_laptop.prepare import generate_model, patch, INPUT
from sonic_laptop.session import Session

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT.parent/'GR00T-WholeBodyControl'

class LaptopTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.p=Path(self.tmp.name)
        self.model=load_model(UPSTREAM)
        self.q=np.array(self.model['default']); self.q[[15,22]]=-.4
        self.q[16]=.4; self.q[23]=-.4; self.q[[18,25]]=1
    def tearDown(self): self.tmp.cleanup()
    def clip(self,name, q=None):
        p=self.p/name;p.mkdir();q=self.q if q is None else q
        isaac=np.zeros(29);isaac[self.model['mapping']]=q
        for name,values in [('joint_pos',isaac),('joint_vel',np.zeros(29)),('body_pos',[0,0,.793]),('body_quat',[1,0,0,0])]:
            np.savetxt(p/(name+'.csv'),np.tile(values,(50,1)),delimiter=',',header=','.join(['x']*len(values)),comments='')
        (p/'metadata.txt').write_text('Body part indexes:\n[0]\n')
        return p
    def test_snapshot_and_limits(self):
        idle=self.clip('idle'); self.clip('wave')
        catalog=stage_library(idle,self.p,self.p/'session/motions',self.model)
        self.assertEqual(len(catalog),2)
        # Source edits cannot change an already accepted reference snapshot.
        (idle/'joint_pos.csv').write_text('corrupt')
        self.assertGreater((self.p/'session/motions/0000/joint_pos.csv').stat().st_size,100)
        q=self.q.copy();q[0]=100
        bad=self.clip('bad',q)
        from sonic_laptop.motions import read_clip
        with self.assertRaisesRegex(ValueError,'limit'): read_clip(bad,self.model)
    def test_collision_and_endpoints(self):
        from sonic_laptop.motions import read_clip
        self.assertGreater(clearance(self.model,self.q),.025)
        self.assertLess(clearance(self.model,self.model['default']),.025)
        with self.assertRaisesRegex(ValueError,'clearance'):
            read_clip(self.clip('body_collision',np.array(self.model['default'])),self.model)
        path=self.clip('moving_end');a=np.loadtxt(path/'joint_vel.csv',delimiter=',',skiprows=1)
        a[-1,0]=.2;np.savetxt(path/'joint_vel.csv',a,delimiter=',',header=','.join(['x']*29),comments='')
        with self.assertRaisesRegex(ValueError,'stationary'):read_clip(path,self.model)
    def test_fk_matches_mujoco(self):
        import mujoco
        m=mujoco.MjModel.from_xml_path(str(UPSTREAM/'gear_sonic_deploy/g1/g1_29dof.xml'))
        d=mujoco.MjData(m)
        # Independent MuJoCo forward kinematics for the same envelope calculation.
        for name,q in zip(self.model['names'],self.q):
            j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name); d.qpos[m.jnt_qposadr[j]]=q
        mujoco.mj_forward(m,d)
        def point(name,p):
            i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
            return d.xpos[i]+d.xmat[i].reshape(3,3)@p
        hands=[point(side+'_wrist_yaw_link',self.model['hand_offset']) for side in ('left','right')]
        distances=[np.linalg.norm(hands[0]-hands[1])-2*self.model['hand_radius']]
        for hand in hands:
            for name,center,half in self.model['boxes']:
                i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
                delta=np.abs(d.xmat[i].reshape(3,3).T@(hand-d.xpos[i])-center)-half
                distances.append(np.linalg.norm(np.maximum(delta,0))+min(max(delta),0)-self.model['hand_radius'])
        self.assertAlmostEqual(clearance(self.model,self.q),min(distances),places=7)
    def test_patch_and_cpp_gate(self):
        tree=patch(UPSTREAM/'gear_sonic_deploy',self.p/'deploy')
        text=(tree/'src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp').read_text()
        self.assertIn('if(guard_.fault.load()) dex3_hands_.writeDampingOnly();',text)
        self.assertNotIn('dex3_hands_.open(',text)
        self.assertNotIn('dex3_hands_.close(',text)
        self.assertNotIn('dex3_hands_.writeOnce(',text)
        self.assertIn('current_motion_->timesteps - 1',text)
        self.assertIn('Setup failed: ',text)
        self.assertIn('guard_.sim ? ramp_start[i]',text)
        self.assertIn('kps[i] * guard_.setup_gain(i)',text)
        self.assertIn('guard_.planner_active.store(keyboard->use_planner);',text)
        sdk=UPSTREAM/'gear_sonic_deploy/thirdparty/unitree_sdk2'
        subprocess.run(['g++','-std=c++17','-fsyntax-only','-x','c++','-I/usr/include/eigen3',
            '-I'+str(sdk/'include'),'-I'+str(sdk/'thirdparty/include'),
            '-I'+str(sdk/'thirdparty/include/ddscxx'),
            '-include',str(tree/'src/g1/g1_deploy_onnx_ref/include/dex3_hands.hpp'),'/dev/null'],check=True)
        (self.p/'guard_model.hpp').write_text(generate_model(self.model))
        (self.p/'guard.hpp').write_text((ROOT/'sonic_laptop/native/guard.hpp').read_text())
        (self.p/'init.txt').write_text(' '.join(map(str,self.q)))
        source=r'''
#include "guard.hpp"
#include <cassert>
#include <limits>
struct Motor {double value=0,speed=0;double q() const{return value;}double dq() const{return speed;}};
struct State {std::array<Motor,29> motors;const auto& motor_state() const{return motors;}};
struct Command {std::array<double,29> q_target{},dq_target{},kp{},kd{},tau_ff{};};
int main(int argc,char**argv) {
 sonic_guard::Deadman d;
 assert(!d.active(10));assert(d.refresh(0,10,10));assert(d.active(10.149));assert(!d.active(10.151));
 assert(!d.refresh(0,9,10));assert(!d.refresh(0,11,10));assert(!d.refresh(4,10,10));
 assert(!d.refresh(0,std::numeric_limits<double>::quiet_NaN(),10));
 assert(d.refresh(1,10,10));d.cancel();assert(!d.active(10));
 setenv("SONIC_GUARD_DIR",argv[1],1);
 auto lease=[&](){std::ofstream f(std::string(argv[1])+"/lease");f.precision(17);f<<sonic_guard::now()<<" 1\n";};
 lease(); sonic_guard::Guard g;Command c;State s;
 c.q_target=g.init;for(int i=0;i<29;i++)s.motors[i].value=g.init[i];
 assert(g.validate(c,s,.01,.01));
 g.waiting.store(true);assert(g.validate(c,s,.01,.01));assert(!g.setup_ready.load());
 g.settled_since=sonic_guard::now()-2.1;assert(g.validate(c,s,.01,.01));assert(g.setup_ready.load());
 s.motors[0].value+=.13;assert(g.validate(c,s,.01,.01));assert(!g.setup_ready.load());s.motors[0].value=g.init[0];
 c.q_target[0]=std::numeric_limits<double>::quiet_NaN();assert(!g.validate(c,s,.01,.01));
 c.q_target=g.init;assert(!g.validate(c,s,.01,.01)); // latch cannot recover
 lease();sonic_guard::Guard stale;assert(!stale.validate(c,s,.11,.01));
 lease();sonic_guard::Guard jump;assert(jump.validate(c,s,.01,.01));c.q_target[0]+=.3;
 assert(!jump.validate(c,s,.01,.01));c.q_target=g.init;
 {std::ofstream f(std::string(argv[1])+"/lease");f<<"0 1\n";}
 sonic_guard::Guard expired;assert(!expired.validate(c,s,.01,.01));
 lease();sonic_guard::Guard stalled;stalled.input_tick.store(sonic_guard::now()-.2);assert(!stalled.validate(c,s,.01,.01));
 lease();sonic_guard::Guard limit;c.q_target[0]=100;assert(!limit.validate(c,s,.01,.01));
 assert(!limit.sim && limit.setup_gain(20)==1.0f);
 // Simulation profile: SONIC's start-up transients pass, runaway or corrupt commands still latch.
 setenv("SONIC_GUARD_SIM","1",1);c.q_target=g.init;
 lease();sonic_guard::Guard sim;assert(sim.sim && sim.setup_gain(20)==8.0f && sim.setup_gain(13)==4.0f && sim.setup_gain(3)==1.0f);
 assert(sim.validate(c,s,.01,.01));c.q_target[4]+=.9;assert(sim.validate(c,s,.01,.01));
 s.motors[4].speed=20;assert(sim.validate(c,s,.01,.01));s.motors[4].speed=0;
 c.q_target[4]+=1.6;assert(!sim.validate(c,s,.01,.01));c.q_target=g.init;
 lease();sonic_guard::Guard runaway;assert(runaway.validate(c,s,.01,.01));s.motors[3].speed=17;assert(!runaway.validate(c,s,.01,.01));s.motors[3].speed=0;
 lease();sonic_guard::Guard corrupt;assert(corrupt.validate(c,s,.01,.01));c.q_target[0]=1e30;assert(!corrupt.validate(c,s,.01,.01));
 unsetenv("SONIC_GUARD_SIM");
}
'''
        (self.p/'test.cpp').write_text(source)
        subprocess.run(['g++','-std=c++17','-O3','-fno-fast-math','-I/usr/include/eigen3',str(self.p/'test.cpp'),'-o',str(self.p/'gate')],check=True)
        subprocess.run([str(self.p/'gate'),str(self.p)],check=True)
    def test_native_keyboard_and_ipc(self):
        tree=patch(UPSTREAM/'gear_sonic_deploy',self.p/'deploy')
        include=tree/'src/g1/g1_deploy_onnx_ref/include'
        (include/'guarded_input.inc').write_text(INPUT)
        (self.p/'init.txt').write_text(' '.join(map(str,self.q)))
        subprocess.run(['g++','-std=c++17','-O2','-fno-fast-math','-I/usr/include/eigen3',
                        '-I'+str(include),str(ROOT/'tests/native_keyboard.cpp'),'-lyaml-cpp','-lz',
                        '-o',str(self.p/'keyboard')],check=True)
        subprocess.run([str(self.p/'keyboard'),str(self.p)],check=True,timeout=10)

    def test_hand_damping_command(self):
        sdk=UPSTREAM/'gear_sonic_deploy/thirdparty/unitree_sdk2'
        subprocess.run(['g++','-std=c++17','-I'+str(sdk/'include'),
            '-I'+str(sdk/'thirdparty/include'),'-I'+str(sdk/'thirdparty/include/ddscxx'),
            '-I'+str(ROOT/'sonic_laptop/native'),str(ROOT/'tests/native_hand_damping.cpp'),
            '-o',str(self.p/'hands')],check=True)
        subprocess.run([str(self.p/'hands')],check=True)

    def test_activity_log_and_status(self):
        s=Session(self.p,self.p,'lo',True)
        (self.p/'controller.log').write_text('[Setup] Loading policy\npartial')
        self.assertEqual(s.read_activity(),['[Setup] Loading policy'])
        self.assertEqual(s.setup_phase,'Loading policy')
        with (self.p/'controller.log').open('a') as f:f.write(' line\n')
        self.assertEqual(s.read_activity(),['partial line'])
        self.assertEqual(s.read_activity(),[])
        (self.p/'status').write_text('1 1 0 0 0 0 0 1\nok\n\nJoint settling details\n')
        self.assertTrue(s.poll()['setup_ready'])
        self.assertEqual(s.status['detail'],'Joint settling details')

    def test_idle_overrides_pending_move(self):
        from unittest.mock import Mock
        s=Session(self.p,self.p,'lo',True);s.process=Mock();s.process.poll.return_value=None
        self.assertTrue(s.command('move',0))
        self.assertFalse(s.command('move',1))
        self.assertTrue(s.command('idle'))
        fields=(self.p/'request').read_text().split()
        self.assertEqual(fields[:3],['2','idle','0'])
        self.assertGreater(float(fields[3]),0)

    def test_real_never_disables_crc_and_stale_status(self):
        # A SIM/HW mismatch must fail before reading or executing any binary.
        s=Session(self.p,self.p,'lo',False)
        with self.assertRaisesRegex(ValueError,'Simulation'):s.arm()
        s.stop()
        with self.assertRaises(RuntimeError):s.command('motion',0)

if __name__=='__main__':unittest.main()
