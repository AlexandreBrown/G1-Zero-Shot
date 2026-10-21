"""Build a separate, reproducible patched deployment tree; leave upstream untouched."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from .model import load_model

ROOT = Path(__file__).resolve().parents[1]
REL = Path('src/g1/g1_deploy_onnx_ref')


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Upstream changed: expected exactly one anchor: {old[:90]!r}')
    return text.replace(old,new,1)


def generate_model(model):
    def nums(xs): return ','.join(format(x,'.12g') for x in xs)
    lines=['#pragma once','#include <Eigen/Geometry>','#include <array>', 'namespace sonic_guard {']
    for name in ('lower','upper','torque'):
        lines.append(f'inline const std::array<double,29> {name} = {{{nums(model[name])}}};')
    lines += [f'inline constexpr double minimum_clearance = {model["clearance"]};',
              'inline double clearance(const std::array<double,29>& q) {',
              f'std::array<Eigen::Isometry3d,{len(model["bodies"])}> t;',
              'using V=Eigen::Vector3d;']
    names={b['name']:i for i,b in enumerate(model['bodies'])}
    for i,b in enumerate(model['bodies']):
        lines += [f't[{i}]=Eigen::Isometry3d::Identity();',
                  f't[{i}].translate(V({nums(b["pos"])}));',
                  f't[{i}].rotate(Eigen::Quaterniond({nums(b["quat"])}).normalized());']
        if b['joint']>=0:
            lines += [f't[{i}].rotate(Eigen::AngleAxisd(q[{b["joint"]}],V({nums(b["axis"])})));']
        if b['parent']>=0: lines += [f't[{i}]=t[{b["parent"]}]*t[{i}];']
    lines += [f'V l=t[{names["left_wrist_yaw_link"]}]*V({nums(model["hand_offset"])});',
              f'V r=t[{names["right_wrist_yaw_link"]}]*V({nums(model["hand_offset"])});',
              f'double d=(l-r).norm()-{2*model["hand_radius"]};', 'for(const V& h:{l,r}) {']
    for name,center,half in model['boxes']:
        lines += ['{',f'V delta=(t[{names[name]}].inverse()*h-V({nums(center)})).cwiseAbs()-V({nums(half)});',
                  f'd=std::min(d,delta.cwiseMax(0).norm()+std::min(delta.maxCoeff(),0.0)-{model["hand_radius"]});','}']
    lines += ['} return d; }','}']
    return '\n'.join(lines)+'\n'


INPUT = r'''
      guard_.input_tick.store(sonic_guard::now());
      guard_.waiting.store(program_state_==ProgramState::WAIT_FOR_CONTROL);
      static sonic_guard::Deadman deadman;
      static unsigned long guard_seq=0;
      static std::string guard_result="ready";
      auto* keyboard=dynamic_cast<SimpleKeyboard*>(input_interface_.get());
      if(!keyboard) { guard_.trip("Only guarded keyboard input is supported"); return; }
      if(keyboard->guard_planner_start_pending && sonic_guard::now()-keyboard->guard_planner_start_time>5)
        guard_.trip("Planner startup timed out (>5 s)");
      unsigned long sequence=0; std::string command; int argument=0; double stamp=0;
      std::ifstream request(guard_.dir+"/request");
      if(request>>sequence>>command>>argument>>stamp && sequence>guard_seq && !guard_.fault.load()) {
        guard_seq=sequence;
        guard_result="rejected: invalid command/state";
        bool fresh=std::isfinite(stamp) && sonic_guard::now()-stamp<=.15 && stamp<=sonic_guard::now()+.01;
        if(!fresh) { deadman.cancel(); guard_result="rejected: expired command"; }
        else if(command=="grounded_start" && program_state_==ProgramState::WAIT_FOR_CONTROL && guard_.setup_ready.load()) {
          keyboard->PushStdinChar(']'); guard_result="ok";
        } else if(command=="planner" && program_state_==ProgramState::CONTROL && !operator_state.play && !keyboard->use_planner) {
          keyboard->guard_planner_start_time=sonic_guard::now();
          keyboard->PushStdinChar('\n'); guard_result="ok";
        } else if(command=="idle" && keyboard->use_planner) {
          deadman.cancel(); keyboard->PushStdinChar('r'); guard_result="ok";
        } else if(command=="move" && keyboard->use_planner && !keyboard->guard_planner_start_pending && program_state_==ProgramState::CONTROL && argument>=0 && argument<4) {
          keyboard->planner_use_movement_mode=LocomotionMode::SLOW_WALK;
          keyboard->planner_use_movement_speed=0.15;
          deadman.refresh(argument,stamp,sonic_guard::now()); guard_result="ok";
        } else if(command=="motion" && program_state_==ProgramState::CONTROL && !keyboard->use_planner && !operator_state.play && argument>=0 && argument<(int)motion_reader_.motions.size()) {
          std::lock_guard<std::mutex> lock(current_motion_mutex_);
          auto next=motion_reader_.GetMotionShared(argument);
          bool close=true;
          for(int j=0;j<29;++j) if(std::abs(current_motion_->JointPositions(current_frame_)[j]-next->JointPositions(0)[j])>.20) close=false;
          if(close) {
            motion_reader_.current_motion_index_=argument; current_motion_=next;
            current_frame_=0; operator_state.play=true; guard_result="ok";
          } else { guard_result="rejected: reference discontinuity"; }
        }
      }
      if(keyboard->use_planner) keyboard->PushStdinChar(deadman.active(sonic_guard::now()) ? "ws,."[deadman.direction] : 'r');
'''
STATUS = r'''
      {
        std::lock_guard<std::mutex> lock(current_motion_mutex_);
        guard_.status(guard_seq,guard_result,static_cast<int>(program_state_),
                      motion_reader_.current_motion_index_,current_frame_,operator_state.play,keyboard->use_planner);
      }
'''
GATE = r'''
      auto command_stamp=motor_command_buffer_.GetDataWithTime();
      auto state_stamp=low_state_buffer_.GetDataWithTime();
      bool valid=false;
      if(mc && state_stamp.data) {
        auto now=std::chrono::steady_clock::now();
        valid=guard_.validate(*mc,*state_stamp.data,
            std::chrono::duration<double>(now-command_stamp.timestamp).count(),
            std::chrono::duration<double>(now-state_stamp.timestamp).count());
      } else if(mc) { guard_.trip("Missing robot state"); }
'''
DAMP = r'''
        if(!valid) {
          for(size_t i=0;i<G1_NUM_MOTOR;++i) {
            auto& m=dds_low_command.motor_cmd().at(i);
            m.tau()=0; m.q()=0; m.dq()=0; m.kp()=0; m.kd()=8;
          }
        }
'''


def patch(source, destination):
    source, destination=Path(source),Path(destination)
    # Strict source hashes prevent silent use against an unaudited upstream revision.
    pinned=json.loads((ROOT/'sonic_laptop/upstream-lock.json').read_text())
    for path, digest in pinned.items():
        if hashlib.sha256((source/path).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Upstream differs from audited source: {path}')
    if destination.exists(): raise ValueError('Build tree already exists; use a new --output or remove the old generated tree explicitly')
    destination.mkdir(parents=True)
    for p in source.iterdir():
        if p.name in ('build','target','logs') or p.name.startswith('.'): continue
        if p.name in ('policy','planner','reference','g1'):
            (destination/p.name).symlink_to(p.resolve(),target_is_directory=True)
        elif p.is_dir(): shutil.copytree(p,destination/p.name)
        else: shutil.copy2(p,destination/p.name)
    model=load_model(source.parent)
    (destination/'guard-model.json').write_text(json.dumps(model,indent=2))
    include=destination/REL/'include'
    (include/'guard_model.hpp').write_text(generate_model(model))
    shutil.copyfile(ROOT/'sonic_laptop/native/guard.hpp',include/'sonic_guard.hpp')
    shutil.copyfile(ROOT/'sonic_laptop/native/hand_damping.hpp',include/'hand_damping.hpp')
    cpp=destination/REL/'src/g1_deploy_onnx_ref.cpp'
    s=cpp.read_text()
    s='#include "sonic_guard.hpp"\n'+s
    s=replace_once(s,'    Dex3Hands dex3_hands_;','    sonic_guard::Guard guard_;\n    Dex3Hands dex3_hands_;')
    s=replace_once(s,'      dex3_hands_.initialize("");','      dex3_hands_.initialize(""); // Publishers reserved for latched damping only.')
    s=replace_once(s,'      dex3_hands_.writeOnce();','      if(guard_.fault.load()) dex3_hands_.writeDampingOnly();')
    # Delay release until all models and input/output interfaces are initialized.
    start=s.index('      // try to shutdown motion control-related service')
    end=s.index('      // create publisher',start)
    release=s[start:end]
    s=s[:start]+s[end:]
    # Require heartbeat before releasing the manufacturer controller; bounded attempts.
    release=release.replace('while (msc_->CheckMode(form, name), !name.empty()) {',
        'for (int attempt=0; ; ++attempt) {\n        if(!guard_.lease()) throw std::runtime_error("UI lease absent before release");\n        if(msc_->CheckMode(form,name)!=0) throw std::runtime_error("Cannot read robot mode");\n        if(name.empty()) break;\n        if(attempt>=3) throw std::runtime_error("Cannot release robot mode");')
    anchor='    bool InitControl() {'
    s=replace_once(s,anchor,anchor+'\n      if(!low_state_buffer_.GetDataWithTime().data) return false;\n      static bool released=false;\n      if(!released) {\n        if(!guard_.lease()) return false;\n        auto initial=low_state_buffer_.GetDataWithTime();\n        MotorCommand check{};\n        for(int j=0;j<29;++j) check.q_target[j]=initial.data->motor_state()[j].q();\n        if(!guard_.validate(check,*initial.data,0,std::chrono::duration<double>(std::chrono::steady_clock::now()-initial.timestamp).count())) return false;\n        if(!std::getenv("SONIC_GUARD_SIM")) {\n'+release+'\n        }\n        released=true;\n      }')
    a=s.index('    bool InitControl()'); b=s.index('    bool CheckSafety()',a)
    section=s[a:b].replace('default_angles[i]','guard_.init[i]')
    section=section.replace('    bool InitControl() {','    bool InitControl() {\n      try {',1)
    end=section.rfind('      return true;\n    }')
    assert end>=0
    section=section[:end]+section[end:].replace('      return true;\n    }',
        '      return true;\n      } catch(const std::exception& error) {\n        guard_.trip(std::string("Setup failed: ")+error.what()); return false;\n      }\n    }',1)
    s=s[:a]+section+s[b:]
    s=replace_once(s,'      if (operator_state.stop) { return; }\n\n      switch (program_state_)',
        '      if (operator_state.stop || guard_.fault.load()) { return; }\n\n      switch (program_state_)')
    s=replace_once(s,'      if (mc) {\n        for (size_t i = 0;',GATE+'\n      if (mc) {\n        for (size_t i = 0;')
    s=replace_once(s,'        dds_low_command.crc() =',DAMP+'\n        dds_low_command.crc() =')
    s=replace_once(s,'            current_frame_ = 0; // Reset to beginning','            current_frame_ = current_motion_->timesteps - 1; // Hold final standing reference')
    s=replace_once(s,'            reinitialize_heading_ = true;\n            std::cout << "Reset to frame 0." << std::endl;',
        '            std::cout << "Holding final reference; SONIC remains active." << std::endl;')
    s=replace_once(s,'      // Update input interface (poll for new data)',INPUT+'\n      // Update input interface (poll for new data)')
    s=replace_once(s,'      if (playback_input_file_ && program_state_ == ProgramState::CONTROL)',STATUS+'\n      if (playback_input_file_ && program_state_ == ProgramState::CONTROL)')
    # Waiting for the operator is an active stance, not a stalled policy thread.
    anchor='          if (operator_state.start) {'
    s=replace_once(s,anchor,'          if(auto hold=motor_command_buffer_.GetDataWithTime().data) motor_command_buffer_.SetData(*hold);\n'+anchor)
    a=s.index('    void Control() {')
    s=s[:a]+s[a:].replace('operator_state.stop = true;', 'guard_.trip("Upstream control/sensor/inference failure");')
    s=s.replace('    void LowCommandWriter() {','    void LowCommandWriter() {\n      guard_.lease(); // Also service stop before any body command exists.')
    # Remove all automatic finger gestures; only the dedicated damping path publishes.
    for action in ('open','close'):
        for side in ('true','false'):
            s=s.replace(f'        dex3_hands_.{action}({side});','        // No automatic finger motion.')
    for anchor,label in [
        ('      // Initialize control policy','Loading SONIC policy / building GPU engine'),
        ('        std::cout << "Initializing encoder..."','Loading motion encoder / building GPU engine'),
        ('        planner_ = std::make_unique<LocalMotionPlannerTensorRT>','Loading locomotion planner / building GPU engine'),
        ('      input_thread_ptr_ = CreateRecurrentThreadEx','Models ready; waiting for robot state'),
    ]:
        at=s.index(anchor)
        s=s[:at]+'      std::cout << "[Setup] '+label+'" << std::endl;\n'+s[at:]
    s=s.replace('        released=true;', '        std::cout << "[Setup] Handover complete; ramping to idle pose" << std::endl;\n        released=true;')
    s=s.replace('std::cout << "Init Done"', 'std::cout << "[Setup] Pose ramp complete; checking joint settling"')
    cpp.write_text(s)
    hands=include/'dex3_hands.hpp'
    h=hands.read_text()
    h=replace_once(h,'    // Returns a snapshot of the latest state for the requested hand.',
        '''    // Bypass position smoothing and normal command buffers on a latched stop.
    void writeDampingOnly() {
        unitree_hg::msg::dds_::HandCmd_ command;
        sonic_guard::hand_damping(command);
        if(left_.publisher) left_.publisher->Write(command);
        if(right_.publisher) right_.publisher->Write(command);
    }

    // Returns a snapshot of the latest state for the requested hand.''')
    hands.write_text('#include "hand_damping.hpp"\n'+h)

    p=include/'input_interface/input_interface.hpp'
    s=p.read_text().replace('return read(STDIN_FILENO, &ch, 1) > 0;', 'return false; // Only guarded IPC may enqueue commands.')
    p.write_text(s)
    p=include/'input_interface/keyboard_handler.hpp'
    s=replace_once(p.read_text(),'const double momentum_decay_rate = 0.999;', 'const double momentum_decay_rate = 0.80;')
    s=s.replace('      tcgetattr(STDIN_FILENO, &old_termios_);', '      old_termios_ = {};')
    s=s.replace('      tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);', '      // No terminal device in guarded mode.')
    s=s.replace('      tcsetattr(STDIN_FILENO, TCSANOW, &old_termios_);', '      // No terminal device in guarded mode.')
    s=s.replace('              std::cout << "Emergency Stop! Movement momentum reset." << std::endl;', '              // Guarded idle is refreshed continuously.')
    s=replace_once(s,'planner_use_movement_speed = std::max(planner_use_movement_speed, 0.2);',
                   'planner_use_movement_speed = std::max(planner_use_movement_speed, 0.1);')
    # Upstream blocks Input for up to five seconds during planner startup. Keep
    # input/stop/deadman servicing live while the planner thread initializes.
    s=replace_once(s,'    bool use_planner = false;',
        '    bool guard_planner_start_pending = false;\n    double guard_planner_start_time = 0;\n    bool use_planner = false;')
    begin=s.index('          // Wait for planner to be initialized with timeout (5 seconds)')
    end=s.index('        } else {\n            std::cout << "Planner disabled"',begin)
    s=s[:begin]+'          guard_planner_start_pending = true;\n'+s[end:]
    s=replace_once(s,'      if (has_planner && planner_state.enabled && planner_state.initialized) {',
        '''      if (guard_planner_start_pending && planner_state.enabled && planner_state.initialized) {
          std::lock_guard<std::mutex> lock(current_motion_mutex);
          if(current_motion && current_motion->name == "planner_motion" && current_motion->timesteps > 0) {
              guard_planner_start_pending = false;
              operator_state.play = true;
          }
      }
      if (has_planner && planner_state.enabled && planner_state.initialized) {''')
    # Explicit deadman forces planner idle after 150 ms without a fresh movement request.
    p.write_text(s)
    p=include/'motion_data_reader.hpp'
    s=replace_once(p.read_text(),'      for (const auto& motion_name : motion_names) {',
                   '      std::sort(motion_names.begin(),motion_names.end());\n      for (const auto& motion_name : motion_names) {')
    p.write_text('#include <algorithm>\n'+s)
    p=destination/'CMakeLists.txt'
    p.write_text(p.read_text().replace('-O3 -ffast-math','-O3 -fno-fast-math'))
    (destination/'GUARDED_BUILD').write_text('sonic-laptop-v3\n')
    return destination

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--upstream',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); print(patch(args.upstream/'gear_sonic_deploy',args.output))
