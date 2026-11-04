// Runs the actual generated IPC block and patched upstream keyboard, without DDS/GPU.
#include <mutex>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include "sonic_guard.hpp"
#include "input_interface/keyboard_handler.hpp"
class Harness {
public:
    enum class ProgramState { INIT, WAIT_FOR_CONTROL, CONTROL };
    ProgramState program_state_=ProgramState::WAIT_FOR_CONTROL;
    sonic_guard::Guard guard_;
    std::unique_ptr<InputInterface> input_interface_=std::make_unique<SimpleKeyboard>();
    MotionDataReader motion_reader_;
    std::shared_ptr<const MotionSequence> current_motion_=std::make_shared<MotionSequence>();
    int current_frame_=0;
    OperatorState operator_state;
    std::mutex current_motion_mutex_;
    DataBuffer<HeadingState> heading;
    DataBuffer<MovementState> movement;
    PlannerState planner;
    bool reinit=false, temperature=false;
    unsigned long seq=0;
    SimpleKeyboard& keyboard(){return *static_cast<SimpleKeyboard*>(input_interface_.get());}
    void command(const std::string& name,int value=0,double age=0){
        std::ofstream f(guard_.dir+"/request");
        f<<++seq<<' '<<name<<' '<<value<<' '<<std::setprecision(17)<<sonic_guard::now()-age<<'\n';
    }
    void tick(){
#include "guarded_input.inc"
        input_interface_->update();
        input_interface_->handle_input(motion_reader_,current_motion_,current_frame_,operator_state,
            reinit,heading,true,planner,movement,current_motion_mutex_,temperature);
    }
};
int main(int argc,char** argv){
    setenv("SONIC_GUARD_DIR",argv[1],1);
    Harness h;
    h.command("grounded_start");h.tick();assert(!h.operator_state.start);
    h.guard_.setup_ready.store(true);h.command("grounded_start");h.tick();assert(h.operator_state.start);
    h.program_state_=Harness::ProgramState::CONTROL;
    h.command("planner");double begin=sonic_guard::now();h.tick();
    assert(sonic_guard::now()-begin<.15);assert(h.keyboard().guard_planner_start_pending);assert(!h.operator_state.play);
    h.command("move");h.tick();assert(h.keyboard().movement_momentum==0);
    auto motion=std::make_shared<MotionSequence>();motion->name="planner_motion";motion->timesteps=10;
    h.current_motion_=motion;h.planner.initialized=true;h.tick();assert(h.operator_state.play);
    for(int argument=0;argument<8;argument++){
        int direction=argument%4;  // 4-7: same directions at the fine positioning speed
        h.command("move",argument);h.tick();auto value=h.movement.GetDataWithTime().data;
        assert(value && value->locomotion_mode==static_cast<int>(LocomotionMode::SLOW_WALK));
        assert(std::abs(value->movement_speed-(argument<4 ? .4 : .2))<1e-6);
        assert(value->movement_direction[direction<2 ? 0 : 1]*(direction%2 ? -1 : 1)>.99);
        h.command("idle");h.tick();value=h.movement.GetDataWithTime().data;
        assert(value->locomotion_mode==static_cast<int>(LocomotionMode::IDLE));
        assert(value->movement_direction[0]==0 && value->movement_direction[1]==0);
    }
    h.command("move",8);h.tick();
    assert(h.movement.GetDataWithTime().data->locomotion_mode==static_cast<int>(LocomotionMode::IDLE));
    h.command("move");h.tick();std::this_thread::sleep_for(std::chrono::milliseconds(160));h.tick();
    assert(h.movement.GetDataWithTime().data->locomotion_mode==static_cast<int>(LocomotionMode::IDLE));
    h.command("move",0,.2);h.tick();assert(h.keyboard().movement_momentum==0);
    // reference: refused while a walk is live; otherwise leaves the planner through a blend from the planner's
    // frame into clip 0, and no walk/stop key follows the one that leaves.
    auto ready=std::make_shared<MotionSequence>();ready->name="ready";ready->ReserveCapacity(50);ready->timesteps=50;
    for(int f=0;f<50;f++) for(int j=0;j<29;j++) ready->JointPositions(f)[j]=.5;
    h.motion_reader_.motions={ready};motion->ReserveCapacity(10);
    h.command("move",0);h.tick();h.command("reference");h.tick();assert(h.keyboard().use_planner);
    h.command("idle");h.tick();std::this_thread::sleep_for(std::chrono::milliseconds(160));h.tick();
    h.command("reference");h.tick();assert(!h.keyboard().use_planner);
    assert(h.current_motion_->name=="planner_exit_blend" && h.operator_state.play && h.current_frame_==0);
    assert(h.motion_reader_.current_motion_index_==0);
    assert(h.current_motion_->JointPositions(0)[3]==0 && std::abs(h.current_motion_->JointPositions(49)[3]-.5)<1e-9);
    for(int f=1;f<50;f++)
        assert(std::abs(h.current_motion_->JointPositions(f)[3]-h.current_motion_->JointPositions(f-1)[3])<.02);
    h.tick();assert(!h.keyboard().use_planner && h.current_motion_->name=="planner_exit_blend");
    h.command("planner");h.tick();assert(!h.keyboard().use_planner);  // not while the blend plays
    h.operator_state.play=false;h.command("planner");h.tick();assert(h.keyboard().use_planner);
    h.keyboard().guard_planner_start_pending=true;h.keyboard().guard_planner_start_time=sonic_guard::now()-6;
    h.tick();assert(h.guard_.fault.load());
}
