// Exercise the actual guarded SONIC IPC branch and CSV loader on CPU.
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
    std::shared_ptr<const MotionSequence> current_motion_;
    int current_frame_=0;
    OperatorState operator_state;
    std::mutex current_motion_mutex_;
    unsigned long seq=0;
    void command(const std::string& name,int value=0,double age=0) {
        std::ofstream f(guard_.dir+"/request");
        f<<++seq<<' '<<name<<' '<<value<<' '<<std::setprecision(17)<<sonic_guard::now()-age<<'\n';
    }
    void tick() {
#include "guarded_input.inc"
    }
};
int main(int argc,char** argv) {
    setenv("SONIC_GUARD_DIR",argv[1],1);
    Harness h;
    assert(h.motion_reader_.ReadFromCSV(std::string(argv[1])+"/motions"));
    assert(h.motion_reader_.motions.size()==5);
    assert(h.motion_reader_.motions[0]->name=="0000");
    h.current_motion_=h.motion_reader_.GetMotionShared(0);
    h.command("motion",1);h.tick();assert(!h.operator_state.play); // suspended
    h.program_state_=Harness::ProgramState::CONTROL;
    h.command("motion",1,.2);h.tick();assert(!h.operator_state.play); // expired
    h.command("motion",1);h.tick();assert(h.operator_state.play);
    assert(h.current_motion_->name=="0001");
    assert(h.current_motion_->GetNumJoints()==29);
    assert(h.current_motion_->GetNumBodies()==1);
    h.command("motion",2);h.tick();assert(h.current_motion_->name=="0001"); // no overlap
    h.operator_state.play=false;
    h.current_frame_=h.current_motion_->timesteps-1;
    h.command("motion",2);h.tick();assert(h.operator_state.play);
    assert(h.current_motion_->name=="0002"); // matching endpoints permit transition
    h.operator_state.play=false;
    h.guard_.trip("test stop");
    h.command("motion",3);h.tick();assert(!h.operator_state.play);
}
