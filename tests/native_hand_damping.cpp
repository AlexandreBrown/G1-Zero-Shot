#include <cassert>
#include <cmath>
#include <unitree/idl/hg/HandCmd_.hpp>
#include "hand_damping.hpp"
int main(){
    unitree_hg::msg::dds_::HandCmd_ command;
    command.motor_cmd().resize(7);
    for(auto& m:command.motor_cmd()) {m.kp(99);m.kd(99);m.tau(99);m.q(99);m.dq(99);}
    sonic_guard::hand_damping(command);
    for(int i=0;i<7;i++){
        const auto& m=command.motor_cmd()[i];
        assert(m.mode()==(i|0x10));assert(m.kp()==0);assert(m.tau()==0);assert(m.q()==0);assert(m.dq()==0);
        assert(std::abs(m.kd()-.1)<1e-6);
    }
}
