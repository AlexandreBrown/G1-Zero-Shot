#pragma once
// Dex3's existing driver uses kd=0.1 for its compliant joint commands.
// No position control, desired velocity, or feed-forward torque is applied.
namespace sonic_guard {
template<class HandCommand> void hand_damping(HandCommand& command) {
    command.motor_cmd().resize(7);
    for(int i=0;i<7;++i) {
        auto& motor=command.motor_cmd()[i];
        motor.mode(static_cast<unsigned char>(i | (0x01 << 4)));
        motor.q(0); motor.dq(0); motor.tau(0); motor.kp(0); motor.kd(.1);
    }
}
}
