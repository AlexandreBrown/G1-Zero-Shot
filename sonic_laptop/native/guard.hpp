#pragma once
// Deliberately independent of Unitree/CUDA, so the final publish gate is CPU-testable.
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <time.h>
#include "guard_model.hpp"
namespace sonic_guard {
inline double now() { timespec t{}; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
class Deadman {
public:
    double deadline=0;
    int direction=0;
    bool refresh(int value,double stamp,double t) {
        if(value<0||value>3||!std::isfinite(stamp)||t-stamp>.15||stamp>t+.01) return false;
        direction=value; deadline=stamp+.15; return true;
    }
    void cancel() { deadline=0; }
    bool active(double t) const { return deadline>t; }
};
class Guard {
public:
    std::string dir;
    std::array<double,29> init{};
    std::atomic<bool> fault{false}, waiting{false}, setup_ready{false};
    std::atomic<double> input_tick{0}, setup_error{0}, setup_speed{0}, settled_seconds{0};
    double settled_since=0;
    std::mutex mutex, lease_mutex;
    std::string reason;
    double checked=0, previous_time=0;
    std::array<double,29> previous{};
    explicit Guard() {
        auto p=std::getenv("SONIC_GUARD_DIR");
        if(!p) throw std::runtime_error("Use the guarded launcher: SONIC_GUARD_DIR missing");
        dir=p;
        std::ifstream f(dir+"/init.txt");
        for(auto& q:init) if(!(f>>q)||!std::isfinite(q)) throw std::runtime_error("Invalid init pose");
        for(int i=0;i<29;++i) if(init[i]<lower[i]||init[i]>upper[i]) throw std::runtime_error("Init limit violation");
        if(clearance(init)<minimum_clearance) throw std::runtime_error("Init hand clearance violation");
    }
    void trip(const std::string& why) {
        std::lock_guard<std::mutex> lock(mutex);
        if(!fault.load()) { reason=why; fault.store(true); std::cerr<<"SONIC GUARD LATCHED: "<<why<<std::endl; }
    }
    bool lease() {
        std::lock_guard<std::mutex> lock(lease_mutex);
        double t=now();
        if(t-checked>=.02) {
            checked=t;
            double heartbeat=0; int run=0;
            std::ifstream f(dir+"/lease");
            if(!(f>>heartbeat>>run)||run!=1||!std::isfinite(heartbeat)||t-heartbeat>.30||heartbeat>t+.01)
                trip("UI stop / heartbeat lost (>300 ms)");
        }
        return !fault.load();
    }
    template<class Command, class State>
    bool validate(const Command& cmd, const State& state, double command_age, double state_age) {
        if(!lease()) return false;
        if(input_tick.load()>0 && now()-input_tick.load()>.15) {trip("Input loop stalled (>150 ms)"); return false;}
        if(command_age>.10||state_age>.10) {trip("Control or robot state stale (>100 ms)"); return false;}
        std::array<double,29> target{}, measured{};
        double t=now();
        for(int i=0;i<29;++i) {
            target[i]=cmd.q_target[i]; measured[i]=state.motor_state()[i].q();
            double dq=state.motor_state()[i].dq();
            if(!std::isfinite(target[i])||!std::isfinite(measured[i])||!std::isfinite(dq)||
               !std::isfinite(cmd.kp[i])||!std::isfinite(cmd.kd[i])||!std::isfinite(cmd.tau_ff[i])||!std::isfinite(cmd.dq_target[i])) {
                trip("Nonfinite motor value"); return false;
            }
            if(target[i]<lower[i]||target[i]>upper[i]||measured[i]<lower[i]-.03||measured[i]>upper[i]+.03||
               std::abs(dq)>8||std::abs(cmd.dq_target[i])>4||std::abs(cmd.tau_ff[i])>torque[i]||
               cmd.kp[i]<0||cmd.kp[i]>400||cmd.kd[i]<0||cmd.kd[i]>20) {
                trip("Joint/gain/velocity/torque limit at motor "+std::to_string(i)); return false;
            }
            // Updates arrive at 50 Hz, writer at 500 Hz: allow one bounded 20 ms step.
            if(previous_time && std::abs(target[i]-previous[i])>.20) {
                trip("Target step >0.20 rad at motor "+std::to_string(i)); return false;
            }
        }
        if(clearance(target)<minimum_clearance||clearance(measured)<minimum_clearance) {
            trip("Hand/body envelope clearance <25 mm"); return false;
        }
        double error=0, speed=0;
        for(int i=0;i<29;++i) {
            error=std::max(error,std::abs(measured[i]-init[i]));
            speed=std::max(speed,std::abs(double(state.motor_state()[i].dq())));
        }
        setup_error.store(error); setup_speed.store(speed);
        bool settled=waiting.load();
        for(int i=0;i<29;++i)
            if(std::abs(measured[i]-init[i])>.12 || std::abs(state.motor_state()[i].dq())>.3) settled=false;
        if(!settled) settled_since=0;
        else if(!settled_since) settled_since=t;
        settled_seconds.store(settled_since>0 ? t-settled_since : 0);
        setup_ready.store(settled_since>0 && t-settled_since>=2);
        previous=target; previous_time=t; return true;
    }
    void status(unsigned long seq, const std::string& result, int state, int index, int frame, bool play, bool planner) {
        static double last=0; if(now()-last<.05) return; last=now();
        std::lock_guard<std::mutex> lock(mutex);
        std::ofstream out(dir+"/status.tmp");
        out<<seq<<' '<<state<<' '<<index<<' '<<frame<<' '<<play<<' '<<planner<<' '<<fault.load()<<' '<<setup_ready.load()<<'\n'
           <<result<<'\n'<<reason<<'\n'
           <<"Joint error "<<setup_error.load()<<" rad / 0.12; speed "<<setup_speed.load()
           <<" rad/s / 0.30; settled "<<settled_seconds.load()<<" s / 2.0\n"; out.close();
        std::rename((dir+"/status.tmp").c_str(),(dir+"/status").c_str());
    }
};
}
