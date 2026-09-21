// hand_test.cpp
#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <atomic>
#include <iostream>
#include <mutex>
#include <unistd.h>

using namespace unitree::robot;
using MotorCmds   = unitree_go::msg::dds_::MotorCmds_;
using MotorStates = unitree_go::msg::dds_::MotorStates_;

std::mutex g_mtx; 
MotorStates g_state;
std::atomic<bool> g_have_state{false};

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <network-interface>\n";
        return 1;
    }
    ChannelFactory::Instance()->Init(0, argv[1]);

    auto sub = std::make_shared<ChannelSubscriber<MotorStates>>("rt/inspire/state");
    sub->InitChannel([](const void *msg) {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_state = *(const MotorStates *)msg;
        g_have_state.store(true);
    });

    auto pub = std::make_shared<ChannelPublisher<MotorCmds>>("rt/inspire/cmd");
    pub->InitChannel();

    // Confirm the service is alive BEFORE publishing anything.
    for (int i = 0; i < 25 && !g_have_state.load(); i++) {
        std::cout << "waiting for rt/inspire/state.." << std::endl;
        usleep(200000);
    }

    if (!g_have_state.load()) {
        std::cerr << "no hand state - is dfx_inspire_service running?\n";
        return 1;
    }

    // Read where the fingers are now
    float q0[12]; 
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        for (int i = 0; i < 12; i++) q0[i] = g_state.states()[i].q();
    }
    std::cout << "start R:";
    for (int i = 0; i < 6; i++) std::cout << " " << q0[i];
    std::cout << "\nstart L:";
    for (int i = 6; i < 12; i++) std::cout << " " << q0[i];
    std::cout << std::endl;

    MotorCmds cmd;
    cmd.cmds().resize(12); // required before first write

    // Right index finger only, slow ramp from its current value to half open. 
    const int JOINT = 3; 
    const float target = 0.5f;

    for (int step = 0; step <= 200; step++) {
        float r = step / 200.0f;
        for (int i = 0; i < 12; i++) cmd.cmds()[i].q() = q0[i];   // everything holds
        cmd.cmds()[JOINT].q() = q0[JOINT] + (target - q0[JOINT]) * r;
        pub->Write(cmd);

        if (step % 40 == 0) {
            std::lock_guard<std::mutex> lock(g_mtx);
            std::cout << "cmd=" << cmd.cmds()[JOINT].q()
                      << "  meas=" << g_state.states()[JOINT].q() << std::endl;
        }
        usleep(20000);              // 50 Hz
    }
    std::cout << "done" << std::endl;
    return 0;
}
