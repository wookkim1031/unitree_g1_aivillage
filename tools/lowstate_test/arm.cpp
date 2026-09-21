#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <atomic>
#include <iostream>
#include <unistd.h>
#include <cmath>

using namespace unitree::robot;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_; 

const int G1_NUM_MOTOR = 29; 

// catches the mode_moachine field out of that LowState. 
std::atomic<uint8_t> g_mode_machine{0};
// first LowState has arrived latch
// g_have_state is true, then stamps cmd.mode_machien() = g_mode_machine
std::atomic<bool> g_have_state{false}; 

std::atomic<float> g_q[G1_NUM_MOTOR];

// computes checksum Unitree appends to LowCmd so the firmware can reject corrupted packets
uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    uint32_t xbit = 1 << 31;
    uint32_t data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) { CRC32 <<= 1; CRC32 ^= dwPolynomial; }
      else CRC32 <<= 1;
      if (data & xbit) CRC32 ^= dwPolynomial;
      xbit >>= 1;
    }
  }
  return CRC32;
}

void StateHandler(const void *msg) {
    auto *s = (const LowState *)msg;
    g_mode_machine.store(s->mode_machine());
    g_have_state.store(true);
    for (int i = 0; i < G1_NUM_MOTOR; i++) 
        g_q[i].store(s->motor_state()[i].q());
    g_have_state.store(true);
}

int main(int argc, char **argv) {
    ChannelFactory::Instance()->Init(0, argv[1]);
    
    ChannelSubscriberPtr<LowState> sub(new ChannelSubscriber<LowState>("rt/lowstate"));
    sub->InitChannel(StateHandler, 1);

    ChannelPublisherPtr<LowCmd> pub(new ChannelPublisher<LowCmd>("rt/lowcmd"));
    pub->InitChannel();
    
    // waiting for the robot to send anything
    // rt/lowstate: the robot publishes, you subscribe
    // rt/lowcmd: you publish, the robot subscribes 
    while (!g_have_state.load()) {
        std::cout << "waiting for lowstate.." << std::endl; 
        usleep(200000);
    }
    std::cout << "G1 type (mode_machine): "
              << unsigned(g_mode_machine.load()) << std::endl;

    float q_hold[G1_NUM_MOTOR];
    for (int i = 0; i < G1_NUM_MOTOR; i++) q_hold[i] = g_q[i].load();

    int n = 0; 
    
    while (true) {
        LowCmd cmd;
        cmd.mode_pr() = 0; // PR 
        cmd.mode_machine() = g_mode_machine.load(); // must echo
        double t = n * 0.002; // n counts iterations at 500 Hz
        
        for (int i = 0; i < G1_NUM_MOTOR; i++) {
          float kp = (i == 3 || i == 9) ? 100.0f : 40.0f;
          cmd.motor_cmd().at(i).mode() = 1;   // enable
          cmd.motor_cmd().at(i).q()    = q_hold[i];
          cmd.motor_cmd().at(i).dq()   = 0.0;
          cmd.motor_cmd().at(i).tau()  = 0.0;
          cmd.motor_cmd().at(i).kp()   = kp;
          cmd.motor_cmd().at(i).kd()   = 2.0;
        }
        cmd.motor_cmd().at(22).q() = q_hold[22] + 0.15 * std::sin(2.0 * M_PI * t / 4.0);           // shoulder pitch
        cmd.motor_cmd().at(25).q() = q_hold[25] + 0.15 * std::sin(2.0 * M_PI * t / 4.0 + 0.5);     // elbow
        cmd.crc() = Crc32Core((uint32_t *)&cmd, (sizeof(cmd) >> 2) - 1);
        pub->Write(cmd);
    
        if (++n % 500 == 0) std::cout << "sent " << n << " cmds" << std::endl;
        usleep(2000);  // 500 Hz
  }
}