#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <Eigen/Dense>
#include <array>
#include <atomic>
#include <iostream>
#include <mutex>
#include <unistd.h>

using namespace unitree::robot;
using LowState = unitree_hg::msg::dds_::LowState_;

constexpr int NJ = 29;

const std::array<float, NJ> DEFAULT_Q = {
    -0.1f, 0, 0, 0.3f, -0.2f, 0,   -0.1f, 0, 0, 0.3f, -0.2f, 0,
     0, 0, 0,
     0.35f,  0.18f, 0, 0.87f, 0, 0, 0,
     0.35f, -0.18f, 0, 0.87f, 0, 0, 0};

std::mutex g_mtx;
LowState g_state;
std::atomic<bool> g_have{false};

int main(int argc, char **argv) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <iface>\n"; return 1; }
    ChannelFactory::Instance()->Init(0, argv[1]);

    auto sub = std::make_shared<ChannelSubscriber<LowState>>("rt/lowstate");
    sub->InitChannel([](const void *m) {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_state = *(const LowState *)m;
        g_have.store(true);
    }, 1);

    while (!g_have.load()) { std::cout << "waiting..\n"; usleep(200000); }

    LowState s;
    { std::lock_guard<std::mutex> lk(g_mtx); s = g_state; }

    std::vector<float> obs;
    obs.reserve(98);

    // [0:3] base_ang_vel — gyroscope, body frame already
    for (int i = 0; i < 3; i++) obs.push_back(s.imu_state().gyroscope()[i]);

    // [3:6] projected_gravity — world -Z rotated into base frame
    auto q = s.imu_state().quaternion();              // w, x, y, z
    Eigen::Quaternionf quat(q[0], q[1], q[2], q[3]);
    Eigen::Vector3f g_b = quat.toRotationMatrix().transpose() * Eigen::Vector3f(0, 0, -1);
    for (int i = 0; i < 3; i++) obs.push_back(g_b[i]);

    // [6:9] velocity_commands — zero for now, later from the command bus
    obs.push_back(0.0f); obs.push_back(0.0f); obs.push_back(0.0f);

    // [9:11] gait_phase — period 0.6 s
    float t = 0.0f;                                    // policy time, 0 at reset
    float phase = 2.0f * M_PI * t / 0.6f;
    obs.push_back(std::sin(phase));
    obs.push_back(std::cos(phase));

    // [11:40] joint_pos_rel
    for (int i = 0; i < NJ; i++) obs.push_back(s.motor_state()[i].q() - DEFAULT_Q[i]);

    // [40:69] joint_vel_rel
    for (int i = 0; i < NJ; i++) obs.push_back(s.motor_state()[i].dq());

    // [69:98] last_action — raw network output from previous step
    for (int i = 0; i < NJ; i++) obs.push_back(0.0f);

    const char *labels[] = {"ang_vel", "proj_grav", "vel_cmd", "gait_phase",
                            "joint_pos_rel", "joint_vel_rel", "last_action"};
    const int bounds[] = {0, 3, 6, 9, 11, 40, 69, 98};
    for (int b = 0; b < 7; b++) {
        std::cout << labels[b] << " [" << bounds[b] << ":" << bounds[b+1] << "]:";
        for (int i = bounds[b]; i < bounds[b+1]; i++) std::cout << " " << obs[i];
        std::cout << "\n";
    }
    std::cout << "total: " << obs.size() << std::endl;
    return 0;
}