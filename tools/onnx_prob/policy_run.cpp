#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <onnxruntime_cxx_api.h>
#include <Eigen/Dense>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <unistd.h>

#include <csignal>

#include <algorithm>
#include <string>

std::atomic<float> g_gain_scale{1.0f};
std::atomic<float> g_vx{0.0f}, g_vy{0.0f}, g_wz{0.0f};

using namespace unitree::robot;
using LowState = unitree_hg::msg::dds_::LowState_;
using LowCmd   = unitree_hg::msg::dds_::LowCmd_;

constexpr int NJ = 29;
constexpr int NOBS = 98;
constexpr float STEP_DT = 0.02f;
constexpr float GAIT_PERIOD = 0.6f;

const std::array<float, NJ> DEFAULT_Q = {
    -0.1f, 0, 0, 0.3f, -0.2f, 0,   -0.1f, 0, 0, 0.3f, -0.2f, 0,
     0, 0, 0,
     0.35f,  0.18f, 0, 0.87f, 0, 0, 0,
     0.35f, -0.18f, 0, 0.87f, 0, 0, 0};

const std::array<float, NJ> OFFSET = DEFAULT_Q;   // identical in deploy.yaml

const std::array<float, NJ> SCALE = {
    0.55f, 0.35f, 0.55f, 0.35f, 0.44f, 0.44f,
    0.55f, 0.35f, 0.55f, 0.35f, 0.44f, 0.44f,
    0.55f, 0.44f, 0.44f,
    0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.07f, 0.07f,
    0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.07f, 0.07f};

const std::array<float, NJ> STIFFNESS = {
    40.2f, 99.1f, 40.2f, 99.1f, 28.5f, 28.5f,
    40.2f, 99.1f, 40.2f, 99.1f, 28.5f, 28.5f,
    40.2f, 28.5f, 28.5f,
    14.3f, 14.3f, 14.3f, 14.3f, 14.3f, 16.8f, 16.8f,
    14.3f, 14.3f, 14.3f, 14.3f, 14.3f, 16.8f, 16.8f};

const std::array<float, NJ> DAMPING = {
    2.6f, 6.3f, 2.6f, 6.3f, 1.8f, 1.8f,
    2.6f, 6.3f, 2.6f, 6.3f, 1.8f, 1.8f,
    2.6f, 1.8f, 1.8f,
    0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 1.1f, 1.1f,
    0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 1.1f, 1.1f};

std::mutex g_mtx;
LowState g_state;
std::atomic<bool> g_have{false};
std::atomic<uint8_t> g_mode_machine{0};

std::mutex g_cmd_mtx;
std::array<float, NJ> g_q_des{};
std::atomic<bool> g_cmd_valid{false};
std::atomic<bool> g_running{true};
std::atomic<bool> g_writer_running{true};

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

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <iface> <policy.onnx>\n";
        return 1;
    }
    ChannelFactory::Instance()->Init(0, argv[1]);
    std::signal(SIGINT, [](int){ g_running.store(false); });

    auto sub = std::make_shared<ChannelSubscriber<LowState>>("rt/lowstate");
    sub->InitChannel([](const void *m) {
        auto *p = (const LowState *)m;
        std::lock_guard<std::mutex> lk(g_mtx);
        g_state = *p;
        g_mode_machine.store(p->mode_machine());
        g_have.store(true);
    }, 1);

    auto pub = std::make_shared<ChannelPublisher<LowCmd>>("rt/lowcmd");
    pub->InitChannel();

    Ort::Env ort_env(ORT_LOGGING_LEVEL_WARNING, "policy");
    Ort::SessionOptions opts;
    Ort::Session session(ort_env, argv[2], opts);
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name  = session.GetInputNameAllocated(0, alloc);
    auto out_name = session.GetOutputNameAllocated(0, alloc);
    const char *in_names[]  = {in_name.get()};
    const char *out_names[] = {out_name.get()};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    while (!g_have.load()) { std::cout << "waiting for lowstate..\n"; usleep(200000); }

    // 500 Hz writer — resends the last policy target between policy steps.
    std::thread writer([&]{
        LowCmd cmd{};
        cmd.mode_pr() = 0;
        while (g_writer_running.load()) {
            if (g_cmd_valid.load()) {
                std::array<float, NJ> q;
                { std::lock_guard<std::mutex> lk(g_cmd_mtx); q = g_q_des; }
                cmd.mode_machine() = g_mode_machine.load();
                float gs = g_gain_scale.load();
                for (int i = 0; i < NJ; i++) {
                    cmd.motor_cmd().at(i).mode() = 1;
                    cmd.motor_cmd().at(i).q()    = q[i];
                    cmd.motor_cmd().at(i).dq()   = 0.0f;
                    cmd.motor_cmd().at(i).tau()  = 0.0f;
                    cmd.motor_cmd().at(i).kp()   = STIFFNESS[i] * gs;
                    cmd.motor_cmd().at(i).kd()   = DAMPING[i] * gs;
                }
                cmd.crc() = Crc32Core((uint32_t *)&cmd, (sizeof(cmd) >> 2) - 1);
                pub->Write(cmd);
            }
            usleep(2000);
        }
    });

    std::thread keys([]{
    std::cout << "keys: w/s = forward/back, a/d = left/right, "
                 "q/e = turn, x = stop, Enter after each\n";
    std::string line;
    while (g_running.load() && std::getline(std::cin, line)) {
        float vx = g_vx.load(), vy = g_vy.load(), wz = g_wz.load();
        for (char ch : line) {
            switch (ch) {
                case 'w': vx += 0.1f; break;
                case 's': vx -= 0.1f; break;
                case 'a': vy += 0.1f; break;
                case 'd': vy -= 0.1f; break;
                case 'q': wz += 0.2f; break;
                case 'e': wz -= 0.2f; break;
                case 'x': vx = vy = wz = 0.0f; break;
            }
        }
        vx = std::clamp(vx, -0.5f, 1.0f);
        vy = std::clamp(vy, -0.5f, 0.5f);
        wz = std::clamp(wz, -1.0f, 1.0f);
        g_vx.store(vx); g_vy.store(vy); g_wz.store(wz);
        std::cout << "cmd: vx=" << vx << " vy=" << vy << " wz=" << wz << std::endl;
        }
    });

    std::array<float, NJ> last_action{};          // raw network output
    //float vx = 0.0f, vy = 0.0f, wz = 0.0f;        // command bus, later

    using clock = std::chrono::steady_clock;
    auto next = clock::now();
    
    int step = 0; 
    while (g_running.load()) {
        LowState s;
        { std::lock_guard<std::mutex> lk(g_mtx); s = g_state; }

        std::vector<float> obs;
        obs.reserve(NOBS);

        for (int i = 0; i < 3; i++) obs.push_back(s.imu_state().gyroscope()[i]);

        auto qd = s.imu_state().quaternion();     // w, x, y, z
        Eigen::Quaternionf quat(qd[0], qd[1], qd[2], qd[3]);
        Eigen::Vector3f g_b = quat.toRotationMatrix().transpose() * Eigen::Vector3f(0, 0, -1);
        for (int i = 0; i < 3; i++) obs.push_back(g_b[i]);

        obs.push_back(g_vx.load());
        obs.push_back(g_vy.load());
        obs.push_back(g_wz.load());

        float t = step * STEP_DT;
        float phase = 2.0f * M_PI * t / GAIT_PERIOD;
        obs.push_back(std::sin(phase));
        obs.push_back(std::cos(phase));

        for (int i = 0; i < NJ; i++) obs.push_back(s.motor_state()[i].q() - DEFAULT_Q[i]);
        for (int i = 0; i < NJ; i++) obs.push_back(s.motor_state()[i].dq());
        for (int i = 0; i < NJ; i++) obs.push_back(last_action[i]);

        int64_t dims[] = {1, NOBS};
        auto tensor = Ort::Value::CreateTensor<float>(mem, obs.data(), obs.size(), dims, 2);
        auto out = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
        const float *action = out[0].GetTensorData<float>();

        std::array<float, NJ> q_des{};
        for (int i = 0; i < NJ; i++) {
            last_action[i] = action[i];           // RAW, before offset/scale
            q_des[i] = OFFSET[i] + SCALE[i] * action[i];
        }

        { std::lock_guard<std::mutex> lk(g_cmd_mtx); g_q_des = q_des; }
        g_cmd_valid.store(true);

        if (step % 25 == 0) {
            float max_d = 0.0f; int arg = 0;
            for (int i = 0; i < NJ; i++) {
                float d = std::fabs(q_des[i] - s.motor_state()[i].q());
                if (d > max_d) { max_d = d; arg = i; }
            }
            std::cout << "step " << step << " t=" << t << "  q_des[0..5]:";
            for (int i = 0; i < 6; i++) std::cout << " " << q_des[i];
            std::cout << "  max|q_des-q|=" << max_d << " @" << arg
                      << "  grav=" << g_b[0] << "," << g_b[1] << "," << g_b[2]
                      << std::endl;
        }
        step++;
        next += std::chrono::microseconds(20000);
        std::this_thread::sleep_until(next);
    }

    std::cout << "\nfading out..." << std::endl;
    for (int i = 50; i >= 0; i--) {
        g_gain_scale.store(i / 50.0f);
        usleep(20000);                 // 1 s total
    }
    g_writer_running.store(false);
    writer.join();
    keys.detach();
    std::cout << "done" << std::endl;
    return 0;
}