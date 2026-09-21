#include "cnpy.h"
#include <iostream>

int main(int argc, char **argv) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <motion.npz>\n"; return 1; }
    cnpy::npz_t npz = cnpy::npz_load(argv[1]);

    for (auto &kv : npz) {
        std::cout << kv.first << "  shape:";
        for (auto d : kv.second.shape) std::cout << " " << d;
        std::cout << "  word_size=" << kv.second.word_size << "\n";
    }

    auto &jp = npz["joint_pos"];
    const float *q = jp.data<float>();
    size_t NF = jp.shape[0], NJ = jp.shape[1];
    std::cout << "frame 0 joint_pos:";
    for (size_t i = 0; i < NJ; i++) std::cout << " " << q[i];
    std::cout << "\n";

    auto &bq = npz["body_quat_w"];
    const float *quat = bq.data<float>();
    size_t NB = bq.shape[1];
    std::cout << "frame 0 body 0 quat:";
    for (size_t c = 0; c < 4; c++) std::cout << " " << quat[(0 * NB + 0) * 4 + c];
    std::cout << "\n";

    std::cout << "fps = " << npz["fps"].data<float>()[0] << "\n";
    std::cout << "frames = " << NF << "  duration = " << NF / npz["fps"].data<float>()[0] << " s\n";
    return 0;
}