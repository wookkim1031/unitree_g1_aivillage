#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <policy.onnx>\n";
        return 1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "probe");
    Ort::SessionOptions opts;
    Ort::Session session(env, argv[1], opts);
    Ort::AllocatorWithDefaultOptions alloc;

    auto shape_of = [&](bool input, size_t i) {
        auto info = input ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
        return info.GetTensorTypeAndShapeInfo().GetShape();
    };

    std::cout << "inputs: " << session.GetInputCount() << "\n";
    for (size_t i = 0; i < session.GetInputCount(); i++) {
        auto name = session.GetInputNameAllocated(i, alloc);
        std::cout << "  [" << i << "] " << name.get() << "  shape:";
        for (auto d : shape_of(true, i)) std::cout << " " << d;
        std::cout << "\n";
    }
    std::cout << "outputs: " << session.GetOutputCount() << "\n";
    for (size_t i = 0; i < session.GetOutputCount(); i++) {
        auto name = session.GetOutputNameAllocated(i, alloc);
        std::cout << "  [" << i << "] " << name.get() << "  shape:";
        for (auto d : shape_of(false, i)) std::cout << " " << d;
        std::cout << "\n";
    }

    // Feed zeros and print the output.
    auto in_shape = shape_of(true, 0);
    size_t n_in = 1;
    for (auto d : in_shape) n_in *= (d > 0 ? d : 1);   // batch dim may be -1

    std::vector<float> obs(n_in, 0.0f);
    std::vector<int64_t> dims;
    for (auto d : in_shape) dims.push_back(d > 0 ? d : 1);

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(mem, obs.data(), obs.size(),
                                                  dims.data(), dims.size());

    auto in_name  = session.GetInputNameAllocated(0, alloc);
    auto out_name = session.GetOutputNameAllocated(0, alloc);
    const char *in_names[]  = {in_name.get()};
    const char *out_names[] = {out_name.get()};

    auto out = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
    float *a = out[0].GetTensorMutableData<float>();
    size_t n_out = out[0].GetTensorTypeAndShapeInfo().GetElementCount();

    std::cout << "action (" << n_out << "):";
    for (size_t i = 0; i < n_out; i++) std::cout << " " << a[i];
    std::cout << std::endl;
    return 0;
}