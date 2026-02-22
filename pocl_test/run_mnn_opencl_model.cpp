#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNForwardType.h>
#include <iostream>
#include <dlfcn.h>
#include <memory>
#include <cmath>

int main(int argc, char** argv) {
    const char* model = (argc > 1) ? argv[1] : "tiny_matmul_add.mnn";

    // Load OpenCL backend plugin (built as libMNN_CL.so) so RuntimeCreator(type=3) is registered.
    void* h = dlopen("libMNN_CL.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        std::cerr << "[pocl_test] dlopen(libMNN_CL.so) failed: " << dlerror() << std::endl;
    }

    auto net = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model));
    if (!net) {
        std::cerr << "failed to load model: " << model << "\n";
        return 1;
    }

    MNN::ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_OPENCL;
    cfg.backupType = MNN_FORWARD_CPU; // strict mode handled by MNN_STRICT_OPENCL_NO_CPU_OP
    cfg.numThread = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER; // force BUFFER path on image-unsupported devices

    MNN::BackendConfig bcfg;
    bcfg.precision = MNN::BackendConfig::Precision_Normal;
    bcfg.power = MNN::BackendConfig::Power_Normal;
    bcfg.memory = MNN::BackendConfig::Memory_Normal;
    cfg.backendConfig = &bcfg;

    auto session = net->createSession(cfg);
    if (!session) {
        std::cerr << "createSession failed\n";
        return 2;
    }

    auto input = net->getSessionInput(session, "x");
    std::unique_ptr<MNN::Tensor> hostIn(new MNN::Tensor(input, MNN::Tensor::CAFFE));
    float* p = hostIn->host<float>();
    for (int i = 0; i < 4; i++) p[i] = float(i+1);
    input->copyFromHostTensor(hostIn.get());

    auto code = net->runSession(session);
    std::cerr << "runSession rc=" << (int)code << "\n";

    auto output = net->getSessionOutput(session, "y");
    std::unique_ptr<MNN::Tensor> hostOut(new MNN::Tensor(output, MNN::Tensor::CAFFE));
    output->copyToHostTensor(hostOut.get());
    const float* y = hostOut->host<float>();
    std::cout << "y=[" << y[0] << "," << y[1] << "," << y[2] << "]\n";

    const float expect[3] = {6.5f, 6.75f, 9.0f};
    bool ok = true;
    for (int i = 0; i < 3; ++i) {
        if (std::fabs(y[i] - expect[i]) > 1e-3f) {
            ok = false;
        }
    }
    std::cout << (ok ? "verify=OK" : "verify=FAIL") << "\n";
    return (code == MNN::NO_ERROR && ok) ? 0 : 3;
}
