#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <iostream>
#include <dlfcn.h>
#include <memory>

int main() {
    // Load OpenCL backend plugin (built as libMNN_CL.so) so RuntimeCreator(type=3) is registered.
    void* h = dlopen("libMNN_CL.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        std::cerr << "[pocl_test] dlopen(libMNN_CL.so) failed: " << dlerror() << std::endl;
    }

    auto net = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile("tiny_matmul_add.mnn"));
    if (!net) return 1;

    MNN::ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_OPENCL;
    cfg.backupType = MNN_FORWARD_CPU; // keep default; strict mode handled inside pipeline
    cfg.numThread = 2;

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
    return 0;
}
