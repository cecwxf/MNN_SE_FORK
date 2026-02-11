#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>
#include <iostream>
#include <memory>

int main() {
    const char* model = "tiny_matmul_add.mnn";

    auto net = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model));
    if (!net) {
        std::cerr << "failed to load model\n";
        return 1;
    }

    MNN::ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_OPENCL;
    cfg.numThread = 2;

    // backend config: prefer high precision? keep default.
    MNN::BackendConfig bcfg;
    bcfg.precision = MNN::BackendConfig::Precision_Normal;
    bcfg.power = MNN::BackendConfig::Power_Normal;
    bcfg.memory = MNN::BackendConfig::Memory_Normal;
    cfg.backendConfig = &bcfg;

    auto session = net->createSession(cfg);
    if (!session) {
        std::cerr << "failed to create opencl session (no opencl?)\n";
        return 2;
    }

    auto input = net->getSessionInput(session, "x");
    if (!input) {
        std::cerr << "failed to get input\n";
        return 3;
    }

    // fill input
    std::unique_ptr<MNN::Tensor> hostIn(new MNN::Tensor(input, MNN::Tensor::CAFFE));
    float* p = hostIn->host<float>();
    for (int i = 0; i < 4; i++) p[i] = float(i+1); // [1,2,3,4]
    input->copyFromHostTensor(hostIn.get());

    net->runSession(session);

    auto output = net->getSessionOutput(session, "y");
    std::unique_ptr<MNN::Tensor> hostOut(new MNN::Tensor(output, MNN::Tensor::CAFFE));
    output->copyToHostTensor(hostOut.get());

    const float* y = hostOut->host<float>();
    std::cout << "y=[" << y[0] << ", " << y[1] << ", " << y[2] << "]\n";

    net->releaseSession(session);
    return 0;
}
