#include <iostream>
#include <memory>
#include "source/backend/opencl/core/OpenCLBackend.hpp"
#include "include/MNN/MNNForwardType.h"

int main(){
    MNN::Backend::Info info;
    info.type = (MNNForwardType)MNN_FORWARD_OPENCL;
    auto rt = std::unique_ptr<MNN::OpenCL::CLRuntime>(new MNN::OpenCL::CLRuntime(info));
    std::cout << "isError=" << rt->isCLRuntimeError() << "\n";
    return rt->isCLRuntimeError()?1:0;
}
