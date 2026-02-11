#include "source/core/Backend.hpp"
#include "include/MNN/MNNForwardType.h"
#include <cstdio>

int main(){
    auto c = MNN::MNNGetExtraRuntimeCreator((MNNForwardType)MNN_FORWARD_OPENCL);
    printf("creator=%p\n", (void*)c);
    return c?0:1;
}
