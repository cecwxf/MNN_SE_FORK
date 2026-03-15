#include <CL/cl.h>
#include <iostream>
#include <vector>

static void die(const char* msg, cl_int err) {
    std::cerr << msg << " (err=" << err << ")\n";
    std::exit(1);
}

int main() {
    cl_uint nplat = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &nplat);
    if (err != CL_SUCCESS) die("clGetPlatformIDs(count) failed", err);
    std::vector<cl_platform_id> plats(nplat);
    err = clGetPlatformIDs(nplat, plats.data(), nullptr);
    if (err != CL_SUCCESS) die("clGetPlatformIDs(list) failed", err);

    std::cout << "platforms=" << nplat << "\n";
    for (cl_uint i = 0; i < nplat; i++) {
        char buf[4096]; size_t sz = 0;
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof(buf), buf, &sz);
        std::cout << "["<<i<<"] name=" << buf << "\n";
        clGetPlatformInfo(plats[i], CL_PLATFORM_VENDOR, sizeof(buf), buf, &sz);
        std::cout << "["<<i<<"] vendor=" << buf << "\n";
        clGetPlatformInfo(plats[i], CL_PLATFORM_VERSION, sizeof(buf), buf, &sz);
        std::cout << "["<<i<<"] version=" << buf << "\n";
    }

    // pick first platform, first device
    cl_platform_id plat = plats[0];
    cl_uint ndev = 0;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev);
    if (err != CL_SUCCESS) die("clGetDeviceIDs(count) failed", err);
    std::vector<cl_device_id> devs(ndev);
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, ndev, devs.data(), nullptr);
    if (err != CL_SUCCESS) die("clGetDeviceIDs(list) failed", err);

    char dbuf[4096]; size_t dsz=0;
    clGetDeviceInfo(devs[0], CL_DEVICE_NAME, sizeof(dbuf), dbuf, &dsz);
    std::cout << "device0=" << dbuf << "\n";

    const char* src = "__kernel void add1(__global float* x){ size_t i=get_global_id(0); x[i]+=1.0f; }";
    cl_context ctx = clCreateContext(nullptr, 1, &devs[0], nullptr, nullptr, &err);
    if (!ctx || err != CL_SUCCESS) die("clCreateContext failed", err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, devs[0], nullptr, &err);
    if (!q || err != CL_SUCCESS) die("clCreateCommandQueue failed", err);

    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
    if (!prog || err != CL_SUCCESS) die("clCreateProgramWithSource failed", err);
    err = clBuildProgram(prog, 1, &devs[0], "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logsz=0;
        clGetProgramBuildInfo(prog, devs[0], CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
        std::string log(logsz, '\0');
        clGetProgramBuildInfo(prog, devs[0], CL_PROGRAM_BUILD_LOG, logsz, (void*)log.data(), nullptr);
        std::cerr << "build log:\n" << log << "\n";
        die("clBuildProgram failed", err);
    }

    cl_kernel k = clCreateKernel(prog, "add1", &err);
    if (!k || err != CL_SUCCESS) die("clCreateKernel failed", err);

    float host[4] = {1,2,3,4};
    cl_mem bufm = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(host), host, &err);
    if (!bufm || err != CL_SUCCESS) die("clCreateBuffer failed", err);
    err = clSetKernelArg(k, 0, sizeof(cl_mem), &bufm);
    if (err != CL_SUCCESS) die("clSetKernelArg failed", err);
    size_t g = 4;
    err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) die("clEnqueueNDRangeKernel failed", err);
    clFinish(q);
    err = clEnqueueReadBuffer(q, bufm, CL_TRUE, 0, sizeof(host), host, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) die("clEnqueueReadBuffer failed", err);

    std::cout << "result=" << host[0] << ","<<host[1]<<","<<host[2]<<","<<host[3] << "\n";

    clReleaseMemObject(bufm);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return 0;
}
