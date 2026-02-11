cmake_minimum_required(VERSION 3.10)
project(mnn_pocl_test CXX)

set(CMAKE_CXX_STANDARD 11)

# This CMakeLists is intended to be used from the MNN top-level build.
# It builds small validation executables.

add_executable(pocl_smoke pocl_smoke.cpp)
add_executable(run_mnn_opencl run_mnn_opencl.cpp)
add_executable(run_mnn_opencl_model run_mnn_opencl_model.cpp)
add_executable(check_creator check_creator.cpp)
add_executable(clrt_create clrt_create.cpp)

# Link against OpenCL (ICD loader) and MNN
find_package(OpenCL REQUIRED)

target_link_libraries(pocl_smoke PRIVATE OpenCL::OpenCL)

# The following demos expect libMNN to be available in runtime search path.
# If integrating into the top-level build, you can also link to the MNN target.
