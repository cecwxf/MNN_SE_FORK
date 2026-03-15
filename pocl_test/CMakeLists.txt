cmake_minimum_required(VERSION 3.10)
project(mnn_pocl_test CXX)

set(CMAKE_CXX_STANDARD 11)

# Build small validation executables.
add_executable(pocl_smoke pocl_smoke.cpp)
add_executable(run_mnn_opencl run_mnn_opencl.cpp)
add_executable(run_mnn_opencl_model run_mnn_opencl_model.cpp)
add_executable(check_creator check_creator.cpp)
add_executable(clrt_create clrt_create.cpp)

# Link OpenCL ICD loader
find_package(OpenCL REQUIRED)

# Use MNN shared library from the top-level build directory.
# The top-level build generates libMNN.so into ${CMAKE_BINARY_DIR}.
set(MNN_SHARED_LIB "${CMAKE_BINARY_DIR}/libMNN.so")

# Includes for MNN public headers
set(MNN_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../include")

# OpenCL-only smoke test
add_dependencies(pocl_smoke MNN)
target_link_libraries(pocl_smoke PRIVATE OpenCL::OpenCL)

# MNN-linked demos
foreach(t run_mnn_opencl run_mnn_opencl_model check_creator clrt_create)
  add_dependencies(${t} MNN)
  target_include_directories(${t} PRIVATE "${MNN_INCLUDE_DIR}")
  if (EXISTS "${MNN_SHARED_LIB}")
    target_link_libraries(${t} PRIVATE "${MNN_SHARED_LIB}" OpenCL::OpenCL dl)
  else()
    # Fallback: try -lMNN and build-dir link path
    target_link_directories(${t} PRIVATE "${CMAKE_BINARY_DIR}")
    target_link_libraries(${t} PRIVATE MNN OpenCL::OpenCL dl)
  endif()
endforeach()
