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

target_link_libraries(pocl_smoke PRIVATE OpenCL::OpenCL)

# Link to MNN shared library produced by top-level build.
# In the standard top-level build, libMNN.so is generated into the build root.
# We link by absolute path to avoid needing exported targets.
set(MNN_SHARED_LIB "${CMAKE_BINARY_DIR}/libMNN.so")
if (EXISTS "${MNN_SHARED_LIB}")
  foreach(t run_mnn_opencl run_mnn_opencl_model check_creator clrt_create)
    target_link_libraries(${t} PRIVATE "${MNN_SHARED_LIB}" OpenCL::OpenCL)
    target_include_directories(${t} PRIVATE "${PROJECT_SOURCE_DIR}/../include")
  endforeach()
else()
  message(WARNING "${MNN_SHARED_LIB} not found at configure time; demos will link only to OpenCL. Build may fail at link stage.")
  foreach(t run_mnn_opencl run_mnn_opencl_model check_creator clrt_create)
    target_link_libraries(${t} PRIVATE OpenCL::OpenCL)
    target_include_directories(${t} PRIVATE "${PROJECT_SOURCE_DIR}/../include")
  endforeach()
endif()
