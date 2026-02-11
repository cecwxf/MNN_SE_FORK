# PoCL / OpenCL validation demos for MNN

This directory contains small C++ demo programs used to validate that:

- the OpenCL ICD (e.g. PoCL) is discoverable
- MNN OpenCL runtime can be created successfully
- simple models can run on MNN OpenCL backend

## Files
- `pocl_smoke.cpp`: enumerates OpenCL platforms/devices and runs a tiny kernel
- `run_mnn_opencl.cpp`: simple OpenCL runtime creation / run path sanity
- `run_mnn_opencl_model.cpp`: runs `tiny_matmul_add.mnn` (or other model) on OpenCL backend
- `check_creator.cpp`, `clrt_create.cpp`: helper checks for runtime creator / CLRuntime

## Notes
These are developer validation utilities; they are not part of MNN public API.
