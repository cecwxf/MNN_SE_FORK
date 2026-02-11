#!/usr/bin/env bash
set -euo pipefail

# Build MNN with OpenCL enabled, build PoCL validation demos, and run them.
#
# Expected environment:
# - OpenCL ICD loader installed
# - PoCL installed and registered via /etc/OpenCL/vendors/pocl.icd (system ICD)
#
# Optional strict validation (enabled by patch in this branch):
#   export MNN_STRICT_NO_CPU_RUNTIME=1
#   export MNN_STRICT_OPENCL_NO_CPU_OP=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_pocl_opencl"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${ROOT_DIR}" \
  -GNinja \
  -DMNN_OPENCL=ON \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_BUILD_TOOLS=ON \
  -DMNN_BUILD_CONVERTER=OFF \
  -DMNN_BUILD_DEMO=OFF \
  -DMNN_USE_SYSTEM_LIB=ON \
  -DMNN_BUILD_POCL_TEST=ON

ninja -j"$(nproc)" MNN pocl_smoke run_mnn_opencl run_mnn_opencl_model

export LD_LIBRARY_PATH="${BUILD_DIR}:${LD_LIBRARY_PATH:-}"

echo "\n[Run] pocl_smoke" 
"${BUILD_DIR}/pocl_smoke" || true

echo "\n[Run] run_mnn_opencl_model" 
if [[ -f "${ROOT_DIR}/tiny_matmul_add.mnn" ]]; then
  "${BUILD_DIR}/run_mnn_opencl_model" "${ROOT_DIR}/tiny_matmul_add.mnn" || true
else
  echo "Missing ${ROOT_DIR}/tiny_matmul_add.mnn (optional). See pocl_test/README.md to generate it."
fi
