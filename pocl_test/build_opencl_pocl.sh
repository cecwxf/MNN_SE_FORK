#!/usr/bin/env bash
set -euo pipefail

# Build MNN with OpenCL enabled, and run PoCL/OpenCL validation demos.
# Tested on OpenCloudOS 9.4 + PoCL 7.1 (OpenCL 3.0).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_pocl_opencl"

echo "ROOT_DIR=${ROOT_DIR}"
echo "BUILD_DIR=${BUILD_DIR}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure
cmake "${ROOT_DIR}" \
  -GNinja \
  -DMNN_OPENCL=ON \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_BUILD_TOOLS=ON \
  -DMNN_BUILD_CONVERTER=OFF \
  -DMNN_BUILD_DEMO=OFF \
  -DMNN_USE_SYSTEM_LIB=OFF

# Build
ninja -j"$(nproc)" MNN

# Convenience: point loader to libMNN.so and run demos
export LD_LIBRARY_PATH="${BUILD_DIR}:${LD_LIBRARY_PATH:-}"

echo "\n[Run] pocl_smoke" 
"${ROOT_DIR}/pocl_test/pocl_smoke" || true

echo "\n[Run] run_mnn_opencl_model (expects tiny_matmul_add.mnn in repo root)" 
if [[ -f "${ROOT_DIR}/tiny_matmul_add.mnn" ]]; then
  "${ROOT_DIR}/pocl_test/run_mnn_opencl_model" "${ROOT_DIR}/tiny_matmul_add.mnn" || true
else
  echo "Missing ${ROOT_DIR}/tiny_matmul_add.mnn (optional)"
fi
