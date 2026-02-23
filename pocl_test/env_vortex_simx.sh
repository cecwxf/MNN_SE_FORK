#!/usr/bin/env bash
# One-shot environment setup for MNN + PoCL + Vortex(simx) strict runs.
# Usage:
#   source ./pocl_test/env_vortex_simx.sh [cache-tag]

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Please source this file: source ./pocl_test/env_vortex_simx.sh [cache-tag]" >&2
  exit 1
fi

cache_tag="${1:-tiny}"
root="${HOME}/.openclaw/workspace"

export OCL_ICD_VENDORS="${root}/pocl/build-vx-simx4/ocl-vendors"
export LD_LIBRARY_PATH="${root}/mnn/build_pocl_opencl/source/backend/opencl:${root}/pocl/build-vx-simx4/lib/CL:${root}/pocl/build-vx-simx4/lib/CL/devices/vortex:${root}/vortex/runtime:${LD_LIBRARY_PATH:-}"

export POCL_BUILDING=1
export POCL_DEVICES=vortex
export VORTEX_DRIVER=simx

# Keep default logs concise; override with: export POCL_DEBUG=all
export POCL_DEBUG="${POCL_DEBUG:-error,warn}"

# Toolchain / finalize knobs (can still be overridden by caller)
export POCL_VORTEX_BINTOOL="${POCL_VORTEX_BINTOOL:-OBJCOPY=/usr/bin/llvm-objcopy ${root}/vortex/kernel/scripts/vxbin.py}"
export POCL_VORTEX_CFLAGS="${POCL_VORTEX_CFLAGS:--target-feature +m -target-feature +f}"
export POCL_VORTEX_CODEGEN_FEATURES="${POCL_VORTEX_CODEGEN_FEATURES:-+m,+f,+zicsr,-c}"
export POCL_VORTEX_FINALIZE_CFLAGS="${POCL_VORTEX_FINALIZE_CFLAGS:---target=riscv32-unknown-elf -march=rv32imaf -mabi=ilp32f -nostdlib}"
export POCL_VORTEX_LDFLAGS="${POCL_VORTEX_LDFLAGS:--Wl,-Bstatic,--gc-sections,-T${root}/vortex/kernel/scripts/link32.ld,--defsym=STARTUP_ADDR=0x80000000}"

# Fresh cache by default per invocation tag.
export POCL_CACHE_DIR="${POCL_CACHE_DIR:-/tmp/pocl-mnn-simx-${cache_tag}-$(date +%s)}"
mkdir -p "${POCL_CACHE_DIR}"

# Leave compiler temp files if caller needs deep diagnostics.
export POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES="${POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES:-0}"

echo "[env_vortex_simx] OCL_ICD_VENDORS=${OCL_ICD_VENDORS}"
echo "[env_vortex_simx] POCL_CACHE_DIR=${POCL_CACHE_DIR}"
echo "[env_vortex_simx] POCL_VORTEX_CODEGEN_FEATURES=${POCL_VORTEX_CODEGEN_FEATURES}"
