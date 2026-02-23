#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./pocl_test/env_vortex_simx.sh matrix

models=(
  tiny_matmul_add.mnn
  tiny_matmul_add_relu.mnn
  tiny_matmul_add_reshape.mnn
  tiny_matmul_add_mul1.mnn
)

printf "%-28s %-4s %-20s %-28s\n" "MODEL" "RC" "Y" "NOTE"
printf "%-28s %-4s %-20s %-28s\n" "----------------------------" "----" "--------------------" "----------------------------"

for m in "${models[@]}"; do
  tag="${m%.mnn}"
  export POCL_CACHE_DIR="/tmp/pocl-mnn-${tag}-$(date +%s)"
  mkdir -p "$POCL_CACHE_DIR"
  log="/tmp/${tag}_strict.log"
  set +e
  timeout 240 env MNN_STRICT_OPENCL_NO_CPU_OP=1 ./build_pocl_opencl/pocl_test/run_mnn_opencl_model "./${m}" >"${log}" 2>&1
  rc=$?
  set -e
  y=$(grep -o 'y=\[[^]]*\]' "$log" | tail -n1 || true)
  note="ok"
  if grep -q 'fallback to BUFFER' "$log"; then
    note="buffer-fallback"
  fi
  if grep -Eq 'CL_BUILD_PROGRAM_FAILURE|undeclared identifier|Segmentation fault|decode abort' "$log"; then
    note="build/runtime-error"
  fi
  printf "%-28s %-4s %-20s %-28s\n" "$m" "$rc" "${y:-N/A}" "$note"
done
