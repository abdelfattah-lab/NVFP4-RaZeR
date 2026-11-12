#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --nvcc <absolute-path> --cutlass-root <absolute-path> --output-dir <new-path>" >&2
}

if [[ $# -ne 6 ]]; then
  usage
  exit 1
fi

nvcc_path=""
cutlass_root=""
output_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nvcc)
      nvcc_path="$2"
      shift 2
      ;;
    --cutlass-root)
      cutlass_root="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if [[ "$nvcc_path" != /* || ! -x "$nvcc_path" ]]; then
  echo "--nvcc must name an executable absolute path." >&2
  exit 1
fi
if [[ "$cutlass_root" != /* ||
      ! -f "$cutlass_root/include/cutlass/cutlass.h" ||
      ! -d "$cutlass_root/tools/util/include" ]]; then
  echo "--cutlass-root must name an absolute CUTLASS checkout." >&2
  exit 1
fi
if [[ -z "$output_dir" || -e "$output_dir" ]]; then
  echo "--output-dir must name a path that does not exist." >&2
  exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -- "$output_dir"

common=(
  -O3
  -std=c++17
  --expt-relaxed-constexpr
  "-I$cutlass_root/include"
  "-I$cutlass_root/tools/util/include"
  -gencode
  arch=compute_120a,code=sm_120a
)

build() {
  local output_name="$1"
  shift
  "$nvcc_path" "${common[@]}" "$@" -o "$output_dir/$output_name"
}

# Each GEMM program compiles in all of its CTA configurations (listed by
# --list-configs); each run selects one with --config.
build razer_weight_w4a4 "$script_dir/razer_weight_w4a4.cu"
build razer_full_w4a4 "$script_dir/razer_full_w4a4.cu"
build activation_quantization_test \
  "$script_dir/activation_quantization.cu" \
  "$script_dir/activation_quantization_test.cu"
