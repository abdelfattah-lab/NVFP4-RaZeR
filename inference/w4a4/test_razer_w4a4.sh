#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --bin-dir <absolute-build-output-path>" >&2
}

if [[ $# -ne 2 || "$1" != "--bin-dir" ]]; then
  usage
  exit 1
fi

bin_dir="$2"
if [[ "$bin_dir" != /* || ! -d "$bin_dir" ]]; then
  echo "--bin-dir must name an absolute build output directory." >&2
  exit 1
fi

required=(
  activation_quantization_test
  weight_k128_cooperative
  weight_split2_k128_cooperative
  full_k128_cooperative
  full_split2_k128_cooperative
)
for binary in "${required[@]}"; do
  if [[ ! -x "$bin_dir/$binary" ]]; then
    echo "Missing executable: $bin_dir/$binary" >&2
    exit 1
  fi
done

common_weight=(
  --m=16
  --n=128
  --k=128
  --warmup=1
  --iters=2
  --flush-mb=0
  --weight-special-rate=0.1
  --check
  --max-normalized-error=0.02
)

"$bin_dir/weight_k128_cooperative" \
  --mode=concat --seed=11 --b-second-magnitude=7 \
  "${common_weight[@]}"
"$bin_dir/weight_k128_cooperative" \
  --mode=concat-n-graph --seed=13 --b-second-magnitude=8 \
  "${common_weight[@]}"
"$bin_dir/weight_k128_cooperative" \
  --mode=two-pass-overlap-graph --seed=17 --b-second-magnitude=9 \
  "${common_weight[@]}"
"$bin_dir/weight_split2_k128_cooperative" \
  --mode=concat-n-splitk-graph --seed=19 --b-second-magnitude=9 \
  "${common_weight[@]}"

common_full=(
  --m=16
  --n=128
  --k=128
  --warmup=1
  --iters=2
  --flush-mb=0
  --a-special-rate=0.1
  --b-special-rate=0.1
  --check
  --max-normalized-error=0.02
)

"$bin_dir/full_k128_cooperative" \
  --concat-k --seed=23 --b-second-magnitude=7 \
  "${common_full[@]}"
"$bin_dir/full_k128_cooperative" \
  --overlap-graph --online-a-remap \
  --seed=29 --b-second-magnitude=8 \
  "${common_full[@]}"
"$bin_dir/full_split2_k128_cooperative" \
  --concat-k --online-a-remap \
  --seed=31 --b-second-magnitude=9 \
  "${common_full[@]}"

"$bin_dir/activation_quantization_test"
