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
  razer_weight_w4a4
  razer_full_w4a4
)
for binary in "${required[@]}"; do
  if [[ ! -x "$bin_dir/$binary" ]]; then
    echo "Missing executable: $bin_dir/$binary" >&2
    exit 1
  fi
done

# Every configuration with each second magnitude and with and without split-K.
# M = 200 and N = 1032 are not multiples of the CTA tile, so row and column
# padding are exercised. With these synthetic operands every product and
# partial sum is exactly representable in FP32, so the sampled outputs must
# match the FP64 reference to rounding of the final sum.
common=(
  --m=200
  --n=1032
  --k=1024
  --warmup=0
  --iters=1
  --flush-mb=0
  --b-special-rate=0.1
  --check-samples=256
  --max-normalized-error=1e-5
)

for program in razer_weight_w4a4 razer_full_w4a4; do
  extra=()
  if [[ "$program" == razer_full_w4a4 ]]; then
    extra=(--a-special-rate=0.1)
  fi
  while read -r config; do
    for magnitude in 7 8 9; do
      for split_k in 1 3; do
        "$bin_dir/$program" --config="$config" --split-k="$split_k" \
          --b-second-magnitude="$magnitude" --seed="$magnitude$split_k" \
          "${common[@]}" "${extra[@]}" | grep '^CHECK' |
          sed "s/^/$program $config q2=$magnitude split_k=$split_k: /"
      done
    done
  done < <("$bin_dir/$program" --list-configs)
done

"$bin_dir/activation_quantization_test"
