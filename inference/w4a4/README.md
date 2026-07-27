# W4A4 Blackwell artifact

This directory contains W4A4 RaZeR emulation kernels for current
Blackwell GPUs. They use native block-scaled FP4 tensor-core products:

- `razer_weight_w4a4.cu`: RaZeR weights with NVFP4 activations.
- `razer_full_w4a4.cu`: RaZeR weights with RaZeR activations.
- `razer_w4a4_common.cuh`: shared block-scaled FP4 GEMM configuration and
  validation helpers.
- `activation_quantization.{h,cu}`: asynchronous standard-NVFP4 and RaZeR
  activation quantization API.
- `activation_quantization_test.cu`: focused activation-quantizer test.
- `cutlass_nvfp4_baseline_gemm.cu`: native NVFP4 baseline.
- `cutlass_mxfp8_baseline_gemm.cu`: MXFP8 baseline.
- `cublas_fp16_baseline_gemm.cu`: cuBLAS FP16 baseline.

The standalone GEMM programs generate controlled inputs, validate their output
against the RaZeR arithmetic, and report logical throughput. They require all
measurement parameters explicitly.

## Arithmetic

For RaZeR weights and NVFP4 activations, decompose each weight as
`B = B_main + B_correction`. The implementation supports two block-scaled FP4
products, a concurrent graph, a single `K'=2K` concatenated product, and a
single `N'=2N` product followed by an output add.

For RaZeR activations and weights, use

```text
A B = 2 A_main B_main
    + 2 A_correction B_correction
    - (A_main - A_correction)(B_main - B_correction).
```

The implementation evaluates the three products independently, concurrently in
a reusable graph, or as one `K'=3K` concatenated product. Deterministic Split-K
variants are available for the concatenated path.

The available special magnitudes are `5` and one of `7`, `8`, or `9`. Negative
special values negate both native FP4 components.

## Build

The build script requires explicit toolchain and output paths. The output
directory must not already exist.

```bash
./build_razer_w4a4.sh \
  --nvcc /absolute/path/to/nvcc \
  --cutlass-root /absolute/path/to/cutlass \
  --output-dir /absolute/path/to/new-output-directory
```

The script targets `sm_120a` and builds the schedule variants in this
directory. It does not select a schedule automatically; tile size,
warp-specialized schedule, and Split-K can perform differently across matrix
shapes.

Run the focused correctness suite against that build with:

```bash
./test_razer_w4a4.sh --bin-dir /absolute/path/to/output
```

## Correctness examples

RaZeR weights with NVFP4 activations:

```bash
/absolute/path/to/output/weight_k128_cooperative \
  --mode=concat \
  --m=512 --n=4096 --k=4096 \
  --warmup=5 --iters=20 --flush-mb=0 --seed=17 \
  --weight-special-rate=0.03 --b-second-magnitude=8 \
  --check --max-normalized-error=0.02
```

RaZeR activations and weights:

```bash
/absolute/path/to/output/full_k128_cooperative \
  --concat-k \
  --m=512 --n=4096 --k=4096 \
  --warmup=5 --iters=20 --flush-mb=0 --seed=19 \
  --a-special-rate=0.03 --b-special-rate=0.03 \
  --b-second-magnitude=8 \
  --check --max-normalized-error=0.02
```

To include activation-coordinate generation in the full-RaZeR timed region,
add `--online-a-remap`. To use the concurrent three-product graph, replace
`--concat-k` with `--overlap-graph` on a binary built with
`RAZER_FULL_OVERLAP_GRAPH`, such as `full_k128_cooperative`.

Run the activation-quantization test with:

```bash
/absolute/path/to/output/activation_quantization_test
```

The quantizer API uses caller-owned device outputs and reduction workspace.
Call `activation_quantization_workspace_size` before
`quantize_activations`. The launch is asynchronous on the supplied CUDA stream.
