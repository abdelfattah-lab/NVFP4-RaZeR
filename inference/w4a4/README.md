# W4A4 Blackwell artifact

This directory contains W4A4 RaZeR emulation kernels for current Blackwell
GPUs (SM120a), written in inline PTX. Operands are read once in the RaZeR
storage format (4-bit codes plus one scale byte per 16 values whose spare
bits select the special value); on chip, each value is split into ordinary
E2M1 components, which block-scaled FP4 MMAs
(`mma.sync ... kind::mxf4nvf4.block_scale`) multiply exactly.

- `razer_weight_w4a4.cu`: RaZeR weights with NVFP4 activations.
- `razer_full_w4a4.cu`: RaZeR weights with RaZeR activations.
- `razer_w4a4_kernels.cuh`: encodings, operand layout and the kernels.
- `razer_w4a4_host.cuh`: synthetic operands, exact reference, layout
  rearrangement, and the command-line driver of both programs.
- `activation_quantization.{h,cu}`: NVFP4 and RaZeR activation quantization.
- `activation_quantization_test.cu`: activation-quantizer test.
- `cutlass_nvfp4_baseline_gemm.cu`: native NVFP4 baseline.
- `cutlass_mxfp8_baseline_gemm.cu`: MXFP8 baseline.
- `cublas_fp16_baseline_gemm.cu`: cuBLAS FP16 baseline.

The GEMM programs generate synthetic inputs with the given special-value
rates, time the GEMM, and report logical throughput (2·M·N·K / time).

## Arithmetic

For RaZeR weights and NVFP4 activations, each weight is split as
`B = B_main + B_correction` with ordinary E2M1 components:
ordinary `y = y + 0`, `5 = 4 + 1`, `7 = 4 + 3`, `8 = 4 + 4`, `9 = 6 + 3`.
Both products accumulate into the same registers.

For RaZeR activations and weights,

```text
A B = 2 A_main B_main
    + 2 A_correction B_correction
    - (A_main - A_correction)(B_main - B_correction),
```

with activation special `5 = 3 + 2` and weight specials `5 = 3 + 2`,
`7 = 4 + 3`, `8 = 6 + 2`, `9 = 6 + 3`. The three products are three MMA passes
into the same accumulators; the factor 2 is applied through the scale
exponent.

The available weight special magnitudes are `5` and one of `7`, `8`, or `9`.
Negative special values negate both components.

## Operand layout

Each 16-row (activations) or 8-column (weights) by 64-wide K tile is stored
contiguously in `mma.m16n8k64` register order; the byte count equals that of
row-major NVFP4. Each thread block computes a `BM` x `BN` output tile (`BM`
tokens by `BN` output features, the first two numbers of the configuration
name: 32 and 128 for `32x128_w1x4_s8`). The activation layout depends on
`BM`, the weight layout on `BN`. Weights are rearranged offline with `pack_b`
(`razer_w4a4_host.cuh`); `quantize_activations` writes activations in this
layout. `razer_w4a4_kernels.cuh` gives the exact layout.

## Configurations

`--list-configs` lists the configurations compiled into each program and
`--config` selects one; none is selected automatically, since the fastest
tile size, warp layout, pipeline depth and split-K differ across shapes.
Names are `BMxBN_w<warps_m>x<warps_n>_s<stages>`, or, for full RaZeR,
`BMxBN_w<warps_m>x<warps_n>_c<stages>p<stages>e<warps>` for the variant in
which `e` extra warps split the RaZeR values for all MMA warps.

## Build

The output directory must not already exist. CUTLASS is needed only for the
activation quantizer (`cutlass/float8.h`).

```bash
./build_razer_w4a4.sh \
  --nvcc /absolute/path/to/nvcc \
  --cutlass-root /absolute/path/to/cutlass \
  --output-dir /absolute/path/to/new-output-directory
```

## Examples

RaZeR weights with NVFP4 activations (Qwen3-8B o projection, batch 16):

```bash
/absolute/path/to/output/razer_weight_w4a4 \
  --config=32x128_w1x4_s8 --split-k=4 \
  --m=16 --n=4096 --k=4096 \
  --warmup=10 --iters=100 --flush-mb=512 --seed=1 \
  --b-special-rate=0.06 --b-second-magnitude=7
```

RaZeR activations and weights (batch 8192):

```bash
/absolute/path/to/output/razer_full_w4a4 \
  --config=128x128_w2x4_c2p2e4 --split-k=1 \
  --m=8192 --n=4096 --k=4096 \
  --warmup=10 --iters=100 --flush-mb=512 --seed=1 \
  --a-special-rate=0.03 --b-special-rate=0.06 --b-second-magnitude=7
```

Each timed launch follows a `--flush-mb` MiB cache flush; the mean, median
and minimum are reported. `--check-samples=<n> --max-normalized-error=<e>`
additionally compares n sampled outputs with an FP64 reference.

## Activation quantization

`quantize_activations` quantizes an m x k row-major FP32 matrix into the
codes and scales the GEMMs read: `kNvfp4` for `razer_weight_w4a4`, `kRazer`
for `razer_full_w4a4`. `activation_quantization_output_size` and
`activation_quantization_workspace_size` give the buffer sizes.

```cpp
// Activations for razer_weight_w4a4 --config=32x128_w1x4_s8 (BM = 32).
razer::w4a4::quantize_activations(
    x, m, k, /*block_m=*/32, razer::w4a4::ActivationQuantizationFormat::kNvfp4,
    codes, scales, absmax, workspace, workspace_bytes, stream);
```

`block_m` is the `BM` of the GEMM that reads the output.

## Scope

SM120a only. N must be a multiple of 8 and K a multiple of 64; M is padded to
a multiple of `BM`. There is no PyTorch binding.
