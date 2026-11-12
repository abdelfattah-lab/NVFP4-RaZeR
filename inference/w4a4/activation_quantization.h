#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace razer::w4a4 {

enum class ActivationQuantizationFormat {
  kNvfp4,
  kRazer,
};

// Returns the CUB reduction workspace required by quantize_activations for an
// m x k activation matrix. m must be positive, k a positive multiple of 64, and
// m * k must fit in a signed int.
cudaError_t activation_quantization_workspace_size(
    int m,
    int k,
    std::size_t* workspace_bytes);

// Returns the output sizes of quantize_activations for CTA tile height block_m
// (16, 32, 64 or 128). Rows are padded up to a multiple of block_m.
cudaError_t activation_quantization_output_size(
    int m,
    int k,
    int block_m,
    std::size_t* packed_fp4_bytes,
    std::size_t* block_scale_bytes);

// Quantizes a row-major m x k FP32 activation matrix into packed E2M1 values
// and one UE4M3 block scale per 16 input values along k. tensor_absmax
// receives the device-side tensor maximum; the corresponding NVFP4 global
// scale is tensor_absmax / (6 * 448).
//
// For kRazer, positive-zero E2M1 encodings are remapped to either +5 or -5.
// Bit 7 of each block-scale byte records the selected sign. For kNvfp4,
// bit 7 is clear.
//
// The outputs are written directly in the tile layout that the fused GEMMs
// read (razer_w4a4_kernels.cuh), for a GEMM configuration whose CTA tile height
// is block_m: codes [m/block_m][k/64][block_m/16][32 lanes] x 16 bytes and
// scales [m/block_m][k/64][block_m/16][16 rows] x 4 bytes (m rounded up to a
// multiple of block_m). Padding rows hold
// code 0x8 (-0) and scale 0x38 (1.0).
//
// All pointers are device pointers. The function enqueues work on stream and
// does not allocate or synchronize.
cudaError_t quantize_activations(
    float const* input,
    int m,
    int k,
    int block_m,
    ActivationQuantizationFormat format,
    std::uint8_t* packed_fp4,
    std::uint8_t* block_scales,
    float* tensor_absmax,
    void* workspace,
    std::size_t workspace_bytes,
    cudaStream_t stream);

}  // namespace razer::w4a4
