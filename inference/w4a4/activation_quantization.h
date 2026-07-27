#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace razer::w4a4 {

enum class ActivationQuantizationFormat {
  kNvfp4,
  kRazer,
};

// Returns the CUB reduction workspace required by quantize_activations.
// element_count must be positive, divisible by 16, and fit in a signed int.
cudaError_t activation_quantization_workspace_size(
    std::size_t element_count,
    std::size_t* workspace_bytes);

// Quantizes a contiguous FP32 tensor into packed E2M1 values and one UE4M3
// block scale per 16 input values. tensor_absmax receives the device-side
// tensor maximum; the corresponding NVFP4 global scale is
// tensor_absmax / (6 * 448).
//
// For kRazer, positive-zero E2M1 encodings are remapped to either +5 or -5.
// Bit 7 of each block-scale byte records the selected sign. For kNvfp4,
// bit 7 is clear.
//
// All pointers are device pointers. The function enqueues work on stream and
// does not allocate or synchronize.
cudaError_t quantize_activations(
    float const* input,
    std::size_t element_count,
    ActivationQuantizationFormat format,
    std::uint8_t* packed_fp4,
    std::uint8_t* block_scales,
    float* tensor_absmax,
    void* workspace,
    std::size_t workspace_bytes,
    cudaStream_t stream);

}  // namespace razer::w4a4
