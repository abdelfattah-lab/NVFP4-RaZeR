#include "activation_quantization.h"

#include <cub/device/device_reduce.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "cutlass/float8.h"

namespace razer::w4a4 {
namespace {

constexpr int kBlockSize = 16;
constexpr int kThreads = 256;

struct AbsFloat {
  __host__ __device__ float operator()(float value) const {
    return fabsf(value);
  }
};

struct QuantizedFp4 {
  float value;
  std::uint8_t code;
};

__device__ __forceinline__ QuantizedFp4 nearest_fp4(float value) {
  if (value <= -5.0f) return {-6.0f, 0xFu};
  if (value <= -3.5f) return {-4.0f, 0xEu};
  if (value <= -2.5f) return {-3.0f, 0xDu};
  if (value <= -1.75f) return {-2.0f, 0xCu};
  if (value <= -1.25f) return {-1.5f, 0xBu};
  if (value <= -0.75f) return {-1.0f, 0xAu};
  if (value <= -0.25f) return {-0.5f, 0x9u};
  if (value <= 0.25f) return {0.0f, 0x8u};
  if (value <= 0.75f) return {0.5f, 0x1u};
  if (value <= 1.25f) return {1.0f, 0x2u};
  if (value <= 1.75f) return {1.5f, 0x3u};
  if (value <= 2.5f) return {2.0f, 0x4u};
  if (value <= 3.5f) return {3.0f, 0x5u};
  if (value <= 5.0f) return {4.0f, 0x6u};
  return {6.0f, 0x7u};
}

__device__ __forceinline__ std::uint8_t quantize_ue4m3(float value) {
  cutlass::float_ue4m3_t quantized(value);
  return std::uint8_t(quantized.raw()) & 0x7Fu;
}

__device__ __forceinline__ float decode_ue4m3(std::uint8_t raw) {
  cutlass::float_ue4m3_t value(
      cutlass::float_ue4m3_t::bitcast(std::uint8_t(raw & 0x7Fu)));
  return float(value);
}

template <bool Razer>
__global__ void quantize_fp32_blocks(
    float const* input,
    float const* tensor_absmax,
    std::uint8_t* packed_fp4,
    std::uint8_t* block_scales,
    int num_blocks) {
  int global_thread =
      int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int block_index = global_thread >> 1;
  int pair_lane = global_thread & 1;
  if (block_index >= num_blocks) return;

  unsigned active_mask = __activemask();
  float values[8];
  float local_absmax = 0.0f;
#pragma unroll
  for (int element = 0; element < 8; ++element) {
    float value =
        input[block_index * kBlockSize + pair_lane * 8 + element];
    values[element] = value;
    local_absmax = fmaxf(local_absmax, fabsf(value));
  }
  float block_absmax =
      fmaxf(local_absmax, __shfl_xor_sync(active_mask, local_absmax, 1));
  float absolute_max = *tensor_absmax;

  if (absolute_max == 0.0f) {
    *reinterpret_cast<std::uint32_t*>(
        packed_fp4 + block_index * 8 + pair_lane * 4) =
        0x88888888u;
    if (pair_lane == 0) {
      block_scales[block_index] = quantize_ue4m3(ldexpf(1.0f, -9));
    }
    return;
  }

  float global_scale = absolute_max / (6.0f * 448.0f);
  float scale_unquantized = (block_absmax / global_scale) / 6.0f;
  scale_unquantized =
      fminf(448.0f, fmaxf(ldexpf(1.0f, -9), scale_unquantized));
  std::uint8_t scale_raw = quantize_ue4m3(scale_unquantized);
  float scale = global_scale * decode_ue4m3(scale_raw);

  std::uint32_t packed = 0;
  std::uint8_t positive_special_mask = 0;
  std::uint8_t negative_special_mask = 0;
  float signed_gain = 0.0f;
#pragma unroll
  for (int element = 0; element < 8; ++element) {
    float normalized = values[element] / scale;
    QuantizedFp4 ordinary = nearest_fp4(normalized);
    packed |= std::uint32_t(ordinary.code) << (4 * element);
    if constexpr (Razer) {
      float magnitude = fabsf(normalized);
      if (magnitude >= 4.5f && magnitude <= 5.5f) {
        float gain = 1.0f - 2.0f * fabsf(magnitude - 5.0f);
        signed_gain += copysignf(gain, normalized);
        if (normalized >= 0.0f) {
          positive_special_mask |= std::uint8_t(1u << element);
        } else {
          negative_special_mask |= std::uint8_t(1u << element);
        }
      }
    }
  }

  signed_gain += __shfl_xor_sync(active_mask, signed_gain, 1);
  bool select_positive = signed_gain > 0.0f;
  if constexpr (Razer) {
    std::uint8_t selected_mask =
        select_positive ? positive_special_mask : negative_special_mask;
    while (selected_mask != 0) {
      int element = __ffs(unsigned(selected_mask)) - 1;
      packed &= ~(std::uint32_t(0xFu) << (4 * element));
      selected_mask = std::uint8_t(
          selected_mask & std::uint8_t(selected_mask - 1u));
    }
  }

  *reinterpret_cast<std::uint32_t*>(
      packed_fp4 + block_index * 8 + pair_lane * 4) = packed;
  if (pair_lane == 0) {
    std::uint8_t metadata =
        Razer && !select_positive ? std::uint8_t(0x80u)
                                  : std::uint8_t(0x00u);
    block_scales[block_index] = std::uint8_t(scale_raw | metadata);
  }
}

cudaError_t validate_element_count(std::size_t element_count) {
  if (element_count == 0 ||
      element_count % std::size_t(kBlockSize) != 0 ||
      element_count > std::size_t(INT_MAX)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

}  // namespace

cudaError_t activation_quantization_workspace_size(
    std::size_t element_count,
    std::size_t* workspace_bytes) {
  if (workspace_bytes == nullptr) return cudaErrorInvalidValue;
  cudaError_t status = validate_element_count(element_count);
  if (status != cudaSuccess) return status;

  AbsFloat transform;
  cub::TransformInputIterator<float, AbsFloat, float const*> iterator(
      nullptr, transform);
  return cub::DeviceReduce::Max(
      nullptr,
      *workspace_bytes,
      iterator,
      static_cast<float*>(nullptr),
      int(element_count));
}

cudaError_t quantize_activations(
    float const* input,
    std::size_t element_count,
    ActivationQuantizationFormat format,
    std::uint8_t* packed_fp4,
    std::uint8_t* block_scales,
    float* tensor_absmax,
    void* workspace,
    std::size_t workspace_bytes,
    cudaStream_t stream) {
  cudaError_t status = validate_element_count(element_count);
  if (status != cudaSuccess) return status;
  if (input == nullptr || packed_fp4 == nullptr ||
      block_scales == nullptr || tensor_absmax == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (format != ActivationQuantizationFormat::kNvfp4 &&
      format != ActivationQuantizationFormat::kRazer) {
    return cudaErrorInvalidValue;
  }

  std::size_t required_workspace = 0;
  status = activation_quantization_workspace_size(
      element_count, &required_workspace);
  if (status != cudaSuccess) return status;
  if (workspace == nullptr || workspace_bytes < required_workspace) {
    return cudaErrorInvalidValue;
  }

  AbsFloat transform;
  cub::TransformInputIterator<float, AbsFloat, float const*> iterator(
      input, transform);
  status = cub::DeviceReduce::Max(
      workspace,
      workspace_bytes,
      iterator,
      tensor_absmax,
      int(element_count),
      stream);
  if (status != cudaSuccess) return status;

  int num_blocks = int(element_count / std::size_t(kBlockSize));
  int grid = (num_blocks * 2 + kThreads - 1) / kThreads;
  switch (format) {
    case ActivationQuantizationFormat::kNvfp4:
      quantize_fp32_blocks<false><<<grid, kThreads, 0, stream>>>(
          input, tensor_absmax, packed_fp4, block_scales, num_blocks);
      break;
    case ActivationQuantizationFormat::kRazer:
      quantize_fp32_blocks<true><<<grid, kThreads, 0, stream>>>(
          input, tensor_absmax, packed_fp4, block_scales, num_blocks);
      break;
  }
  return cudaPeekAtLastError();
}

}  // namespace razer::w4a4
