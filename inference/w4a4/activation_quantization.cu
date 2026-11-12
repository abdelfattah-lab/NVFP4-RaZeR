#include "activation_quantization.h"

#include <cub/device/device_reduce.cuh>
#include <thrust/iterator/transform_iterator.h>

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

// Byte offsets of one quantization block's outputs in the GEMM tile layout (razer_w4a4_kernels.cuh).
// Row r, 16-wide block kb, half h (8 values): the 8 codes form one 32-bit word
// of the mma.m16n8k64 A fragment of m-tile r / 16 and k-slice kb / 4.
struct PackedOffsets {
  std::size_t code_word;  // in 32-bit words
  std::size_t scale_byte;
};

__device__ __forceinline__ PackedOffsets packed_offsets(
    int row, int kb, int half, int k_slices, int tiles_per_group) {
  int m_tile = row >> 4;
  int group = m_tile / tiles_per_group;
  int tile_in_group = m_tile - group * tiles_per_group;
  int k_slice = kb >> 2;
  int row_in_tile = row & 15;
  int k_in_slice = (kb & 3) * 16 + half * 8;  // first k of the 8 values
  std::size_t tile =
      (std::size_t(group) * k_slices + k_slice) * tiles_per_group +
      tile_in_group;
  int lane = (row_in_tile & 7) * 4 + ((k_in_slice & 31) >> 3);
  int reg = (row_in_tile >> 3) + 2 * (k_in_slice >> 5);
  return {tile * 128 + lane * 4 + reg,
          tile * 64 + row_in_tile * 4 + (kb & 3)};
}

// One thread pair per 16-value block of the padded m_padded x k matrix; each
// thread quantizes 8 consecutive values.
template <bool Razer>
__global__ void quantize_fp32_blocks(
    float const* input,
    float const* tensor_absmax,
    std::uint8_t* packed_fp4,
    std::uint8_t* block_scales,
    int m,
    int k,
    int m_padded,
    int tiles_per_group) {
  int global_thread =
      int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int k_blocks = k / kBlockSize;
  int block_index = global_thread >> 1;
  int pair_lane = global_thread & 1;
  if (block_index >= m_padded * k_blocks) return;
  int row = block_index / k_blocks;
  int kb = block_index - row * k_blocks;
  PackedOffsets offsets =
      packed_offsets(row, kb, pair_lane, k / 64, tiles_per_group);
  std::uint32_t* code_word =
      reinterpret_cast<std::uint32_t*>(packed_fp4) + offsets.code_word;

  if (row >= m) {  // padding row; uniform across the thread pair
    *code_word = 0x88888888u;
    if (pair_lane == 0) block_scales[offsets.scale_byte] = 0x38u;
    return;
  }

  unsigned active_mask = __activemask();
  float values[8];
  float local_absmax = 0.0f;
#pragma unroll
  for (int element = 0; element < 8; ++element) {
    float value =
        input[std::size_t(row) * k + kb * kBlockSize + pair_lane * 8 +
              element];
    values[element] = value;
    local_absmax = fmaxf(local_absmax, fabsf(value));
  }
  float block_absmax =
      fmaxf(local_absmax, __shfl_xor_sync(active_mask, local_absmax, 1));
  float absolute_max = *tensor_absmax;

  if (absolute_max == 0.0f) {
    *code_word = 0x88888888u;
    if (pair_lane == 0) {
      block_scales[offsets.scale_byte] = quantize_ue4m3(ldexpf(1.0f, -9));
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

  *code_word = packed;
  if (pair_lane == 0) {
    std::uint8_t metadata =
        Razer && !select_positive ? std::uint8_t(0x80u)
                                  : std::uint8_t(0x00u);
    block_scales[offsets.scale_byte] = std::uint8_t(scale_raw | metadata);
  }
}

cudaError_t validate_shape(int m, int k) {
  if (m <= 0 || k <= 0 || k % 64 != 0 ||
      std::size_t(m) * std::size_t(k) > std::size_t(INT_MAX)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

bool valid_block_m(int block_m) {
  return block_m == 16 || block_m == 32 || block_m == 64 || block_m == 128;
}

int padded_rows(int m, int block_m) {
  return (m + block_m - 1) / block_m * block_m;
}

}  // namespace

cudaError_t activation_quantization_workspace_size(
    int m,
    int k,
    std::size_t* workspace_bytes) {
  if (workspace_bytes == nullptr) return cudaErrorInvalidValue;
  cudaError_t status = validate_shape(m, k);
  if (status != cudaSuccess) return status;

  AbsFloat transform;
  thrust::transform_iterator<AbsFloat, float const*, float> iterator(
      static_cast<float const*>(nullptr), transform);
  return cub::DeviceReduce::Max(
      nullptr,
      *workspace_bytes,
      iterator,
      static_cast<float*>(nullptr),
      m * k);
}

cudaError_t activation_quantization_output_size(
    int m,
    int k,
    int block_m,
    std::size_t* packed_fp4_bytes,
    std::size_t* block_scale_bytes) {
  if (packed_fp4_bytes == nullptr || block_scale_bytes == nullptr ||
      !valid_block_m(block_m)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = validate_shape(m, k);
  if (status != cudaSuccess) return status;
  std::size_t rows = std::size_t(padded_rows(m, block_m));
  *packed_fp4_bytes = rows * std::size_t(k) / 2;
  *block_scale_bytes = rows * std::size_t(k) / std::size_t(kBlockSize);
  return cudaSuccess;
}

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
    cudaStream_t stream) {
  cudaError_t status = validate_shape(m, k);
  if (status != cudaSuccess) return status;
  if (!valid_block_m(block_m)) return cudaErrorInvalidValue;
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
      m, k, &required_workspace);
  if (status != cudaSuccess) return status;
  if (workspace == nullptr || workspace_bytes < required_workspace) {
    return cudaErrorInvalidValue;
  }

  // The tensor maximum covers the m real rows only.
  AbsFloat transform;
  thrust::transform_iterator<AbsFloat, float const*, float> iterator(
      input, transform);
  status = cub::DeviceReduce::Max(
      workspace,
      workspace_bytes,
      iterator,
      tensor_absmax,
      m * k,
      stream);
  if (status != cudaSuccess) return status;

  int m_padded = padded_rows(m, block_m);
  if (std::size_t(m_padded) * std::size_t(k) > std::size_t(INT_MAX)) {
    return cudaErrorInvalidValue;
  }
  int num_blocks = m_padded * (k / kBlockSize);
  int grid = (num_blocks * 2 + kThreads - 1) / kThreads;
  int tiles_per_group = block_m / 16;
  switch (format) {
    case ActivationQuantizationFormat::kNvfp4:
      quantize_fp32_blocks<false><<<grid, kThreads, 0, stream>>>(
          input, tensor_absmax, packed_fp4, block_scales, m, k, m_padded,
          tiles_per_group);
      break;
    case ActivationQuantizationFormat::kRazer:
      quantize_fp32_blocks<true><<<grid, kThreads, 0, stream>>>(
          input, tensor_absmax, packed_fp4, block_scales, m, k, m_padded,
          tiles_per_group);
      break;
  }
  return cudaPeekAtLastError();
}

}  // namespace razer::w4a4
