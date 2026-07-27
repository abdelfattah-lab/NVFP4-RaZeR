#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_conversion.h"

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                  \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CUTLASS_CHECK(expr)                                                    \
  do {                                                                         \
    cutlass::Status _status = (expr);                                          \
    if (_status != cutlass::Status::kSuccess) {                                \
      std::cerr << "CUTLASS error: "                                           \
                << cutlassGetStatusString(_status)                             \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

template <typename T>
__host__ __device__ __forceinline__ auto make_iterator(T* pointer) {
  return cute::recast_ptr<T>(pointer);
}

static inline float decode_fp4_e2m1(std::uint8_t nibble) {
  nibble &= 0xFu;
  int sign = (nibble >> 3) & 0x1;
  std::uint8_t magnitude_code = nibble & 0x7u;
  constexpr float values[8] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  float value = values[magnitude_code];
  return sign ? -value : value;
}

static inline float decode_ue4m3(std::uint8_t raw) {
  raw &= 0x7Fu;
  int exponent = (raw >> 3) & 0xFu;
  int mantissa = raw & 0x7u;
  constexpr int kBias = 7;

  if (exponent == 0) {
    if (mantissa == 0) return 0.0f;
    float fraction = static_cast<float>(mantissa) / 8.0f;
    return std::ldexp(fraction, 1 - kBias);
  }
  // UE4M3 has no infinity. Only the all-ones payload is NaN; exponent 15
  // with mantissa 0..6 represents finite values 256..448.
  if (exponent == 0xF && mantissa == 0x7) return NAN;

  float fraction = 1.0f + static_cast<float>(mantissa) / 8.0f;
  return std::ldexp(fraction, exponent - kBias);
}

static inline void print_matrix_window(
    char const* name,
    float const* matrix,
    int rows,
    int columns,
    int start_row,
    int start_column,
    int view_rows = 4,
    int view_columns = 4) {
  int displayed_rows = std::min(view_rows, rows);
  int displayed_columns = std::min(view_columns, columns);
  if (start_row + displayed_rows > rows) {
    start_row = rows >= displayed_rows ? rows - displayed_rows : 0;
  }
  if (start_column + displayed_columns > columns) {
    start_column =
        columns >= displayed_columns ? columns - displayed_columns : 0;
  }

  std::printf(
      "%s (window %dx%d at [%d,%d]) =\n",
      name,
      displayed_rows,
      displayed_columns,
      start_row,
      start_column);
  for (int row = 0; row < displayed_rows; ++row) {
    std::printf("  ");
    for (int column = 0; column < displayed_columns; ++column) {
      float value =
          matrix[(start_row + row) * columns + start_column + column];
      if (std::isnan(value)) {
        std::printf("%12s ", "NaN");
      } else if (std::isinf(value)) {
        std::printf("%12s ", value > 0 ? "Inf" : "-Inf");
      } else {
        float absolute_value = std::fabs(value);
        if (absolute_value >= 10000000.0f ||
            (absolute_value != 0.0f && absolute_value < 0.001f)) {
          std::printf("%12.3e ", static_cast<double>(value));
        } else {
          std::printf("%12.3f ", static_cast<double>(value));
        }
      }
    }
    std::printf("\n");
  }
  std::printf("\n");
}

#if defined(RAZER_RASTER_ALONG_M) && defined(RAZER_RASTER_ALONG_N)
#error "Define at most one raster direction"
#endif

#if defined(RAZER_MAX_SWIZZLE)
static_assert(
    RAZER_MAX_SWIZZLE == 1 || RAZER_MAX_SWIZZLE == 2 ||
        RAZER_MAX_SWIZZLE == 4 || RAZER_MAX_SWIZZLE == 8,
    "RAZER_MAX_SWIZZLE must be one of 1, 2, 4, or 8.");
#endif

using ArchTag = cutlass::arch::Sm120;
using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;

using ElementAccumulator = float;
using ElementCompute = float;
using ElementC = float;
using ElementD = float;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

#ifndef RAZER_OUTPUT_ALIGNMENT
#define RAZER_OUTPUT_ALIGNMENT 1
#endif
static_assert(
    RAZER_OUTPUT_ALIGNMENT == 1 || RAZER_OUTPUT_ALIGNMENT == 4,
    "RAZER_OUTPUT_ALIGNMENT must be 1 or 4 float elements.");
constexpr int AlignmentC = RAZER_OUTPUT_ALIGNMENT;
constexpr int AlignmentD = RAZER_OUTPUT_ALIGNMENT;

#ifndef RAZER_TILE_M
#define RAZER_TILE_M 128
#endif
#ifndef RAZER_TILE_N
#define RAZER_TILE_N 128
#endif
#ifndef RAZER_TILE_K
#define RAZER_TILE_K 128
#endif
#ifndef RAZER_CLUSTER_M
#define RAZER_CLUSTER_M 1
#endif
#ifndef RAZER_CLUSTER_N
#define RAZER_CLUSTER_N 1
#endif
static_assert(
    RAZER_CLUSTER_M == 1 && RAZER_CLUSTER_N == 1,
    "SM120 block-scaled GEMM has no programmatic multicast; "
    "the cluster shape must be 1x1.");

using ThreadBlockShape = cute::Shape<
    cute::Int<RAZER_TILE_M>,
    cute::Int<RAZER_TILE_N>,
    cute::Int<RAZER_TILE_K>>;
using ClusterShape = cute::Shape<
    cute::Int<RAZER_CLUSTER_M>,
    cute::Int<RAZER_CLUSTER_N>,
    cute::_1>;

using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD,
    ElementCompute,
    ElementC>;
using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OpClass,
        ThreadBlockShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementAccumulator,
        ElementC,
        LayoutC,
        AlignmentC,
        ElementD,
        LayoutD,
        AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation>::CollectiveOp;

#if defined(RAZER_STAGE_COUNT)
static_assert(
    RAZER_STAGE_COUNT >= 2 && RAZER_STAGE_COUNT <= 8,
    "RAZER_STAGE_COUNT must be in [2, 8].");
using RazerStageCount =
    cutlass::gemm::collective::StageCount<RAZER_STAGE_COUNT>;
#else
using RazerStageCount =
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>;
#endif

#if defined(RAZER_COOPERATIVE)
using KernelSchedule =
    cutlass::gemm::KernelTmaWarpSpecializedCooperative;
#else
using KernelSchedule =
    cutlass::gemm::KernelTmaWarpSpecializedPingpong;
#endif

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OpClass,
        ElementA,
        LayoutA,
        AlignmentA,
        ElementB,
        LayoutB,
        AlignmentB,
        ElementAccumulator,
        ThreadBlockShape,
        ClusterShape,
        RazerStageCount,
        KernelSchedule>::CollectiveOp;

#if defined(RAZER_STREAM_K)
using RazerTileScheduler = cutlass::gemm::StreamKScheduler;
#else
using RazerTileScheduler = void;
#endif

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    RazerTileScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;
using LayoutSFA =
    typename GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB =
    typename GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig =
    typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

constexpr int K_BLOCK = 16;

__global__ void flush_cache_kernel(
    unsigned char* buffer,
    std::size_t byte_count) {
  std::size_t index =
      std::size_t(blockIdx.x) * std::size_t(blockDim.x) +
      std::size_t(threadIdx.x);
  std::size_t stride =
      std::size_t(blockDim.x) * std::size_t(gridDim.x);
  for (std::size_t byte = index; byte < byte_count; byte += stride) {
    buffer[byte] = static_cast<unsigned char>(byte);
  }
}
