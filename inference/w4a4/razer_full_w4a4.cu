/***************************************************************************************************
  RaZeR activations x RaZeR weights on current Blackwell hardware.

  Decompose each RaZeR coefficient into native E2M1 values

    A = Am + Ac
    B = Bm + Bc

  and evaluate

    AB = 2 Am Bm + 2 Ac Bc - (Am - Ac)(Bm - Bc).

  Ordinary E2M1 values use main=value and correction=0. Positive special
  values use

    +5 = +3 + +2
    +7 = +4 + +3
    +8 = +6 + +2
    +9 = +6 + +3.

  Negative specials negate both components. Main, correction, and difference
  are all native E2M1 values. The products can be evaluated as three
  block-scaled FP4 GEMMs or concatenated along K into one K'=3K GEMM.
  The --overlap-graph mode captures three independent GEMMs on nonblocking
  streams and a vectorized FP32 output add in one reusable CUDA graph.
  Static B preprocessing is performed once. Dynamic A preprocessing is
  performed once in prepacked mode or included in every timed online-remap
  iteration.
***************************************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "razer_w4a4_common.cuh"

#if defined(RAZER_FULL_SPLIT_K) && !defined(RAZER_STREAM_K)
#error "RAZER_FULL_SPLIT_K requires RAZER_STREAM_K"
#endif
#if defined(RAZER_FULL_SPLIT_K_GRAPH) && !defined(RAZER_FULL_SPLIT_K)
#error "RAZER_FULL_SPLIT_K_GRAPH requires RAZER_FULL_SPLIT_K"
#endif
#if defined(RAZER_FULL_SPLIT_K)
#ifndef RAZER_SPLIT_K
#error "RAZER_FULL_SPLIT_K requires RAZER_SPLIT_K"
#endif
static_assert(RAZER_SPLIT_K >= 2, "RAZER_SPLIT_K must be at least 2.");
#endif

template <typename Arguments>
static inline void full_configure_scheduler(
    Arguments& arguments,
    int runtime_swizzle,
    std::string const& runtime_raster) {
#if defined(RAZER_MAX_SWIZZLE)
  arguments.scheduler.max_swizzle_size = RAZER_MAX_SWIZZLE;
#endif
#if defined(RAZER_RASTER_ALONG_M)
  arguments.scheduler.raster_order =
      cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;
#elif defined(RAZER_RASTER_ALONG_N)
  arguments.scheduler.raster_order =
      cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN;
#endif
  if (runtime_swizzle == -1) return;
  arguments.scheduler.max_swizzle_size = runtime_swizzle;
  if (runtime_raster == "heuristic") {
    arguments.scheduler.raster_order =
        cutlass::gemm::kernel::detail::RasterOrderOptions::Heuristic;
  } else if (runtime_raster == "along-m") {
    arguments.scheduler.raster_order =
        cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;
  } else if (runtime_raster == "along-n") {
    arguments.scheduler.raster_order =
        cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN;
  } else {
    std::abort();
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// CLI
///////////////////////////////////////////////////////////////////////////////////////////////////

struct Options {
  bool help = false;
  int m = -1, n = -1, k = -1;
  int warmup = -1;
  int iters = -1;
  int flush_mb = -1;
  int64_t seed = -1;
  int b_second_magnitude = -1;
  double a_special_rate = -1.0;
  double b_special_rate = -1.0;
  double max_normalized_error = -1.0;
  int scheduler_swizzle = -1;
  std::string scheduler_raster;
  bool breakdown = false;
  bool correctness = false;
  bool online_a_remap = false;
  bool concat_k = false;
#if defined(RAZER_FULL_OVERLAP_GRAPH)
  bool overlap_graph = false;
#endif

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);
    help = cmd.check_cmd_line_flag("help");
    breakdown = cmd.check_cmd_line_flag("breakdown");
    correctness = cmd.check_cmd_line_flag("check");
    online_a_remap = cmd.check_cmd_line_flag("online-a-remap");
    concat_k = cmd.check_cmd_line_flag("concat-k");
#if defined(RAZER_FULL_OVERLAP_GRAPH)
    overlap_graph = cmd.check_cmd_line_flag("overlap-graph");
#endif
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("warmup", warmup);
    cmd.get_cmd_line_argument("iters", iters);
    cmd.get_cmd_line_argument("flush-mb", flush_mb);
    cmd.get_cmd_line_argument("seed", seed);
    cmd.get_cmd_line_argument("b-second-magnitude", b_second_magnitude);
    cmd.get_cmd_line_argument("a-special-rate", a_special_rate);
    cmd.get_cmd_line_argument("b-special-rate", b_special_rate);
    cmd.get_cmd_line_argument("max-normalized-error", max_normalized_error);
    cmd.get_cmd_line_argument("scheduler-swizzle", scheduler_swizzle);
    cmd.get_cmd_line_argument("scheduler-raster", scheduler_raster);
  }

  void usage() const {
    std::cout
      << "razer_full_w4a4\n"
      << "  Required: --m=<int> --n=<int> --k=<int> --warmup=<int> --iters=<int>\n"
      << "            --flush-mb=<int> --seed=<uint64>\n"
      << "            --a-special-rate=<0..1> --b-special-rate=<0..1>\n"
      << "            --b-second-magnitude=<7|8|9>\n"
      << "  Modes:    --online-a-remap | omit for prepacked A\n"
      << "            --concat-k evaluates the three products as one K'=3K GEMM\n"
#if defined(RAZER_FULL_OVERLAP_GRAPH)
      << "            --overlap-graph evaluates three K GEMMs concurrently and adds FP32 outputs\n"
#endif
      << "  Optional: --breakdown\n"
      << "            --scheduler-swizzle=<1|2|4|8> and "
         "--scheduler-raster=<heuristic|along-m|along-n>\n"
      << "            --check --max-normalized-error=<positive float>\n";
  }

  bool valid() const {
    if (m <= 0 || n <= 0 || k <= 0 || warmup < 0 || iters <= 0 || flush_mb < 0 || seed < 0) return false;
    if (a_special_rate < 0.0 || a_special_rate > 1.0) return false;
    if (b_special_rate < 0.0 || b_special_rate > 1.0) return false;
    if (b_second_magnitude != 7 &&
        b_second_magnitude != 8 &&
        b_second_magnitude != 9) return false;
    if (correctness && !(max_normalized_error > 0.0)) return false;
    if (!correctness && max_normalized_error >= 0.0) return false;
    bool scheduler_override =
        scheduler_swizzle != -1 || !scheduler_raster.empty();
    if (scheduler_override &&
        ((scheduler_swizzle != 1 && scheduler_swizzle != 2 &&
          scheduler_swizzle != 4 && scheduler_swizzle != 8) ||
         (scheduler_raster != "heuristic" &&
          scheduler_raster != "along-m" &&
          scheduler_raster != "along-n"))) {
      return false;
    }
#if defined(RAZER_MAX_SWIZZLE) || defined(RAZER_RASTER_ALONG_M) || \
    defined(RAZER_RASTER_ALONG_N)
    if (scheduler_override) return false;
#endif
#if defined(RAZER_FULL_SPLIT_K_GRAPH)
    if (breakdown) return false;
#endif
#if defined(RAZER_FULL_SPLIT_K)
    if (!concat_k) return false;
#endif
#if defined(RAZER_FULL_OVERLAP_GRAPH)
    if (concat_k && overlap_graph) return false;
    if (overlap_graph && breakdown) return false;
#endif
    return true;
  }
};

static inline uint8_t fp4_negate_host(uint8_t nibble) {
  return uint8_t((nibble & 0xFu) ^ 0x8u);
}

static inline uint8_t A_embedding_host(uint8_t nibble, bool special_neg, int coordinate) {
  if (coordinate < 0 || coordinate > 2) std::abort();
  if ((nibble & 0xFu) != 0x0u) {
    return coordinate == 1 ? uint8_t(0x0u) : uint8_t(nibble & 0xFu);
  }
  constexpr uint8_t positive_special[3] = {0x5u, 0x4u, 0x2u}; // +3, +2, +1
  return special_neg ? fp4_negate_host(positive_special[coordinate])
                     : positive_special[coordinate];
}

static inline uint8_t B_embedding_host(
    uint8_t nibble, bool special_neg, bool use_second_magnitude,
    int second_magnitude, int coordinate) {
  if (coordinate < 0 || coordinate > 2) std::abort();
  if (second_magnitude != 7 &&
      second_magnitude != 8 &&
      second_magnitude != 9) std::abort();
  if ((nibble & 0xFu) != 0x0u) {
    return coordinate == 1 ? uint8_t(0x0u) : uint8_t(nibble & 0xFu);
  }
  constexpr uint8_t positive_five[3] = {0x5u, 0x4u, 0x2u}; // +3, +2, +1
  constexpr uint8_t positive_seven[3] = {0x6u, 0x5u, 0x2u}; // +4, +3, +1
  constexpr uint8_t positive_eight[3] = {0x7u, 0x4u, 0x6u}; // +6, +2, +4
  constexpr uint8_t positive_nine[3] = {0x7u, 0x5u, 0x5u}; // +6, +3, +3
  uint8_t const* positive_second =
      second_magnitude == 7 ? positive_seven :
      second_magnitude == 8 ? positive_eight : positive_nine;
  uint8_t output = use_second_magnitude
      ? positive_second[coordinate]
      : positive_five[coordinate];
  return special_neg ? fp4_negate_host(output) : output;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Prologue kernels: build dense decomposed operands from packed RaZeR inputs.
///////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ uint8_t fp4_negate(uint8_t nibble) {
  return uint8_t((nibble & 0xFu) ^ 0x8u);
}

__device__ __forceinline__ uint8_t A_embedding(
    uint8_t nibble, bool special_neg, int coordinate) {
  if (nibble != 0x0u) {
    return coordinate == 1 ? uint8_t(0x0u) : nibble;
  }
  constexpr uint8_t positive_special[3] = {0x5u, 0x4u, 0x2u}; // +3, +2, +1
  uint8_t output = positive_special[coordinate];
  return special_neg ? fp4_negate(output) : output;
}

__device__ __forceinline__ uint8_t B_embedding(
    uint8_t nibble, bool special_neg, bool use_second_magnitude,
    int second_magnitude, int coordinate) {
  if (nibble != 0x0u) {
    return coordinate == 1 ? uint8_t(0x0u) : nibble;
  }
  constexpr uint8_t positive_five[3] = {0x5u, 0x4u, 0x2u}; // +3, +2, +1
  constexpr uint8_t positive_seven[3] = {0x6u, 0x5u, 0x2u}; // +4, +3, +1
  constexpr uint8_t positive_eight[3] = {0x7u, 0x4u, 0x6u}; // +6, +2, +4
  constexpr uint8_t positive_nine[3] = {0x7u, 0x5u, 0x5u}; // +6, +3, +3
  uint8_t const* positive_second =
      second_magnitude == 7 ? positive_seven :
      second_magnitude == 8 ? positive_eight : positive_nine;
  uint8_t output = use_second_magnitude
      ? positive_second[coordinate]
      : positive_five[coordinate];
  return special_neg ? fp4_negate(output) : output;
}

__global__ void build_A_three_packed(
  typename ElementA::DataType* A0_packed,
  typename ElementA::DataType* A1_packed,
  typename ElementA::DataType* A2_packed,
  const typename ElementA::DataType* A_in_packed,
  const uint8_t* SFA_raw,
  int m, int k) {

  int idx = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int num_k_blocks = k / K_BLOCK;
  int total_blocks = m * num_k_blocks;
  if (idx >= total_blocks) return;

  int mm = idx / num_k_blocks;
  int kb = idx - mm * num_k_blocks;
  bool neg = (SFA_raw[mm * num_k_blocks + kb] & 0x80u) != 0u;

  const uint8_t* in_bytes = reinterpret_cast<const uint8_t*>(A_in_packed);
  uint8_t* output_bytes[3] = {
      reinterpret_cast<uint8_t*>(A0_packed),
      reinterpret_cast<uint8_t*>(A1_packed),
      reinterpret_cast<uint8_t*>(A2_packed)};

  int bytes_per_row = k >> 1;
  int start_byte = mm * bytes_per_row + kb * (K_BLOCK >> 1);
  uint64_t in64 = *reinterpret_cast<const uint64_t*>(in_bytes + start_byte);
  uint64_t output64[3] = {0, 0, 0};

  #pragma unroll
  for (int bi = 0; bi < 8; ++bi) {
    uint8_t in_byte = uint8_t((in64 >> (8 * bi)) & 0xFFu);
    uint8_t lo = in_byte & 0xFu;
    uint8_t hi = (in_byte >> 4) & 0xFu;
    #pragma unroll
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      uint8_t out_lo = A_embedding(lo, neg, coordinate);
      uint8_t out_hi = A_embedding(hi, neg, coordinate);
      output64[coordinate] |=
          uint64_t(uint8_t((out_hi << 4) | out_lo)) << (8 * bi);
    }
  }

  #pragma unroll
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    *reinterpret_cast<uint64_t*>(output_bytes[coordinate] + start_byte) =
        output64[coordinate];
  }
}

__global__ void build_B_three_packed(
  typename ElementB::DataType* B0_packed,
  typename ElementB::DataType* B1_packed,
  typename ElementB::DataType* B2_packed,
  const typename ElementB::DataType* B_in_packed,
  const uint8_t* SFB_dense,
  int n, int k, int second_magnitude) {

  int idx = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int num_k_blocks = k / K_BLOCK;
  int total_blocks = n * num_k_blocks;
  if (idx >= total_blocks) return;

  int nn = idx / num_k_blocks;
  int kb = idx - nn * num_k_blocks;

  uint8_t meta = SFB_dense[nn * num_k_blocks + kb];
  bool neg  = (meta & 0x80u) != 0u;
  bool use_second_magnitude = (meta & 0x40u) != 0u;

  const uint8_t* B_in_bytes = reinterpret_cast<const uint8_t*>(B_in_packed);
  uint8_t* output_bytes[3] = {
      reinterpret_cast<uint8_t*>(B0_packed),
      reinterpret_cast<uint8_t*>(B1_packed),
      reinterpret_cast<uint8_t*>(B2_packed)};

  int bytes_per_col = k >> 1;
  int start_byte = nn * bytes_per_col + kb * (K_BLOCK >> 1);
  const uint64_t in64 =
      *reinterpret_cast<const uint64_t*>(B_in_bytes + start_byte);
  uint64_t output64[3] = {0, 0, 0};

  #pragma unroll
  for (int bi = 0; bi < 8; ++bi) {
    uint8_t in_byte = uint8_t((in64 >> (8 * bi)) & 0xFFu);
    uint8_t lo = in_byte & 0xFu;
    uint8_t hi = (in_byte >> 4) & 0xFu;
    #pragma unroll
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      uint8_t out_lo = B_embedding(
          lo, neg, use_second_magnitude, second_magnitude, coordinate);
      uint8_t out_hi = B_embedding(
          hi, neg, use_second_magnitude, second_magnitude, coordinate);
      output64[coordinate] |=
          uint64_t(uint8_t((out_hi << 4) | out_lo)) << (8 * bi);
    }
  }

  #pragma unroll
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    *reinterpret_cast<uint64_t*>(output_bytes[coordinate] + start_byte) =
        output64[coordinate];
  }
}

__global__ void build_A_concat_packed(
  typename ElementA::DataType* A_concat_packed,
  const typename ElementA::DataType* A_in_packed,
  const uint8_t* SFA_raw,
  int m, int k) {

  int idx = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int num_k_blocks = k / K_BLOCK;
  int total_blocks = m * num_k_blocks;
  if (idx >= total_blocks) return;

  int mm = idx / num_k_blocks;
  int kb = idx - mm * num_k_blocks;
  bool neg = (SFA_raw[mm * num_k_blocks + kb] & 0x80u) != 0u;

  const uint8_t* input = reinterpret_cast<const uint8_t*>(A_in_packed);
  uint8_t* output = reinterpret_cast<uint8_t*>(A_concat_packed);
  int bytes_per_segment = k >> 1;
  int input_start = mm * bytes_per_segment + kb * (K_BLOCK >> 1);
  int output_row_start = mm * (3 * bytes_per_segment);
  uint64_t in64 = *reinterpret_cast<const uint64_t*>(input + input_start);

  #pragma unroll
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    uint64_t output64 = 0;
    #pragma unroll
    for (int byte_index = 0; byte_index < 8; ++byte_index) {
      uint8_t in_byte = uint8_t((in64 >> (8 * byte_index)) & 0xFFu);
      uint8_t lo = A_embedding(in_byte & 0xFu, neg, coordinate);
      uint8_t hi = A_embedding((in_byte >> 4) & 0xFu, neg, coordinate);
      output64 |= uint64_t(uint8_t((hi << 4) | lo)) << (8 * byte_index);
    }
    int output_start =
        output_row_start + coordinate * bytes_per_segment +
        kb * (K_BLOCK >> 1);
    *reinterpret_cast<uint64_t*>(output + output_start) = output64;
  }
}

__global__ void build_B_concat_packed(
  typename ElementB::DataType* B_concat_packed,
  const typename ElementB::DataType* B_in_packed,
  const uint8_t* SFB_raw,
  int n, int k, int second_magnitude) {

  int idx = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int num_k_blocks = k / K_BLOCK;
  int total_blocks = n * num_k_blocks;
  if (idx >= total_blocks) return;

  int nn = idx / num_k_blocks;
  int kb = idx - nn * num_k_blocks;
  uint8_t metadata = SFB_raw[nn * num_k_blocks + kb];
  bool neg = (metadata & 0x80u) != 0u;
  bool use_second_magnitude = (metadata & 0x40u) != 0u;

  const uint8_t* input = reinterpret_cast<const uint8_t*>(B_in_packed);
  uint8_t* output = reinterpret_cast<uint8_t*>(B_concat_packed);
  int bytes_per_segment = k >> 1;
  int input_start = nn * bytes_per_segment + kb * (K_BLOCK >> 1);
  int output_column_start = nn * (3 * bytes_per_segment);
  uint64_t in64 = *reinterpret_cast<const uint64_t*>(input + input_start);

  #pragma unroll
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    uint64_t output64 = 0;
    #pragma unroll
    for (int byte_index = 0; byte_index < 8; ++byte_index) {
      uint8_t in_byte = uint8_t((in64 >> (8 * byte_index)) & 0xFFu);
      uint8_t lo = B_embedding(
          in_byte & 0xFu, neg, use_second_magnitude,
          second_magnitude, coordinate);
      uint8_t hi = B_embedding(
          (in_byte >> 4) & 0xFu, neg, use_second_magnitude,
          second_magnitude, coordinate);
      if (coordinate == 2) {
        lo = fp4_negate(lo);
        hi = fp4_negate(hi);
      }
      output64 |= uint64_t(uint8_t((hi << 4) | lo)) << (8 * byte_index);
    }
    int output_start =
        output_column_start + coordinate * bytes_per_segment +
        kb * (K_BLOCK >> 1);
    *reinterpret_cast<uint64_t*>(output + output_start) = output64;
  }
}

#if defined(RAZER_FULL_OVERLAP_GRAPH)
__global__ void full_add_three_output_float4(
    float4* output, float4 const* second, float4 const* third,
    size_t vector_count) {
  size_t index = size_t(blockIdx.x) * size_t(blockDim.x) +
      size_t(threadIdx.x);
  if (index >= vector_count) return;
  float4 first_value = output[index];
  float4 second_value = second[index];
  float4 third_value = third[index];
  first_value.x = (first_value.x + second_value.x) + third_value.x;
  first_value.y = (first_value.y + second_value.y) + third_value.y;
  first_value.z = (first_value.z + second_value.z) + third_value.z;
  first_value.w = (first_value.w + second_value.w) + third_value.w;
  output[index] = first_value;
}
#endif

static int run_concat_case(
    Options const& opt,
    std::vector<uint8_t> const& h_A,
    std::vector<uint8_t> const& h_B,
    std::vector<uint8_t> const& h_SFA_raw,
    std::vector<uint8_t> const& h_SFA_mma,
    std::vector<uint8_t> const& h_SFB_raw,
    std::vector<uint8_t> const& h_SFB_mma) {

  using namespace cute;
  int m = opt.m;
  int n = opt.n;
  int k = opt.k;
  int expanded_k = 3 * k;
  int num_k_blocks = k / K_BLOCK;

  static_assert(
      Sm1xxBlkScaledConfig::SFVecSize == K_BLOCK,
      "The concatenated scale replication assumes one scale per 16 K values.");

  cutlass::device_memory::allocation<uint8_t> d_SFA_raw(
      size_t(m) * size_t(num_k_blocks));
  cutlass::device_memory::allocation<uint8_t> d_SFB_raw(
      size_t(n) * size_t(num_k_blocks));
  CUDA_CHECK(cudaMemcpy(
      d_SFA_raw.get(), h_SFA_raw.data(), h_SFA_raw.size(),
      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      d_SFB_raw.get(), h_SFB_raw.data(), h_SFB_raw.size(),
      cudaMemcpyHostToDevice));

  using SFAType = ElementA::ScaleFactorType;
  using SFBType = ElementB::ScaleFactorType;
  cutlass::HostTensor<
      typename ElementA::DataType,
      cutlass::layout::PackedVectorLayout> block_A_in;
  cutlass::HostTensor<
      typename ElementA::DataType,
      cutlass::layout::PackedVectorLayout> block_A_concat;
  cutlass::HostTensor<
      typename ElementB::DataType,
      cutlass::layout::PackedVectorLayout> block_B_in;
  cutlass::HostTensor<
      typename ElementB::DataType,
      cutlass::layout::PackedVectorLayout> block_B_concat;
  cutlass::HostTensor<
      SFAType, cutlass::layout::PackedVectorLayout> block_SFA;
  cutlass::HostTensor<
      SFBType, cutlass::layout::PackedVectorLayout> block_SFB;
  cutlass::HostTensor<
      ElementC, cutlass::layout::PackedVectorLayout> block_C;
  cutlass::HostTensor<
      ElementD, cutlass::layout::PackedVectorLayout> block_D;

  auto stride_A_in =
      cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B_in =
      cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, {m, expanded_k, 1});
  auto stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, {n, expanded_k, 1});
  auto stride_C =
      cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  auto stride_D =
      cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  auto layout_A_in =
      make_layout(make_shape(m, k, 1), stride_A_in);
  auto layout_B_in =
      make_layout(make_shape(n, k, 1), stride_B_in);
  auto layout_A =
      make_layout(make_shape(m, expanded_k, 1), stride_A);
  auto layout_B =
      make_layout(make_shape(n, expanded_k, 1), stride_B);
  auto layout_C = make_layout(make_shape(m, n, 1), stride_C);
  auto layout_D = make_layout(make_shape(m, n, 1), stride_D);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      make_shape(m, n, expanded_k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      make_shape(m, n, expanded_k, 1));

  block_A_in.reset(cutlass::make_Coord(int(size(layout_A_in))));
  block_A_concat.reset(cutlass::make_Coord(int(size(layout_A))));
  block_B_in.reset(cutlass::make_Coord(int(size(layout_B_in))));
  block_B_concat.reset(cutlass::make_Coord(int(size(layout_B))));
  block_SFA.reset(cutlass::make_Coord(int(size(filter_zeros(layout_SFA)))));
  block_SFB.reset(cutlass::make_Coord(int(size(filter_zeros(layout_SFB)))));
  block_C.reset(cutlass::make_Coord(int(size(layout_C))));
  block_D.reset(cutlass::make_Coord(int(size(layout_D))));

  {
    auto tensor_A =
        make_tensor(make_iterator(block_A_in.host_data()), layout_A_in);
    for (int mm = 0; mm < m; ++mm) {
      for (int kk = 0; kk < k; ++kk) {
        typename ElementA::DataType value;
        value.raw() = typename ElementA::DataType::Base::Storage(
            h_A[size_t(mm) * size_t(k) + size_t(kk)] & 0xFu);
        tensor_A(mm, kk, 0) = value;
      }
    }
  }
  {
    auto tensor_B =
        make_tensor(make_iterator(block_B_in.host_data()), layout_B_in);
    for (int nn = 0; nn < n; ++nn) {
      for (int kk = 0; kk < k; ++kk) {
        typename ElementB::DataType value;
        value.raw() = typename ElementB::DataType::Base::Storage(
            h_B[size_t(kk) * size_t(n) + size_t(nn)] & 0xFu);
        tensor_B(nn, kk, 0) = value;
      }
    }
  }
  {
    auto tensor_SFA = make_tensor(block_SFA.host_data(), layout_SFA);
    auto tensor_SFB = make_tensor(block_SFB.host_data(), layout_SFB);
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      for (int mm = 0; mm < m; ++mm) {
        for (int kk = 0; kk < k; kk += K_BLOCK) {
          uint8_t raw = h_SFA_mma[
              size_t(mm) * size_t(num_k_blocks) +
              size_t(kk / K_BLOCK)];
          SFAType value;
          std::memcpy(&value, &raw, 1);
          tensor_SFA(mm, coordinate * k + kk, 0) = value;
        }
      }
      for (int nn = 0; nn < n; ++nn) {
        for (int kk = 0; kk < k; kk += K_BLOCK) {
          uint8_t raw = h_SFB_mma[
              size_t(nn) * size_t(num_k_blocks) +
              size_t(kk / K_BLOCK)];
          SFBType value;
          std::memcpy(&value, &raw, 1);
          if (coordinate < 2) {
            float target = 2.0f * float(value);
            SFBType doubled(target);
            if (!std::isfinite(target) || float(doubled) != target) {
              std::cerr
                  << "Cannot represent doubled B scale at n=" << nn
                  << " k_block=" << (kk / K_BLOCK)
                  << " raw_scale=" << int(raw)
                  << " decoded_scale=" << float(value) << "\n";
              return 4;
            }
            value = doubled;
          }
          tensor_SFB(nn, coordinate * k + kk, 0) = value;
        }
      }
    }
  }
  std::fill(
      block_C.host_data(), block_C.host_data() + size(layout_C), 0.0f);

  block_A_in.sync_device();
  block_A_concat.sync_device();
  block_B_in.sync_device();
  block_B_concat.sync_device();
  block_SFA.sync_device();
  block_SFB.sync_device();
  block_C.sync_device();
  block_D.sync_device();

  auto launch_A_concat = [&](cudaStream_t stream = nullptr) {
    int threads = 128;
    int total_blocks = m * num_k_blocks;
    int blocks = (total_blocks + threads - 1) / threads;
    build_A_concat_packed<<<blocks, threads, 0, stream>>>(
        block_A_concat.device_data(), block_A_in.device_data(),
        d_SFA_raw.get(), m, k);
    CUDA_CHECK(cudaGetLastError());
  };
  auto launch_B_concat = [&](cudaStream_t stream = nullptr) {
    int threads = 128;
    int total_blocks = n * num_k_blocks;
    int blocks = (total_blocks + threads - 1) / threads;
    build_B_concat_packed<<<blocks, threads, 0, stream>>>(
        block_B_concat.device_data(), block_B_in.device_data(),
        d_SFB_raw.get(), n, k, opt.b_second_magnitude);
    CUDA_CHECK(cudaGetLastError());
  };

  Gemm gemm;
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {m, n, expanded_k, 1},
    {
      block_A_concat.device_data(), stride_A,
      block_B_concat.device_data(), stride_B,
      block_SFA.device_data(), layout_SFA,
      block_SFB.device_data(), layout_SFB
    },
    {
      {1.0f, 0.0f},
      block_C.device_data(), stride_C,
      block_D.device_data(), stride_D
    }
  };
  full_configure_scheduler(
      arguments, opt.scheduler_swizzle, opt.scheduler_raster);
#if defined(RAZER_FULL_SPLIT_K)
  using StreamKParams =
      cutlass::gemm::kernel::detail::
          PersistentTileSchedulerSm90StreamKParams;
  arguments.scheduler.splits = RAZER_SPLIT_K;
  arguments.scheduler.decomposition_mode =
      StreamKParams::DecompositionMode::SplitK;
  arguments.scheduler.reduction_mode =
      StreamKParams::ReductionMode::Deterministic;
#endif
  size_t workspace_bytes = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_bytes);
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  cudaStream_t execution_stream = nullptr;
#if defined(RAZER_FULL_SPLIT_K_GRAPH)
  cudaGraph_t execution_graph = nullptr;
  cudaGraphExec_t execution_graph_exec = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(
      &execution_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamBeginCapture(
      execution_stream, cudaStreamCaptureModeGlobal));
  CUTLASS_CHECK(GemmKernel::initialize_workspace(
      arguments, workspace.get(), execution_stream));
  CUTLASS_CHECK(gemm.run(execution_stream));
  CUDA_CHECK(cudaStreamEndCapture(execution_stream, &execution_graph));
  CUDA_CHECK(cudaGraphInstantiate(
      &execution_graph_exec, execution_graph, nullptr, nullptr, 0));
#endif

  auto run_gemm = [&]() {
#if defined(RAZER_FULL_SPLIT_K_GRAPH)
    CUDA_CHECK(cudaGraphLaunch(execution_graph_exec, execution_stream));
#elif defined(RAZER_FULL_SPLIT_K)
    CUTLASS_CHECK(GemmKernel::initialize_workspace(
        arguments, workspace.get(), execution_stream));
    CUTLASS_CHECK(gemm.run(execution_stream));
#else
    CUTLASS_CHECK(gemm.run(execution_stream));
#endif
  };

  if (opt.correctness) {
    std::cout << "\n=== Concatenated correctness pass "
              << m << "x" << n << "x" << k << " ===\n";
    launch_A_concat(execution_stream);
    launch_B_concat(execution_stream);
    run_gemm();
    CUDA_CHECK(cudaStreamSynchronize(execution_stream));
    block_A_concat.sync_host();
    block_B_concat.sync_host();
    block_D.sync_host();

    auto tensor_A =
        make_tensor(make_iterator(block_A_concat.host_data()), layout_A);
    auto tensor_B =
        make_tensor(make_iterator(block_B_concat.host_data()), layout_B);
    for (int mm = 0; mm < m; ++mm) {
      for (int kk = 0; kk < k; ++kk) {
        uint8_t input =
            h_A[size_t(mm) * size_t(k) + size_t(kk)] & 0xFu;
        bool neg = (
            h_SFA_raw[
                size_t(mm) * size_t(num_k_blocks) +
                size_t(kk / K_BLOCK)] & 0x80u) != 0u;
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
          typename ElementA::DataType got_value =
              tensor_A(mm, coordinate * k + kk, 0);
          uint8_t got =
              uint8_t(uint8_t(got_value.raw()) & 0xFu);
          uint8_t expected =
              A_embedding_host(input, neg, coordinate);
          if (got != expected) {
            std::cerr << "CONCAT A EMBEDDING CHECK FAILED at m="
                      << mm << " k=" << kk
                      << " coordinate=" << coordinate
                      << " expected=" << int(expected)
                      << " got=" << int(got) << "\n";
            return 2;
          }
        }
      }
    }
    for (int nn = 0; nn < n; ++nn) {
      for (int kk = 0; kk < k; ++kk) {
        uint8_t input =
            h_B[size_t(kk) * size_t(n) + size_t(nn)] & 0xFu;
        uint8_t metadata =
            h_SFB_raw[
                size_t(nn) * size_t(num_k_blocks) +
                size_t(kk / K_BLOCK)];
        bool neg = (metadata & 0x80u) != 0u;
        bool use_second_magnitude = (metadata & 0x40u) != 0u;
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
          typename ElementB::DataType got_value =
              tensor_B(nn, coordinate * k + kk, 0);
          uint8_t got =
              uint8_t(uint8_t(got_value.raw()) & 0xFu);
          uint8_t expected =
              B_embedding_host(
                  input, neg, use_second_magnitude,
                  opt.b_second_magnitude, coordinate);
          if (coordinate == 2) {
            expected = fp4_negate_host(expected);
          }
          if (got != expected) {
            std::cerr << "CONCAT B EMBEDDING CHECK FAILED at n="
                      << nn << " k=" << kk
                      << " coordinate=" << coordinate
                      << " expected=" << int(expected)
                      << " got=" << int(got) << "\n";
            return 2;
          }
        }
      }
    }
    std::cout << "Concatenated A/B embedding checks: PASS\n";

    auto tensor_D = make_tensor(block_D.host_data(), layout_D);
    std::vector<float> reference(size_t(m) * size_t(n));
    float max_abs_reference = 0.0f;
    for (int mm = 0; mm < m; ++mm) {
      for (int nn = 0; nn < n; ++nn) {
        float accumulator = 0.0f;
        for (int kk = 0; kk < k; ++kk) {
          int k_block = kk / K_BLOCK;
          float scale_a = decode_ue4m3(
              h_SFA_mma[
                  size_t(mm) * size_t(num_k_blocks) +
                  size_t(k_block)]);
          float scale_b = decode_ue4m3(
              h_SFB_mma[
                  size_t(nn) * size_t(num_k_blocks) +
                  size_t(k_block)]);
          uint8_t a_nibble =
              h_A[size_t(mm) * size_t(k) + size_t(kk)] & 0xFu;
          uint8_t b_nibble =
              h_B[size_t(kk) * size_t(n) + size_t(nn)] & 0xFu;
          uint8_t a_metadata =
              h_SFA_raw[
                  size_t(mm) * size_t(num_k_blocks) +
                  size_t(k_block)];
          uint8_t b_metadata =
              h_SFB_raw[
                  size_t(nn) * size_t(num_k_blocks) +
                  size_t(k_block)];
          float a_coefficient = decode_fp4_e2m1(a_nibble);
          if (a_nibble == 0x0u) {
            a_coefficient =
                (a_metadata & 0x80u) ? -5.0f : 5.0f;
          }
          float b_coefficient = decode_fp4_e2m1(b_nibble);
          if (b_nibble == 0x0u) {
            float magnitude = (b_metadata & 0x40u)
                ? float(opt.b_second_magnitude)
                : 5.0f;
            b_coefficient =
                (b_metadata & 0x80u) ? -magnitude : magnitude;
          }
          accumulator +=
              (a_coefficient * scale_a) *
              (b_coefficient * scale_b);
        }
        reference[size_t(mm) * size_t(n) + size_t(nn)] = accumulator;
        max_abs_reference =
            std::max(max_abs_reference, fabsf(accumulator));
      }
    }

    float max_abs_error = 0.0f;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    size_t nonfinite = 0;
    for (int mm = 0; mm < m; ++mm) {
      for (int nn = 0; nn < n; ++nn) {
        float gpu = tensor_D(mm, nn, 0);
        float cpu = reference[size_t(mm) * size_t(n) + size_t(nn)];
        if (!std::isfinite(gpu) || !std::isfinite(cpu)) {
          ++nonfinite;
          continue;
        }
        float difference = fabsf(gpu - cpu);
        max_abs_error = std::max(max_abs_error, difference);
        squared_error += double(difference) * double(difference);
        squared_reference += double(cpu) * double(cpu);
      }
    }
    double normalized_max_error =
        double(max_abs_error) /
        std::max(1.0, double(max_abs_reference));
    double relative_l2_error = std::sqrt(
        squared_error / std::max(1.0, squared_reference));
    std::cout << std::scientific
              << "Correctness: max_abs=" << max_abs_error
              << " normalized_max=" << normalized_max_error
              << " relative_l2=" << relative_l2_error
              << " nonfinite=" << nonfinite << "\n"
              << std::defaultfloat;
    if (nonfinite != 0 ||
        normalized_max_error > opt.max_normalized_error) {
      std::cerr << "CORRECTNESS FAILED: normalized_max_error="
                << normalized_max_error
                << " threshold=" << opt.max_normalized_error
                << " nonfinite=" << nonfinite << "\n";
      return 3;
    }
    std::cout << "Correctness threshold: PASS\n";
  }

  size_t flush_bytes =
      size_t(opt.flush_mb) * 1024ull * 1024ull;
  unsigned char* flush_buffer = nullptr;
  if (flush_bytes != 0) {
    CUDA_CHECK(cudaMalloc(&flush_buffer, flush_bytes));
  }
  auto flush = [&]() {
    if (flush_buffer != nullptr) {
      flush_cache_kernel<<<1024, 256>>>(
          flush_buffer, flush_bytes);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  };

  launch_B_concat(execution_stream);
  if (!opt.online_a_remap) launch_A_concat(execution_stream);
  CUDA_CHECK(cudaStreamSynchronize(execution_stream));
  for (int iteration = 0; iteration < opt.warmup; ++iteration) {
    flush();
    if (opt.online_a_remap) launch_A_concat(execution_stream);
    run_gemm();
  }
  CUDA_CHECK(cudaStreamSynchronize(execution_stream));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<double> samples;
  samples.reserve(size_t(opt.iters));
  for (int iteration = 0; iteration < opt.iters; ++iteration) {
    flush();
    CUDA_CHECK(cudaEventRecord(start, execution_stream));
    if (opt.online_a_remap) launch_A_concat(execution_stream);
    run_gemm();
    CUDA_CHECK(cudaEventRecord(stop, execution_stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    samples.push_back(double(elapsed));
  }
  std::vector<double> sorted_samples = samples;
  std::sort(sorted_samples.begin(), sorted_samples.end());
  double sum = 0.0;
  for (double sample : samples) sum += sample;
  double best = sorted_samples.front();
  double median = sorted_samples[sorted_samples.size() / 2];
  double p90 = sorted_samples[
      size_t(std::floor(0.9 * double(sorted_samples.size() - 1)))];
  double average = sum / double(samples.size());
  double effective_operations =
      2.0 * double(m) * double(n) * double(k);
  std::cout << "METHOD=full-razer-concat"
#if defined(RAZER_FULL_SPLIT_K)
            << "-splitk"
#endif
#if defined(RAZER_FULL_SPLIT_K_GRAPH)
            << "-graph"
#endif
            << "\n"
            << "M,N,K = " << m << "," << n << "," << k << "\n"
            << std::fixed << std::setprecision(6)
            << "Best: " << best << " ms, Median: " << median
            << " ms, P90: " << p90 << " ms, Avg: " << average
            << " ms over " << opt.iters << " iters (warmup "
            << opt.warmup << ")\n"
            << "Timing includes: "
            << (opt.online_a_remap
                    ? "online A concat remap"
                    : "prepacked A concat")
            << " + 1x GEMM(K'=3K)"
#if defined(RAZER_FULL_SPLIT_K)
            << " with deterministic Split-K=" << RAZER_SPLIT_K
#endif
#if defined(RAZER_FULL_SPLIT_K_GRAPH)
            << " in one reusable CUDA graph"
#endif
            << "; static B preprocessing excluded; "
            << "flush_mb=" << opt.flush_mb << "\n"
            << std::setprecision(2)
            << "Effective logical throughput: best "
            << effective_operations / (best * 1.0e9)
            << " TFLOPs, avg "
            << effective_operations / (average * 1.0e9)
            << " TFLOPs (2*M*N*K logical ops; actual tensor work is 3x)\n\n";
  std::cout << "SAMPLES_MS=";
  for (size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) std::cout << ",";
    std::cout << std::setprecision(9) << samples[index];
  }
  std::cout << "\n";

  if (opt.breakdown) {
    cudaEvent_t event0, event1, event2;
    CUDA_CHECK(cudaEventCreate(&event0));
    CUDA_CHECK(cudaEventCreate(&event1));
    CUDA_CHECK(cudaEventCreate(&event2));
    flush();
    CUDA_CHECK(cudaEventRecord(event0, execution_stream));
    if (opt.online_a_remap) launch_A_concat(execution_stream);
    CUDA_CHECK(cudaEventRecord(event1, execution_stream));
    run_gemm();
    CUDA_CHECK(cudaEventRecord(event2, execution_stream));
    CUDA_CHECK(cudaEventSynchronize(event2));
    float remap_ms = 0.0f;
    float gemm_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&remap_ms, event0, event1));
    CUDA_CHECK(cudaEventElapsedTime(&gemm_ms, event1, event2));
    std::cout << std::fixed << std::setprecision(6)
              << "Stage breakdown (1 iter): a_remap "
              << remap_ms << " ms, concat_gemm "
              << gemm_ms << " ms, sum "
              << double(remap_ms) + double(gemm_ms)
              << " ms\n\n";
    CUDA_CHECK(cudaEventDestroy(event0));
    CUDA_CHECK(cudaEventDestroy(event1));
    CUDA_CHECK(cudaEventDestroy(event2));
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  if (flush_buffer != nullptr) CUDA_CHECK(cudaFree(flush_buffer));
#if defined(RAZER_FULL_SPLIT_K_GRAPH)
  CUDA_CHECK(cudaGraphExecDestroy(execution_graph_exec));
  CUDA_CHECK(cudaGraphDestroy(execution_graph));
  CUDA_CHECK(cudaStreamDestroy(execution_stream));
#endif
  return 0;
}

static int run_case(Options const& opt) {
  int m = opt.m;
  int n = opt.n;
  int k = opt.k;
  if (k % 64 != 0 || k % K_BLOCK != 0) {
    std::cerr << "Require k multiple of 64 and 16.\n";
    return 1;
  }
  if (m % 16 != 0 || n % 8 != 0) {
    std::cerr << "Require m multiple of 16 and n multiple of 8.\n";
    return 1;
  }

  int num_k_blocks = k / K_BLOCK;

  // Dense host buffers for raw FP4 values and block scales.
  std::vector<uint8_t> h_A(size_t(m) * size_t(k));
  std::vector<uint8_t> h_B(size_t(k) * size_t(n));
  // B scales use an unsigned E3M3 payload in bits [5:0], special-value sign
  // in bit 7, and second-magnitude selection in bit 6. A scales use an
  // unsigned UE4M3 payload in bits [6:0] and special-value sign in bit 7.
  // CUTLASS consumes numeric UE4M3 scale bytes, so metadata-bearing and
  // numeric-only representations are kept separately.
  std::vector<uint8_t> h_SFA_raw(size_t(m) * size_t(num_k_blocks));
  std::vector<uint8_t> h_SFA_mma(size_t(m) * size_t(num_k_blocks));
  std::vector<uint8_t> h_SFB_raw(size_t(n) * size_t(num_k_blocks));
  std::vector<uint8_t> h_SFB_mma(size_t(n) * size_t(num_k_blocks));

  std::mt19937_64 rng(uint64_t(opt.seed));
  std::uniform_int_distribution<int> normal_nibble(1, 15);
  std::uniform_int_distribution<int> bit(0, 1);
  std::uniform_int_distribution<int> e3_payload(0, 63);
  std::uniform_int_distribution<int> e4_payload(0, 126);
  std::bernoulli_distribution a_is_special(opt.a_special_rate);
  std::bernoulli_distribution b_is_special(opt.b_special_rate);

  size_t a_special_count = 0;
  size_t b_special_count = 0;
  for (size_t i = 0; i < h_A.size(); ++i) {
    bool special = a_is_special(rng);
    h_A[i] = special ? 0x0u : uint8_t(normal_nibble(rng));
    a_special_count += size_t(special);
  }
  for (size_t i = 0; i < h_B.size(); ++i) {
    bool special = b_is_special(rng);
    h_B[i] = special ? 0x0u : uint8_t(normal_nibble(rng));
    b_special_count += size_t(special);
  }

  auto e3m3_to_ue4m3_for_mma = [](uint8_t raw) -> uint8_t {
    // Convert the E3M3 payload in bits [5:0] to the equivalent UE4M3 payload
    // by adjusting the exponent bias from 3 to 7.
    uint8_t payload = raw & 0x3Fu;
    return uint8_t((payload + 0x20u) & 0x7Fu);
  };

  for (int i = 0; i < m * num_k_blocks; ++i) {
    uint8_t payload = uint8_t(e4_payload(rng));
    uint8_t sign = bit(rng) ? 0x80u : 0x00u;
    h_SFA_raw[i] = uint8_t(payload | sign);
    h_SFA_mma[i] = payload;
  }
  for (int i = 0; i < n * num_k_blocks; ++i) {
    uint8_t payload = uint8_t(e3_payload(rng));
    uint8_t sign = bit(rng) ? 0x80u : 0x00u;
    uint8_t mag  = bit(rng) ? 0x40u : 0x00u;
    uint8_t raw = uint8_t(payload | sign | mag);
    h_SFB_raw[i] = raw;
    h_SFB_mma[i] = e3m3_to_ue4m3_for_mma(raw);
  }

  std::cout << std::setprecision(10)
            << "Observed A sentinel rate = "
            << (double(a_special_count) / double(h_A.size())) << " ("
            << a_special_count << "/" << h_A.size() << ")\n"
            << "Observed B sentinel rate = "
            << (double(b_special_count) / double(h_B.size())) << " ("
            << b_special_count << "/" << h_B.size() << ")\n";

  if (opt.concat_k) {
    return run_concat_case(
        opt, h_A, h_B, h_SFA_raw, h_SFA_mma, h_SFB_raw, h_SFB_mma);
  }

  // Device allocations (natural dense metadata for preprocessing)
  cutlass::device_memory::allocation<uint8_t> d_SFA_raw(size_t(m) * size_t(num_k_blocks));
  cutlass::device_memory::allocation<uint8_t> d_SFB_raw(size_t(n) * size_t(num_k_blocks));
  CUDA_CHECK(cudaMemcpy(d_SFA_raw.get(), h_SFA_raw.data(), h_SFA_raw.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_SFB_raw.get(), h_SFB_raw.data(), h_SFB_raw.size(), cudaMemcpyHostToDevice));

  // CUTLASS HostTensors for three embedding coordinates, scales, and float C/D.
  using SFAType = ElementA::ScaleFactorType;
  using SFBType = ElementB::ScaleFactorType;

  cutlass::HostTensor<typename ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A_in;
  cutlass::HostTensor<typename ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A0;
  cutlass::HostTensor<typename ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A1;
  cutlass::HostTensor<typename ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A2;
  cutlass::HostTensor<typename ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B_in;
  cutlass::HostTensor<typename ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B0;
  cutlass::HostTensor<typename ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B1;
  cutlass::HostTensor<typename ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B2;

  cutlass::HostTensor<SFAType, cutlass::layout::PackedVectorLayout> block_SFA;
  cutlass::HostTensor<SFBType, cutlass::layout::PackedVectorLayout> block_SFB;

  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;

  // Strides / layouts
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_A = cute::make_layout(cute::make_shape(m, k, 1), stride_A);
  auto layout_B = cute::make_layout(cute::make_shape(n, k, 1), stride_B);
  auto layout_C = cute::make_layout(cute::make_shape(m, n, 1), stride_C);
  auto layout_D = cute::make_layout(cute::make_shape(m, n, 1), stride_D);

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  // Allocate host/device storage
  block_A_in.reset(cutlass::make_Coord(int(cute::size(layout_A))));
  block_A0.reset(cutlass::make_Coord(int(cute::size(layout_A))));
  block_A1.reset(cutlass::make_Coord(int(cute::size(layout_A))));
  block_A2.reset(cutlass::make_Coord(int(cute::size(layout_A))));
  block_B_in.reset(cutlass::make_Coord(int(cute::size(layout_B))));
  block_B0.reset(cutlass::make_Coord(int(cute::size(layout_B))));
  block_B1.reset(cutlass::make_Coord(int(cute::size(layout_B))));
  block_B2.reset(cutlass::make_Coord(int(cute::size(layout_B))));

  block_SFA.reset(cutlass::make_Coord(int(cute::size(cute::filter_zeros(layout_SFA)))));
  block_SFB.reset(cutlass::make_Coord(int(cute::size(cute::filter_zeros(layout_SFB)))));

  block_C.reset(cutlass::make_Coord(int(cute::size(layout_C))));
  block_D.reset(cutlass::make_Coord(int(cute::size(layout_D))));

  // HostTensor uses PackedVectorLayout for subbyte (4-bit) storage. Inputs are
  // packed on host; all three embedded coordinates are produced on device.
  {
    using namespace cute;

    auto tA = make_tensor(make_iterator(block_A_in.host_data()), layout_A);
    for (int mm = 0; mm < m; ++mm) {
      for (int kk = 0; kk < k; ++kk) {
        typename ElementA::DataType v;
        v.raw() = typename ElementA::DataType::Base::Storage(h_A[size_t(mm) * size_t(k) + size_t(kk)] & 0xFu);
        tA(mm, kk, 0) = v;
      }
    }
  }

  // Pack B into the CUTLASS packed layout (layout_B: N x K, ColumnMajor) on host.
  // h_B is stored as KxN row-major (idx = kk*n + nn).
  {
    using namespace cute;

    auto tB = make_tensor(make_iterator(block_B_in.host_data()), layout_B);
    for (int nn = 0; nn < n; ++nn) {
      for (int kk = 0; kk < k; ++kk) {
        typename ElementB::DataType v;
        v.raw() = typename ElementB::DataType::Base::Storage(h_B[size_t(kk) * size_t(n) + size_t(nn)] & 0xFu);
        tB(nn, kk, 0) = v;
      }
    }
  }

  // Pack dense SFA/SFB into CUTLASS interleaved layouts via cute::Tensor indexing
  {
    using namespace cute;

    auto tSFA = make_tensor(block_SFA.host_data(), layout_SFA);
    auto tSFB = make_tensor(block_SFB.host_data(), layout_SFB);

    // SFA: depends on (m, kblock)
    for (int mm = 0; mm < m; ++mm) {
      for (int kk = 0; kk < k; kk += K_BLOCK) {
        int kblock = kk / K_BLOCK;
        uint8_t raw = h_SFA_mma[size_t(mm) * size_t(num_k_blocks) + size_t(kblock)];
        SFAType v;
        std::memcpy(&v, &raw, 1);
        tSFA(mm, kk, 0) = v;
      }
    }

    // SFB: depends on (n, kblock)
    for (int nn = 0; nn < n; ++nn) {
      for (int kk = 0; kk < k; kk += K_BLOCK) {
        int kblock = kk / K_BLOCK;
        uint8_t raw = h_SFB_mma[size_t(nn) * size_t(num_k_blocks) + size_t(kblock)];
        SFBType v;
        std::memcpy(&v, &raw, 1);
        tSFB(nn, kk, 0) = v;
      }
    }
  }

  // C = 0
  std::fill(block_C.host_data(), block_C.host_data() + cute::size(layout_C), 0.0f);

  // Sync all CUTLASS tensors to device
  block_A_in.sync_device();
  block_A0.sync_device();
  block_A1.sync_device();
  block_A2.sync_device();
  block_B_in.sync_device();
  block_B0.sync_device();
  block_B1.sync_device();
  block_B2.sync_device();
  block_SFA.sync_device();
  block_SFB.sync_device();
  block_C.sync_device();
  block_D.sync_device();

#if defined(RAZER_FULL_OVERLAP_GRAPH)
  auto launch_A_remap_packed = [&](cudaStream_t stream = nullptr) {
#else
  auto launch_A_remap_packed = [&]() {
#endif
    int threads = 128;
    int total_blocks = m * (k / K_BLOCK);
    int blocks = (total_blocks + threads - 1) / threads;
#if defined(RAZER_FULL_OVERLAP_GRAPH)
    build_A_three_packed<<<blocks, threads, 0, stream>>>(
#else
    build_A_three_packed<<<blocks, threads>>>(
#endif
        block_A0.device_data(),
        block_A1.device_data(),
        block_A2.device_data(),
        block_A_in.device_data(),
        d_SFA_raw.get(),
        m, k);
    CUDA_CHECK(cudaGetLastError());
  };

  auto launch_B_remap_packed = [&]() {
    int threads = 128;
    int total_blocks = n * (k / K_BLOCK);
    int blocks = (total_blocks + threads - 1) / threads;
    build_B_three_packed<<<blocks, threads>>>(
        block_B0.device_data(),
        block_B1.device_data(),
        block_B2.device_data(),
        block_B_in.device_data(),
        d_SFB_raw.get(),
        n, k, opt.b_second_magnitude);
    CUDA_CHECK(cudaGetLastError());
  };

  Gemm gemm0;
  Gemm gemm1;
  Gemm gemm2;

  typename Gemm::Arguments args0{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {m, n, k, 1},
    {
      block_A0.device_data(), stride_A,
      block_B0.device_data(), stride_B,
      block_SFA.device_data(),    layout_SFA,
      block_SFB.device_data(),    layout_SFB
    },
    {
      {2.0f, 0.0f},
      block_C.device_data(), stride_C,
      block_D.device_data(), stride_D
    }
  };
  full_configure_scheduler(
      args0, opt.scheduler_swizzle, opt.scheduler_raster);

  typename Gemm::Arguments args1{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {m, n, k, 1},
    {
      block_A1.device_data(), stride_A,
      block_B1.device_data(), stride_B,
      block_SFA.device_data(),    layout_SFA,
      block_SFB.device_data(),    layout_SFB
    },
    {
      {2.0f, 1.0f},
      block_D.device_data(), stride_D,
      block_D.device_data(), stride_D
    }
  };
  full_configure_scheduler(
      args1, opt.scheduler_swizzle, opt.scheduler_raster);

  typename Gemm::Arguments args2{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {m, n, k, 1},
    {
      block_A2.device_data(), stride_A,
      block_B2.device_data(), stride_B,
      block_SFA.device_data(),    layout_SFA,
      block_SFB.device_data(),    layout_SFB
    },
    {
      {-1.0f, 1.0f},
      block_D.device_data(), stride_D,
      block_D.device_data(), stride_D
    }
  };
  full_configure_scheduler(
      args2, opt.scheduler_swizzle, opt.scheduler_raster);

  size_t workspace_bytes = std::max({
      Gemm::get_workspace_size(args0),
      Gemm::get_workspace_size(args1),
      Gemm::get_workspace_size(args2)});
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_bytes);

  CUTLASS_CHECK(gemm0.can_implement(args0));
  CUTLASS_CHECK(gemm1.can_implement(args1));
  CUTLASS_CHECK(gemm2.can_implement(args2));
  CUTLASS_CHECK(gemm0.initialize(args0, workspace.get()));
  CUTLASS_CHECK(gemm1.initialize(args1, workspace.get()));
  CUTLASS_CHECK(gemm2.initialize(args2, workspace.get()));

#if defined(RAZER_FULL_OVERLAP_GRAPH)
  float* overlap_D1 = nullptr;
  float* overlap_D2 = nullptr;
  uint8_t* overlap_workspace0 = nullptr;
  uint8_t* overlap_workspace1 = nullptr;
  uint8_t* overlap_workspace2 = nullptr;
  cudaStream_t overlap_stream0 = nullptr;
  cudaStream_t overlap_stream1 = nullptr;
  cudaStream_t overlap_stream2 = nullptr;
  cudaEvent_t overlap_fork = nullptr;
  cudaEvent_t overlap_done1 = nullptr;
  cudaEvent_t overlap_done2 = nullptr;
  cudaGraph_t overlap_graph = nullptr;
  cudaGraphExec_t overlap_graph_exec = nullptr;
#if defined(RAZER_FULL_OVERLAP_GRAPH)
  if (opt.overlap_graph) {
    size_t output_count = size_t(m) * size_t(n);
    CUDA_CHECK(cudaMalloc(&overlap_D1, output_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&overlap_D2, output_count * sizeof(float)));

    typename Gemm::Arguments overlap_args0{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {
        block_A0.device_data(), stride_A,
        block_B0.device_data(), stride_B,
        block_SFA.device_data(), layout_SFA,
        block_SFB.device_data(), layout_SFB
      },
      {
        {2.0f, 0.0f},
        block_C.device_data(), stride_C,
        block_D.device_data(), stride_D
      }
    };
    full_configure_scheduler(
        overlap_args0, opt.scheduler_swizzle, opt.scheduler_raster);
    typename Gemm::Arguments overlap_args1{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {
        block_A1.device_data(), stride_A,
        block_B1.device_data(), stride_B,
        block_SFA.device_data(), layout_SFA,
        block_SFB.device_data(), layout_SFB
      },
      {
        {2.0f, 0.0f},
        block_C.device_data(), stride_C,
        overlap_D1, stride_D
      }
    };
    full_configure_scheduler(
        overlap_args1, opt.scheduler_swizzle, opt.scheduler_raster);
    typename Gemm::Arguments overlap_args2{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {
        block_A2.device_data(), stride_A,
        block_B2.device_data(), stride_B,
        block_SFA.device_data(), layout_SFA,
        block_SFB.device_data(), layout_SFB
      },
      {
        {-1.0f, 0.0f},
        block_C.device_data(), stride_C,
        overlap_D2, stride_D
      }
    };
    full_configure_scheduler(
        overlap_args2, opt.scheduler_swizzle, opt.scheduler_raster);
    size_t overlap_workspace_bytes0 =
        Gemm::get_workspace_size(overlap_args0);
    size_t overlap_workspace_bytes1 =
        Gemm::get_workspace_size(overlap_args1);
    size_t overlap_workspace_bytes2 =
        Gemm::get_workspace_size(overlap_args2);
    if (overlap_workspace_bytes0 != 0) {
      CUDA_CHECK(cudaMalloc(
          &overlap_workspace0, overlap_workspace_bytes0));
    }
    if (overlap_workspace_bytes1 != 0) {
      CUDA_CHECK(cudaMalloc(
          &overlap_workspace1, overlap_workspace_bytes1));
    }
    if (overlap_workspace_bytes2 != 0) {
      CUDA_CHECK(cudaMalloc(
          &overlap_workspace2, overlap_workspace_bytes2));
    }

    Gemm overlap_gemm0;
    Gemm overlap_gemm1;
    Gemm overlap_gemm2;
    CUTLASS_CHECK(overlap_gemm0.can_implement(overlap_args0));
    CUTLASS_CHECK(overlap_gemm1.can_implement(overlap_args1));
    CUTLASS_CHECK(overlap_gemm2.can_implement(overlap_args2));
    CUTLASS_CHECK(overlap_gemm0.initialize(
        overlap_args0, overlap_workspace0));
    CUTLASS_CHECK(overlap_gemm1.initialize(
        overlap_args1, overlap_workspace1));
    CUTLASS_CHECK(overlap_gemm2.initialize(
        overlap_args2, overlap_workspace2));

    launch_B_remap_packed();
    if (!opt.online_a_remap) launch_A_remap_packed();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &overlap_stream0, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &overlap_stream1, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &overlap_stream2, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(
        &overlap_fork, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(
        &overlap_done1, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(
        &overlap_done2, cudaEventDisableTiming));

    CUDA_CHECK(cudaStreamBeginCapture(
        overlap_stream0, cudaStreamCaptureModeGlobal));
    if (opt.online_a_remap) {
      launch_A_remap_packed(overlap_stream0);
    }
    CUDA_CHECK(cudaEventRecord(overlap_fork, overlap_stream0));
    CUDA_CHECK(cudaStreamWaitEvent(
        overlap_stream1, overlap_fork, 0));
    CUDA_CHECK(cudaStreamWaitEvent(
        overlap_stream2, overlap_fork, 0));
    CUTLASS_CHECK(overlap_gemm0.run(overlap_stream0));
    CUTLASS_CHECK(overlap_gemm1.run(overlap_stream1));
    CUTLASS_CHECK(overlap_gemm2.run(overlap_stream2));
    CUDA_CHECK(cudaEventRecord(overlap_done1, overlap_stream1));
    CUDA_CHECK(cudaEventRecord(overlap_done2, overlap_stream2));
    CUDA_CHECK(cudaStreamWaitEvent(
        overlap_stream0, overlap_done1, 0));
    CUDA_CHECK(cudaStreamWaitEvent(
        overlap_stream0, overlap_done2, 0));
    size_t output_vector_count = output_count / 4;
    int add_threads = 256;
    int add_blocks = int(
        (output_vector_count + size_t(add_threads) - 1) /
        size_t(add_threads));
    full_add_three_output_float4<<<
        add_blocks, add_threads, 0, overlap_stream0>>>(
        reinterpret_cast<float4*>(block_D.device_data()),
        reinterpret_cast<float4 const*>(overlap_D1),
        reinterpret_cast<float4 const*>(overlap_D2),
        output_vector_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamEndCapture(
        overlap_stream0, &overlap_graph));
    CUDA_CHECK(cudaGraphInstantiate(
        &overlap_graph_exec, overlap_graph, nullptr, nullptr, 0));
  }
#endif

  if (opt.correctness) {
    printf("\n=== Correctness pass %dx%dx%d (CPU ref + 1 GPU run) ===\n", m, n, k);

    launch_A_remap_packed();
    launch_B_remap_packed();

#if defined(RAZER_FULL_OVERLAP_GRAPH)
    if (opt.overlap_graph) {
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGraphLaunch(
          overlap_graph_exec, overlap_stream0));
    } else {
#endif
      CUTLASS_CHECK(gemm0.run());
      CUTLASS_CHECK(gemm1.run());
      CUTLASS_CHECK(gemm2.run());
#if defined(RAZER_FULL_OVERLAP_GRAPH)
    }
#endif
    CUDA_CHECK(cudaDeviceSynchronize());

    block_A0.sync_host();
    block_A1.sync_host();
    block_A2.sync_host();
    block_B0.sync_host();
    block_B1.sync_host();
    block_B2.sync_host();
    {
      using namespace cute;
      auto tA0 = make_tensor(make_iterator(block_A0.host_data()), layout_A);
      auto tA1 = make_tensor(make_iterator(block_A1.host_data()), layout_A);
      auto tA2 = make_tensor(make_iterator(block_A2.host_data()), layout_A);

      for (int mm = 0; mm < m; ++mm) {
        for (int kk = 0; kk < k; ++kk) {
          uint8_t a = h_A[size_t(mm) * size_t(k) + size_t(kk)] & 0xFu;
          int kb = kk / K_BLOCK;
          bool neg = (h_SFA_raw[size_t(mm) * size_t(num_k_blocks) + size_t(kb)] & 0x80u) != 0u;
          typename ElementA::DataType a0_dt = tA0(mm, kk, 0);
          typename ElementA::DataType a1_dt = tA1(mm, kk, 0);
          typename ElementA::DataType a2_dt = tA2(mm, kk, 0);
          uint8_t got[3] = {
              uint8_t(uint8_t(a0_dt.raw()) & 0xFu),
              uint8_t(uint8_t(a1_dt.raw()) & 0xFu),
              uint8_t(uint8_t(a2_dt.raw()) & 0xFu)};
          for (int coordinate = 0; coordinate < 3; ++coordinate) {
            uint8_t expected = A_embedding_host(a, neg, coordinate);
            if (got[coordinate] != expected) {
              std::cerr << "A EMBEDDING CHECK FAILED at (m=" << mm
                        << ", k=" << kk << ", coordinate=" << coordinate << ")\n"
                        << "  raw=" << int(a) << " neg=" << neg
                        << " expected=" << int(expected)
                        << " got=" << int(got[coordinate]) << "\n";
              return 2;
            }
          }
        }
      }
      std::cout << "A embedding check: PASS\n";
    }

    {
      using namespace cute;
      auto tB0 = make_tensor(make_iterator(block_B0.host_data()), layout_B);
      auto tB1 = make_tensor(make_iterator(block_B1.host_data()), layout_B);
      auto tB2 = make_tensor(make_iterator(block_B2.host_data()), layout_B);

      for (int kk = 0; kk < k; ++kk) {
        for (int nn = 0; nn < n; ++nn) {
          size_t idx = size_t(kk) * size_t(n) + size_t(nn);
          uint8_t b = h_B[idx] & 0xFu;

          int kb = kk / K_BLOCK;
          uint8_t meta = h_SFB_raw[size_t(nn) * size_t(num_k_blocks) + size_t(kb)];
          bool neg = (meta & 0x80u) != 0u;
          bool use_second_magnitude = (meta & 0x40u) != 0u;
          typename ElementB::DataType b0_dt = tB0(nn, kk, 0);
          typename ElementB::DataType b1_dt = tB1(nn, kk, 0);
          typename ElementB::DataType b2_dt = tB2(nn, kk, 0);
          uint8_t got[3] = {
              uint8_t(uint8_t(b0_dt.raw()) & 0xFu),
              uint8_t(uint8_t(b1_dt.raw()) & 0xFu),
              uint8_t(uint8_t(b2_dt.raw()) & 0xFu)};
          for (int coordinate = 0; coordinate < 3; ++coordinate) {
            uint8_t expected =
                B_embedding_host(
                    b, neg, use_second_magnitude,
                    opt.b_second_magnitude, coordinate);
            if (got[coordinate] != expected) {
              std::cerr << "B EMBEDDING CHECK FAILED at (k=" << kk
                        << ", n=" << nn << ", coordinate=" << coordinate << ")\n"
                        << "  raw=" << int(b)
                        << " meta=0x" << std::hex << int(meta) << std::dec
                        << " (neg=" << neg
                        << ", use_second_magnitude="
                        << use_second_magnitude
                        << ", second_magnitude="
                        << opt.b_second_magnitude << ")\n"
                        << "  expected=" << int(expected)
                        << " got=" << int(got[coordinate]) << "\n";
              return 2;
            }
          }
        }
      }
      std::cout << "B embedding check: PASS\n";
    }

    block_D.sync_host();

    using namespace cute;
    auto tSFA = make_tensor(block_SFA.host_data(), layout_SFA);
    auto tSFB = make_tensor(block_SFB.host_data(), layout_SFB);
    auto tD = make_tensor(block_D.host_data(), layout_D);

    std::vector<float> D_gpu(size_t(m) * size_t(n));
    std::vector<float> D_ref(size_t(m) * size_t(n));

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        D_gpu[size_t(i) * size_t(n) + size_t(j)] = tD(i, j, 0);
      }
    }

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (int kk = 0; kk < k; ++kk) {
          uint8_t sfa_u8 = 0;
          uint8_t sfb_u8 = 0;
          std::memcpy(&sfa_u8, &tSFA(i, kk, 0), 1);
          std::memcpy(&sfb_u8, &tSFB(j, kk, 0), 1);
          float sa = decode_ue4m3(sfa_u8);
          float sb = decode_ue4m3(sfb_u8);

          uint8_t a_nib = h_A[size_t(i) * size_t(k) + size_t(kk)] & 0xFu;
          uint8_t b_nib = h_B[size_t(kk) * size_t(n) + size_t(j)] & 0xFu;
          int kb = kk / K_BLOCK;
          uint8_t a_meta = h_SFA_raw[size_t(i) * size_t(num_k_blocks) + size_t(kb)];
          uint8_t b_meta = h_SFB_raw[size_t(j) * size_t(num_k_blocks) + size_t(kb)];

          float a_coeff = decode_fp4_e2m1(a_nib);
          if (a_nib == 0x0u) {
            a_coeff = (a_meta & 0x80u) ? -5.0f : 5.0f;
          }
          float b_coeff = decode_fp4_e2m1(b_nib);
          if (b_nib == 0x0u) {
            float magnitude = (b_meta & 0x40u)
                ? float(opt.b_second_magnitude)
                : 5.0f;
            b_coeff = (b_meta & 0x80u) ? -magnitude : magnitude;
          }
          acc += (a_coeff * sa) * (b_coeff * sb);
        }
        D_ref[size_t(i) * size_t(n) + size_t(j)] = acc;
      }
    }

    float max_abs_err = 0.f;
    float max_rel_err = 0.f;
    float max_abs_ref = 0.f;
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;
    size_t nonfinite = 0;
    for (int i = 0; i < m * n; ++i) {
      if (!std::isfinite(D_ref[i]) || !std::isfinite(D_gpu[i])) {
        ++nonfinite;
        continue;
      }
      float diff = fabsf(D_ref[i] - D_gpu[i]);
      if (diff > max_abs_err) max_abs_err = diff;
      if (fabsf(D_ref[i]) > max_abs_ref) max_abs_ref = fabsf(D_ref[i]);
      float denom = fmaxf(1.0f, fabsf(D_ref[i]));
      float rel = diff / denom;
      if (rel > max_rel_err) max_rel_err = rel;
      sum_sq_err += double(diff) * double(diff);
      sum_sq_ref += double(D_ref[i]) * double(D_ref[i]);
    }
    double normalized_max_error = double(max_abs_err) / std::max(1.0, double(max_abs_ref));
    double relative_l2_error = std::sqrt(sum_sq_err / std::max(1.0, sum_sq_ref));
    printf("Correctness: max_abs=%e max_point_rel=%e normalized_max=%e relative_l2=%e nonfinite=%zu\n",
           max_abs_err, max_rel_err, normalized_max_error, relative_l2_error, nonfinite);

    int start_r = 0;
    int start_c = 0;
    print_matrix_window("D_ref (correctness)", D_ref.data(), m, n, start_r, start_c);
    print_matrix_window("D_mma (correctness)", D_gpu.data(), m, n, start_r, start_c);

    constexpr int kViewRows = 4;
    constexpr int kViewColumns = 4;
    float diff_block[kViewRows * kViewColumns];
    for (int r = 0; r < kViewRows && r < m; ++r) {
      for (int c = 0; c < kViewColumns && c < n; ++c) {
        int rr = start_r + r;
        int cc = start_c + c;
        diff_block[r * kViewColumns + c] =
            fabsf(D_ref[rr * n + cc] - D_gpu[rr * n + cc]);
      }
    }
    printf("Abs diff (same correctness window at [%d,%d]) =\n", start_r, start_c);
    for (int r = 0; r < kViewRows && r < m; ++r) {
      printf("  ");
      for (int c = 0; c < kViewColumns && c < n; ++c) {
        printf(
            "%12.3f ",
            static_cast<double>(diff_block[r * kViewColumns + c]));
      }
      printf("\n");
    }
    printf("\n");

    if (nonfinite != 0 || normalized_max_error > opt.max_normalized_error) {
      std::cerr << "CORRECTNESS FAILED: normalized_max_error=" << normalized_max_error
                << " threshold=" << opt.max_normalized_error
                << " nonfinite=" << nonfinite << "\n";
      return 3;
    }
    std::cout << "Correctness threshold: PASS\n";
  }

  if (opt.overlap_graph) {
    const size_t flush_bytes =
        size_t(opt.flush_mb) * 1024ull * 1024ull;
    unsigned char* flush_buffer = nullptr;
    if (flush_bytes != 0) {
      CUDA_CHECK(cudaMalloc(&flush_buffer, flush_bytes));
    }
    auto flush = [&]() {
      if (flush_buffer != nullptr) {
        flush_cache_kernel<<<1024, 256>>>(
            flush_buffer, flush_bytes);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    };

    for (int iteration = 0; iteration < opt.warmup; ++iteration) {
      flush();
      CUDA_CHECK(cudaGraphLaunch(
          overlap_graph_exec, overlap_stream0));
      CUDA_CHECK(cudaStreamSynchronize(overlap_stream0));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> samples;
    samples.reserve(size_t(opt.iters));
    for (int iteration = 0; iteration < opt.iters; ++iteration) {
      flush();
      CUDA_CHECK(cudaEventRecord(start, overlap_stream0));
      CUDA_CHECK(cudaGraphLaunch(
          overlap_graph_exec, overlap_stream0));
      CUDA_CHECK(cudaEventRecord(stop, overlap_stream0));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float elapsed = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
      samples.push_back(double(elapsed));
    }

    std::vector<double> sorted_samples = samples;
    std::sort(sorted_samples.begin(), sorted_samples.end());
    double sum = 0.0;
    for (double sample : samples) sum += sample;
    double best = sorted_samples.front();
    double median = sorted_samples[sorted_samples.size() / 2];
    double p90 = sorted_samples[
        size_t(std::floor(0.9 * double(sorted_samples.size() - 1)))];
    double average = sum / double(samples.size());
    double effective_operations =
        2.0 * double(m) * double(n) * double(k);
    std::cout << "METHOD=full-razer-overlap-graph\n"
              << "M,N,K = " << m << "," << n << "," << k << "\n"
              << std::fixed << std::setprecision(6)
              << "Best: " << best << " ms, Median: " << median
              << " ms, P90: " << p90 << " ms, Avg: " << average
              << " ms over " << opt.iters << " iters (warmup "
              << opt.warmup << ")\n"
              << "Timing includes: "
              << (opt.online_a_remap
                      ? "online A remap + "
                      : "prepacked A + ")
              << "graphed concurrent 3x GEMM + FP32 output add; "
              << "static B preprocessing excluded; flush_mb="
              << opt.flush_mb << "\n"
              << std::setprecision(2)
              << "Effective logical throughput: avg "
              << effective_operations / (average * 1.0e9)
              << " TFLOPs (2*M*N*K logical ops; actual tensor work is 3x)\n";
    std::cout << "SAMPLES_MS=";
    for (size_t index = 0; index < samples.size(); ++index) {
      if (index != 0) std::cout << ",";
      std::cout << std::setprecision(9) << samples[index];
    }
    std::cout << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (flush_buffer != nullptr) CUDA_CHECK(cudaFree(flush_buffer));
    CUDA_CHECK(cudaGraphExecDestroy(overlap_graph_exec));
    CUDA_CHECK(cudaGraphDestroy(overlap_graph));
    CUDA_CHECK(cudaEventDestroy(overlap_fork));
    CUDA_CHECK(cudaEventDestroy(overlap_done1));
    CUDA_CHECK(cudaEventDestroy(overlap_done2));
    CUDA_CHECK(cudaStreamDestroy(overlap_stream0));
    CUDA_CHECK(cudaStreamDestroy(overlap_stream1));
    CUDA_CHECK(cudaStreamDestroy(overlap_stream2));
    if (overlap_workspace0 != nullptr) {
      CUDA_CHECK(cudaFree(overlap_workspace0));
    }
    if (overlap_workspace1 != nullptr) {
      CUDA_CHECK(cudaFree(overlap_workspace1));
    }
    if (overlap_workspace2 != nullptr) {
      CUDA_CHECK(cudaFree(overlap_workspace2));
    }
    CUDA_CHECK(cudaFree(overlap_D1));
    CUDA_CHECK(cudaFree(overlap_D2));
    return 0;
  }
#endif

  {
    const int WARMUP_ITERS  = opt.warmup;
    const int MEASURE_ITERS = opt.iters;

    const size_t FLUSH_BYTES = size_t(opt.flush_mb) * 1024ull * 1024ull;
    unsigned char* d_flush = nullptr;
    if (FLUSH_BYTES != 0) CUDA_CHECK(cudaMalloc(&d_flush, FLUSH_BYTES));
    dim3 flush_block(256);
    dim3 flush_grid(1024);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Weight preprocessing is static. Activation preprocessing is either static for the
    // GEMM-only mode or included in each iteration for the online-remap mode.
    launch_B_remap_packed();
    if (!opt.online_a_remap) launch_A_remap_packed();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
      if (d_flush) {
        flush_cache_kernel<<<flush_grid, flush_block>>>(d_flush, FLUSH_BYTES);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
      if (opt.online_a_remap) launch_A_remap_packed();
      CUTLASS_CHECK(gemm0.run());
      CUTLASS_CHECK(gemm1.run());
      CUTLASS_CHECK(gemm2.run());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    double sum_ms = 0.0;
    double best_ms = 1e30;
    std::vector<double> samples_ms;
    samples_ms.reserve(size_t(MEASURE_ITERS));

    for (int i = 0; i < MEASURE_ITERS; ++i) {
      if (d_flush) {
        flush_cache_kernel<<<flush_grid, flush_block>>>(d_flush, FLUSH_BYTES);
        CUDA_CHECK(cudaDeviceSynchronize());
      }

      CUDA_CHECK(cudaEventRecord(start));

      if (opt.online_a_remap) launch_A_remap_packed();
      CUTLASS_CHECK(gemm0.run());
      CUTLASS_CHECK(gemm1.run());
      CUTLASS_CHECK(gemm2.run());
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));

      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      sum_ms += ms;
      samples_ms.push_back(double(ms));
      if (ms < best_ms) best_ms = ms;
    }

    double avg_ms = sum_ms / double(MEASURE_ITERS);
    std::vector<double> sorted_samples = samples_ms;
    std::sort(sorted_samples.begin(), sorted_samples.end());
    double median_ms = sorted_samples[sorted_samples.size() / 2];
    double p90_ms = sorted_samples[size_t(std::floor(0.9 * double(sorted_samples.size() - 1)))];
    double effective_ops = 2.0 * double(m) * double(n) * double(k);
    double tflops_best = effective_ops / (best_ms * 1e9);
    double tflops_avg  = effective_ops / (avg_ms  * 1e9);

    printf("M,N,K = %d,%d,%d\n", m, n, k);
    printf("Best: %.6f ms, Median: %.6f ms, P90: %.6f ms, Avg: %.6f ms over %d iters (warmup %d)\n",
           best_ms, median_ms, p90_ms, avg_ms, MEASURE_ITERS, WARMUP_ITERS);
    printf("Timing includes: %s + 3x GEMM; static B preprocessing excluded; flush_mb=%d\n",
           opt.online_a_remap ? "online A remap" : "prepacked A", opt.flush_mb);
    printf("Effective throughput: best %.2f TFLOPs, avg %.2f TFLOPs (2*M*N*K ops)\n\n", tflops_best, tflops_avg);
    std::cout << "SAMPLES_MS=";
    for (size_t i = 0; i < samples_ms.size(); ++i) {
      if (i != 0) std::cout << ",";
      std::cout << std::setprecision(9) << samples_ms[i];
    }
    std::cout << "\n";

    if (opt.breakdown) {
      cudaEvent_t e0, e1, e2, e3, e4;
      CUDA_CHECK(cudaEventCreate(&e0));
      CUDA_CHECK(cudaEventCreate(&e1));
      CUDA_CHECK(cudaEventCreate(&e2));
      CUDA_CHECK(cudaEventCreate(&e3));
      CUDA_CHECK(cudaEventCreate(&e4));

      // One extra iteration to estimate per-stage GPU time.
      if (d_flush) {
        flush_cache_kernel<<<flush_grid, flush_block>>>(d_flush, FLUSH_BYTES);
        CUDA_CHECK(cudaDeviceSynchronize());
      }

      CUDA_CHECK(cudaEventRecord(e0));
      if (opt.online_a_remap) launch_A_remap_packed();
      CUDA_CHECK(cudaEventRecord(e1));
      CUTLASS_CHECK(gemm0.run());
      CUDA_CHECK(cudaEventRecord(e2));
      CUTLASS_CHECK(gemm1.run());
      CUDA_CHECK(cudaEventRecord(e3));
      CUTLASS_CHECK(gemm2.run());
      CUDA_CHECK(cudaEventRecord(e4));
      CUDA_CHECK(cudaEventSynchronize(e4));

      float ms_remap = 0.f, ms_gemm0 = 0.f, ms_gemm1 = 0.f, ms_gemm2 = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms_remap, e0, e1));
      CUDA_CHECK(cudaEventElapsedTime(&ms_gemm0, e1, e2));
      CUDA_CHECK(cudaEventElapsedTime(&ms_gemm1, e2, e3));
      CUDA_CHECK(cudaEventElapsedTime(&ms_gemm2, e3, e4));
      printf("Stage breakdown (1 iter): a_remap %.6f ms, gemm0 %.6f ms, gemm1 %.6f ms, gemm2 %.6f ms, sum %.6f ms\n\n",
             ms_remap, ms_gemm0, ms_gemm1, ms_gemm2,
             double(ms_remap) + double(ms_gemm0) + double(ms_gemm1) + double(ms_gemm2));

      CUDA_CHECK(cudaEventDestroy(e0));
      CUDA_CHECK(cudaEventDestroy(e1));
      CUDA_CHECK(cudaEventDestroy(e2));
      CUDA_CHECK(cudaEventDestroy(e3));
      CUDA_CHECK(cudaEventDestroy(e4));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (d_flush) CUDA_CHECK(cudaFree(d_flush));
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const** argv) {
  Options opt;
  opt.parse(argc, argv);
  if (opt.help) { opt.usage(); return 0; }
  if (!opt.valid()) {
    std::cerr << "Invalid or missing required argument.\n";
    opt.usage();
    return 1;
  }
  return run_case(opt);
}
