/***************************************************************************************************
  Native-NVFP4 activations x RaZeR weights on current Blackwell hardware.

  Weight decomposition for q in {5, second_magnitude}:

    ordinary y: (main, remainder) = (y, 0)
    +5:         (4, 1)
    +7:         (4, 3)
    +8:         (4, 4)
    +9:         (6, 3)

  Negative specials negate both coordinates.  Execution modes include:

    two-pass:         A*B_main followed by A*B_remainder + D
    two-pass-overlap: concurrent A*B_main and A*B_remainder, then an FP32 add
    two-pass-overlap-graph: the same DAG captured as one reusable CUDA graph
    concat:           [A A] * [B_main; B_remainder], one GEMM with K'=2K
    concat-n:         A * [B_main B_remainder], one GEMM with N'=2N, then add
  Weight decomposition is static. In concat mode, activation duplication can
  be prepacked or included in the timed region with --online-a-duplicate.

***************************************************************************************************/

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "razer_w4a4_common.cuh"

#ifndef RAZER_SPLIT_K
#define RAZER_SPLIT_K 2
#endif

static_assert(
    RAZER_SPLIT_K >= 2,
    "RAZER_SPLIT_K must be at least 2 for Split-K modes.");

template <typename Arguments>
static inline void weight_configure_scheduler(
    Arguments& arguments, int runtime_swizzle,
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
  if (runtime_swizzle != -1) {
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
}

#if defined(RAZER_STREAM_K)
template <typename Arguments>
static inline void weight_configure_split_k(
    Arguments& arguments, int splits, bool deterministic) {
  using StreamKParams =
      cutlass::gemm::kernel::detail::
          PersistentTileSchedulerSm90StreamKParams;
  arguments.scheduler.splits = splits;
  arguments.scheduler.decomposition_mode =
      splits == 1
          ? StreamKParams::DecompositionMode::DataParallel
          : StreamKParams::DecompositionMode::SplitK;
  arguments.scheduler.reduction_mode =
      deterministic
          ? StreamKParams::ReductionMode::Deterministic
          : StreamKParams::ReductionMode::Nondeterministic;
}
#endif

struct WeightOptions {
  std::string mode;
  int m = -1;
  int n = -1;
  int k = -1;
  int warmup = -1;
  int iters = -1;
  int flush_mb = -1;
  int64_t seed = -1;
  int second_magnitude = -1;
  double weight_special_rate = -1.0;
  double max_normalized_error = -1.0;
  int scheduler_swizzle = -1;
  std::string scheduler_raster;
  bool check = false;
  bool breakdown = false;
  bool online_a_duplicate = false;
  bool help = false;

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);
    help = cmd.check_cmd_line_flag("help");
    check = cmd.check_cmd_line_flag("check");
    breakdown = cmd.check_cmd_line_flag("breakdown");
    online_a_duplicate = cmd.check_cmd_line_flag("online-a-duplicate");
    cmd.get_cmd_line_argument("mode", mode);
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("warmup", warmup);
    cmd.get_cmd_line_argument("iters", iters);
    cmd.get_cmd_line_argument("flush-mb", flush_mb);
    cmd.get_cmd_line_argument("seed", seed);
    cmd.get_cmd_line_argument("b-second-magnitude", second_magnitude);
    cmd.get_cmd_line_argument("weight-special-rate", weight_special_rate);
    cmd.get_cmd_line_argument("max-normalized-error", max_normalized_error);
    cmd.get_cmd_line_argument("scheduler-swizzle", scheduler_swizzle);
    cmd.get_cmd_line_argument("scheduler-raster", scheduler_raster);
  }

  void usage() const {
    std::cout
      << "razer_weight_w4a4\n"
        << "  Required: --mode=<two-pass|two-pass-overlap|"
           "two-pass-overlap-graph|concat|concat-n|concat-n-graph"
#if defined(RAZER_STREAM_K)
           "|concat-splitk|concat-splitk-atomic|concat-splitk-graph|"
           "concat-n-splitk|"
           "concat-n-splitk-atomic|concat-n-splitk-graph"
#endif
           ">\n"
        << "            --m=<int> --n=<int> --k=<int>\n"
        << "            --warmup=<int> --iters=<int> --flush-mb=<int>\n"
        << "            --seed=<nonnegative int64>\n"
        << "            --b-second-magnitude=<7|8|9>\n"
        << "            --weight-special-rate=<0..1>\n"
        << "  Optional: --online-a-duplicate (concat only)\n"
        << "            --scheduler-swizzle=<1|2|4|8> and "
           "--scheduler-raster=<heuristic|along-m|along-n>\n"
        << "            --breakdown\n"
        << "            --check --max-normalized-error=<positive float>\n";
  }

  bool valid() const {
    if (mode != "two-pass" && mode != "two-pass-overlap" &&
        mode != "two-pass-overlap-graph" &&
        mode != "concat" && mode != "concat-n" &&
        mode != "concat-n-graph"
#if defined(RAZER_STREAM_K)
        && mode != "concat-splitk" && mode != "concat-splitk-atomic" &&
        mode != "concat-splitk-graph" &&
        mode != "concat-n-splitk" &&
        mode != "concat-n-splitk-atomic" &&
        mode != "concat-n-splitk-graph"
#endif
        ) return false;
    if (m <= 0 || n <= 0 || k <= 0) return false;
    if (warmup < 0 || iters <= 0 || flush_mb < 0 || seed < 0) return false;
    if (second_magnitude != 7 && second_magnitude != 8 && second_magnitude != 9) return false;
    if (weight_special_rate < 0.0 || weight_special_rate > 1.0) return false;
    if (check && !(max_normalized_error > 0.0)) return false;
    if (!check && max_normalized_error >= 0.0) return false;
    auto scheduler_pair_valid =
        [](int swizzle, std::string const& raster) {
          bool any = swizzle != -1 || !raster.empty();
          if (!any) return true;
          return (swizzle == 1 || swizzle == 2 ||
                  swizzle == 4 || swizzle == 8) &&
              (raster == "heuristic" ||
               raster == "along-m" ||
               raster == "along-n");
        };
    bool scheduler_override =
        scheduler_swizzle != -1 || !scheduler_raster.empty();
    if (!scheduler_pair_valid(scheduler_swizzle, scheduler_raster)) {
      return false;
    }
#if defined(RAZER_MAX_SWIZZLE) || defined(RAZER_RASTER_ALONG_M) || \
    defined(RAZER_RASTER_ALONG_N)
    if (scheduler_override) return false;
#endif
#if defined(RAZER_STREAM_K)
    if (mode == "concat-splitk-graph" &&
        (online_a_duplicate || breakdown)) return false;
#endif
    if (online_a_duplicate && mode != "concat"
#if defined(RAZER_STREAM_K)
        && mode != "concat-splitk" && mode != "concat-splitk-atomic"
#endif
        ) return false;
    return true;
  }
};

struct WeightHostData {
  std::vector<uint8_t> a_packed;     // row-major MxK, two nibbles per byte
  std::vector<uint8_t> b_packed;     // CUTLASS B physical order: N columns, K contiguous
  std::vector<uint8_t> sfa_mma;      // dense [M,K/16], numeric UE4M3
  std::vector<uint8_t> sfb_raw;      // dense [N,K/16], metadata + E3M3 payload
  std::vector<uint8_t> sfb_mma;      // dense [N,K/16], numeric UE4M3
  size_t special_count = 0;
};

static inline uint8_t packed_nibble(
    std::vector<uint8_t> const& packed, size_t logical_index) {
  uint8_t byte = packed[logical_index >> 1];
  return (logical_index & 1u) ? uint8_t(byte >> 4) : uint8_t(byte & 0xFu);
}

static inline void set_packed_nibble(
    std::vector<uint8_t>& packed, size_t logical_index, uint8_t nibble) {
  uint8_t& byte = packed[logical_index >> 1];
  if (logical_index & 1u) {
    byte = uint8_t((byte & 0x0Fu) | ((nibble & 0xFu) << 4));
  } else {
    byte = uint8_t((byte & 0xF0u) | (nibble & 0xFu));
  }
}

static WeightHostData make_weight_data(WeightOptions const& opt) {
  int k_blocks = opt.k / K_BLOCK;
  WeightHostData data;
  data.a_packed.resize(size_t(opt.m) * size_t(opt.k) / 2);
  data.b_packed.resize(size_t(opt.n) * size_t(opt.k) / 2);
  data.sfa_mma.resize(size_t(opt.m) * size_t(k_blocks));
  data.sfb_raw.resize(size_t(opt.n) * size_t(k_blocks));
  data.sfb_mma.resize(size_t(opt.n) * size_t(k_blocks));

  std::mt19937_64 rng(uint64_t(opt.seed));
  std::uniform_int_distribution<int> a_nibble(0, 15);
  std::uniform_int_distribution<int> ordinary_b_nibble(1, 15);
  std::uniform_int_distribution<int> bit(0, 1);
  std::uniform_int_distribution<int> scale_choice(0, 3);
  std::bernoulli_distribution is_special(opt.weight_special_rate);

  for (size_t logical = 0; logical < size_t(opt.m) * size_t(opt.k); ++logical) {
    set_packed_nibble(data.a_packed, logical, uint8_t(a_nibble(rng)));
  }
  for (int nn = 0; nn < opt.n; ++nn) {
    for (int kk = 0; kk < opt.k; ++kk) {
      bool special = is_special(rng);
      uint8_t value = special ? 0x0u : uint8_t(ordinary_b_nibble(rng));
      set_packed_nibble(
          data.b_packed, size_t(nn) * size_t(opt.k) + size_t(kk), value);
      data.special_count += size_t(special);
    }
  }

  // Explicit finite scale test set: 0.5, 1.0, 1.5, and 2.0.
  constexpr uint8_t ue4m3_scales[4] = {0x30u, 0x38u, 0x3Cu, 0x40u};
  constexpr uint8_t e3m3_payloads[4] = {0x10u, 0x18u, 0x1Cu, 0x20u};
  for (uint8_t& scale : data.sfa_mma) {
    scale = ue4m3_scales[scale_choice(rng)];
  }
  for (size_t index = 0; index < data.sfb_raw.size(); ++index) {
    uint8_t payload = e3m3_payloads[scale_choice(rng)];
    uint8_t metadata = uint8_t((bit(rng) ? 0x80u : 0u) | (bit(rng) ? 0x40u : 0u));
    data.sfb_raw[index] = uint8_t(payload | metadata);
    data.sfb_mma[index] = uint8_t(payload + 0x20u);
  }
  return data;
}

__host__ __device__ static inline void weight_decompose(
    uint8_t raw, uint8_t metadata, int second_magnitude,
    uint8_t& main_value, uint8_t& remainder_value) {
  raw &= 0xFu;
  if (raw != 0x0u) {
    main_value = raw;
    remainder_value = 0x0u;
    return;
  }
  bool negative = (metadata & 0x80u) != 0u;
  bool use_second = (metadata & 0x40u) != 0u;
  int magnitude = use_second ? second_magnitude : 5;
  uint8_t positive_main = magnitude == 9 ? 0x7u : 0x6u;  // +6 or +4
  uint8_t positive_remainder =
      magnitude == 5 ? 0x2u :       // +1
      magnitude == 7 ? 0x5u :       // +3
      magnitude == 8 ? 0x6u : 0x5u; // +4 or +3
  main_value = negative ? uint8_t(positive_main ^ 0x8u) : positive_main;
  remainder_value =
      negative ? uint8_t(positive_remainder ^ 0x8u) : positive_remainder;
}

__global__ void weight_build_B_two_packed(
    typename ElementB::DataType* b_main,
    typename ElementB::DataType* b_remainder,
    typename ElementB::DataType const* b_input,
    uint8_t const* sfb_raw,
    int n, int k, int second_magnitude) {
  int k_blocks = k / K_BLOCK;
  int block_index = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (block_index >= n * k_blocks) return;
  int nn = block_index / k_blocks;
  int kb = block_index - nn * k_blocks;
  int byte_offset = nn * (k / 2) + kb * 8;
  uint64_t input =
      *reinterpret_cast<uint64_t const*>(
          reinterpret_cast<uint8_t const*>(b_input) + byte_offset);
  uint64_t output_main = 0;
  uint64_t output_remainder = 0;
  uint8_t metadata = sfb_raw[nn * k_blocks + kb];
  #pragma unroll
  for (int byte_index = 0; byte_index < 8; ++byte_index) {
    uint8_t byte = uint8_t(input >> (8 * byte_index));
    uint8_t m0, r0, m1, r1;
    weight_decompose(
        byte & 0xFu, metadata, second_magnitude, m0, r0);
    weight_decompose(
        (byte >> 4) & 0xFu, metadata, second_magnitude, m1, r1);
    output_main |= uint64_t(uint8_t(m0 | (m1 << 4))) << (8 * byte_index);
    output_remainder |= uint64_t(uint8_t(r0 | (r1 << 4))) << (8 * byte_index);
  }
  *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(b_main) + byte_offset) =
      output_main;
  *reinterpret_cast<uint64_t*>(
      reinterpret_cast<uint8_t*>(b_remainder) + byte_offset) = output_remainder;
}

__global__ void weight_duplicate_A_concat(
    typename ElementA::DataType* a_concat,
    typename ElementA::DataType const* a_input,
    int m, int k) {
  int k_blocks = k / K_BLOCK;
  int block_index = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (block_index >= m * k_blocks) return;
  int mm = block_index / k_blocks;
  int kb = block_index - mm * k_blocks;
  int bytes_per_segment = k / 2;
  int input_offset = mm * bytes_per_segment + kb * 8;
  int output_row = mm * (2 * bytes_per_segment);
  uint64_t value =
      *reinterpret_cast<uint64_t const*>(
          reinterpret_cast<uint8_t const*>(a_input) + input_offset);
  uint8_t* output = reinterpret_cast<uint8_t*>(a_concat);
  *reinterpret_cast<uint64_t*>(
      output + output_row + kb * 8) = value;
  *reinterpret_cast<uint64_t*>(
      output + output_row + bytes_per_segment + kb * 8) = value;
}

__global__ void weight_build_B_concat(
    typename ElementB::DataType* b_concat,
    typename ElementB::DataType const* b_input,
    uint8_t const* sfb_raw,
    int n, int k, int second_magnitude) {
  int k_blocks = k / K_BLOCK;
  int block_index = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (block_index >= n * k_blocks) return;
  int nn = block_index / k_blocks;
  int kb = block_index - nn * k_blocks;
  int bytes_per_segment = k / 2;
  int input_offset = nn * bytes_per_segment + kb * 8;
  int output_column = nn * (2 * bytes_per_segment);
  uint64_t input =
      *reinterpret_cast<uint64_t const*>(
          reinterpret_cast<uint8_t const*>(b_input) + input_offset);
  uint64_t output_main = 0;
  uint64_t output_remainder = 0;
  uint8_t metadata = sfb_raw[nn * k_blocks + kb];
  #pragma unroll
  for (int byte_index = 0; byte_index < 8; ++byte_index) {
    uint8_t byte = uint8_t(input >> (8 * byte_index));
    uint8_t m0, r0, m1, r1;
    weight_decompose(
        byte & 0xFu, metadata, second_magnitude, m0, r0);
    weight_decompose(
        (byte >> 4) & 0xFu, metadata, second_magnitude, m1, r1);
    output_main |= uint64_t(uint8_t(m0 | (m1 << 4))) << (8 * byte_index);
    output_remainder |= uint64_t(uint8_t(r0 | (r1 << 4))) << (8 * byte_index);
  }
  uint8_t* output = reinterpret_cast<uint8_t*>(b_concat);
  *reinterpret_cast<uint64_t*>(
      output + output_column + kb * 8) = output_main;
  *reinterpret_cast<uint64_t*>(
      output + output_column + bytes_per_segment + kb * 8) = output_remainder;
}

__global__ void weight_build_B_concat_n(
    typename ElementB::DataType* b_concat_n,
    typename ElementB::DataType const* b_input,
    uint8_t const* sfb_raw,
    int n, int k, int second_magnitude) {
  int k_blocks = k / K_BLOCK;
  int block_index = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (block_index >= n * k_blocks) return;
  int nn = block_index / k_blocks;
  int kb = block_index - nn * k_blocks;
  int bytes_per_column = k / 2;
  int input_offset = nn * bytes_per_column + kb * 8;
  uint64_t input =
      *reinterpret_cast<uint64_t const*>(
          reinterpret_cast<uint8_t const*>(b_input) + input_offset);
  uint64_t output_main = 0;
  uint64_t output_remainder = 0;
  uint8_t metadata = sfb_raw[nn * k_blocks + kb];
  #pragma unroll
  for (int byte_index = 0; byte_index < 8; ++byte_index) {
    uint8_t byte = uint8_t(input >> (8 * byte_index));
    uint8_t m0, r0, m1, r1;
    weight_decompose(
        byte & 0xFu, metadata, second_magnitude, m0, r0);
    weight_decompose(
        (byte >> 4) & 0xFu, metadata, second_magnitude, m1, r1);
    output_main |=
        uint64_t(uint8_t(m0 | (m1 << 4))) << (8 * byte_index);
    output_remainder |=
        uint64_t(uint8_t(r0 | (r1 << 4))) << (8 * byte_index);
  }
  uint8_t* output = reinterpret_cast<uint8_t*>(b_concat_n);
  *reinterpret_cast<uint64_t*>(
      output + nn * bytes_per_column + kb * 8) = output_main;
  *reinterpret_cast<uint64_t*>(
      output + (n + nn) * bytes_per_column + kb * 8) = output_remainder;
}

using WeightSFAType = ElementA::ScaleFactorType;
using WeightSFBType = ElementB::ScaleFactorType;
using WeightAPacked = typename ElementA::DataType;
using WeightBPacked = typename ElementB::DataType;

static inline WeightAPacked* as_a(uint8_t* pointer) {
  return reinterpret_cast<WeightAPacked*>(pointer);
}
static inline WeightBPacked* as_b(uint8_t* pointer) {
  return reinterpret_cast<WeightBPacked*>(pointer);
}

struct SampleSummary {
  double best;
  double median;
  double p90;
  double average;
};

static SampleSummary summarize(std::vector<double> const& samples) {
  if (samples.empty()) std::abort();
  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  double sum = 0.0;
  for (double sample : samples) sum += sample;
  return {
      sorted.front(),
      sorted[sorted.size() / 2],
      sorted[size_t(std::floor(0.9 * double(sorted.size() - 1)))],
      sum / double(samples.size())};
}

static void print_samples(std::vector<double> const& samples) {
  std::cout << "SAMPLES_MS=";
  for (size_t index = 0; index < samples.size(); ++index) {
    if (index) std::cout << ",";
    std::cout << std::setprecision(9) << samples[index];
  }
  std::cout << "\n";
}

static int validate_output(
    WeightOptions const& opt,
    WeightHostData const& host,
    std::vector<float> const& output) {
  int k_blocks = opt.k / K_BLOCK;
  float max_abs_reference = 0.0f;
  float max_abs_error = 0.0f;
  double squared_reference = 0.0;
  double squared_error = 0.0;
  size_t nonfinite = 0;
  for (int mm = 0; mm < opt.m; ++mm) {
    for (int nn = 0; nn < opt.n; ++nn) {
      float reference = 0.0f;
      for (int kk = 0; kk < opt.k; ++kk) {
        int kb = kk / K_BLOCK;
        float scale_a = decode_ue4m3(
            host.sfa_mma[size_t(mm) * size_t(k_blocks) + size_t(kb)]);
        float scale_b = decode_ue4m3(
            host.sfb_mma[size_t(nn) * size_t(k_blocks) + size_t(kb)]);
        uint8_t a_raw = packed_nibble(
            host.a_packed, size_t(mm) * size_t(opt.k) + size_t(kk));
        uint8_t b_raw = packed_nibble(
            host.b_packed, size_t(nn) * size_t(opt.k) + size_t(kk));
        float a = decode_fp4_e2m1(a_raw);
        float b = decode_fp4_e2m1(b_raw);
        if (b_raw == 0x0u) {
          uint8_t metadata =
              host.sfb_raw[size_t(nn) * size_t(k_blocks) + size_t(kb)];
          float magnitude = (metadata & 0x40u)
              ? float(opt.second_magnitude) : 5.0f;
          b = (metadata & 0x80u) ? -magnitude : magnitude;
        }
        reference += (a * scale_a) * (b * scale_b);
      }
      float gpu = output[size_t(mm) * size_t(opt.n) + size_t(nn)];
      if (!std::isfinite(reference) || !std::isfinite(gpu)) {
        ++nonfinite;
        continue;
      }
      float difference = fabsf(reference - gpu);
      max_abs_reference = std::max(max_abs_reference, fabsf(reference));
      max_abs_error = std::max(max_abs_error, difference);
      squared_reference += double(reference) * double(reference);
      squared_error += double(difference) * double(difference);
    }
  }
  double normalized_max =
      double(max_abs_error) / std::max(1.0, double(max_abs_reference));
  double relative_l2 =
      std::sqrt(squared_error / std::max(1.0, squared_reference));
  std::cout << std::scientific
            << "Correctness: max_abs=" << max_abs_error
            << " normalized_max=" << normalized_max
            << " relative_l2=" << relative_l2
            << " nonfinite=" << nonfinite << "\n"
            << std::defaultfloat;
  if (nonfinite || normalized_max > opt.max_normalized_error) {
    std::cerr << "CORRECTNESS FAILED: threshold="
              << opt.max_normalized_error << "\n";
    return 3;
  }
  std::cout << "Correctness threshold: PASS\n";
  return 0;
}

template <class Function>
static std::vector<double> time_iterations(
    WeightOptions const& opt, Function&& function,
    unsigned char* flush_buffer, size_t flush_bytes,
    cudaStream_t execution_stream = nullptr) {
  auto flush = [&]() {
    if (flush_buffer) {
      flush_cache_kernel<<<1024, 256>>>(flush_buffer, flush_bytes);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  };
  for (int iteration = 0; iteration < opt.warmup; ++iteration) {
    flush();
    function();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<double> samples;
  samples.reserve(size_t(opt.iters));
  for (int iteration = 0; iteration < opt.iters; ++iteration) {
    flush();
    CUDA_CHECK(cudaEventRecord(start, execution_stream));
    function();
    CUDA_CHECK(cudaEventRecord(stop, execution_stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    samples.push_back(double(elapsed));
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return samples;
}

__global__ void weight_add_output_float4(
    float4* output, float4 const* addend, size_t vector_count) {
  size_t index = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
  if (index >= vector_count) return;
  float4 destination = output[index];
  float4 source = addend[index];
  destination.x += source.x;
  destination.y += source.y;
  destination.z += source.z;
  destination.w += source.w;
  output[index] = destination;
}

__global__ void weight_add_n_halves_float4(
    float4* output,
    float4 const* expanded_output,
    int n_vectors,
    size_t output_vector_count) {
  size_t index =
      size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
  if (index >= output_vector_count) return;
  size_t row = index / size_t(n_vectors);
  size_t column_vector = index - row * size_t(n_vectors);
  size_t expanded_row_base = row * size_t(2 * n_vectors);
  float4 main_value =
      expanded_output[expanded_row_base + column_vector];
  float4 remainder_value =
      expanded_output[
          expanded_row_base + size_t(n_vectors) + column_vector];
  main_value.x += remainder_value.x;
  main_value.y += remainder_value.y;
  main_value.z += remainder_value.z;
  main_value.w += remainder_value.w;
  output[index] = main_value;
}

static int run_dense_mode(
    WeightOptions const& opt, WeightHostData const& host) {
  using namespace cute;
  bool concat = opt.mode == "concat"
#if defined(RAZER_STREAM_K)
      || opt.mode == "concat-splitk" ||
      opt.mode == "concat-splitk-atomic" ||
      opt.mode == "concat-splitk-graph"
#endif
      ;
  bool concat_n =
      opt.mode == "concat-n" || opt.mode == "concat-n-graph"
#if defined(RAZER_STREAM_K)
      || opt.mode == "concat-n-splitk" ||
      opt.mode == "concat-n-splitk-atomic" ||
      opt.mode == "concat-n-splitk-graph"
#endif
      ;
  bool concat_n_graph =
      opt.mode == "concat-n-graph"
#if defined(RAZER_STREAM_K)
      || opt.mode == "concat-n-splitk-graph"
#endif
      ;
  bool graph_two_pass = opt.mode == "two-pass-overlap-graph";
  bool overlap_two_pass = opt.mode == "two-pass-overlap" || graph_two_pass;
  int expansion = concat ? 2 : 1;
  int execution_k = expansion * opt.k;
  int execution_n = concat_n ? 2 * opt.n : opt.n;
  int k_blocks = opt.k / K_BLOCK;

  cutlass::device_memory::allocation<uint8_t> d_a_input(host.a_packed.size());
  cutlass::device_memory::allocation<uint8_t> d_b_input(host.b_packed.size());
  cutlass::device_memory::allocation<uint8_t> d_sfb_raw(host.sfb_raw.size());
  cutlass::device_memory::allocation<uint8_t> d_a_execution(
      concat ? 2 * host.a_packed.size() : size_t(1));
  cutlass::device_memory::allocation<uint8_t> d_b_main(
      size_t(execution_n) * size_t(execution_k) / 2);
  cutlass::device_memory::allocation<uint8_t> d_b_remainder(
      (concat || concat_n) ? size_t(1) : host.b_packed.size());
  cutlass::device_memory::allocation<float> d_output(
      size_t(opt.m) * size_t(opt.n));
  cutlass::device_memory::allocation<float> d_concat_n_output(
      concat_n ? size_t(opt.m) * size_t(execution_n) : size_t(1));
  cutlass::device_memory::allocation<float> d_remainder_output(
      overlap_two_pass ? size_t(opt.m) * size_t(opt.n) : size_t(1));
  CUDA_CHECK(cudaMemcpy(
      d_a_input.get(), host.a_packed.data(), host.a_packed.size(),
      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      d_b_input.get(), host.b_packed.data(), host.b_packed.size(),
      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      d_sfb_raw.get(), host.sfb_raw.data(), host.sfb_raw.size(),
      cudaMemcpyHostToDevice));

  auto stride_a =
      cutlass::make_cute_packed_stride(StrideA{}, {opt.m, execution_k, 1});
  auto stride_b =
      cutlass::make_cute_packed_stride(
          StrideB{}, {execution_n, execution_k, 1});
  auto stride_c =
      cutlass::make_cute_packed_stride(
          StrideC{}, {opt.m, execution_n, 1});
  auto stride_d =
      cutlass::make_cute_packed_stride(
          StrideD{}, {opt.m, execution_n, 1});
  auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      make_shape(opt.m, execution_n, execution_k, 1));
  auto layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      make_shape(opt.m, execution_n, execution_k, 1));
  cutlass::HostTensor<
      WeightSFAType, cutlass::layout::PackedVectorLayout> block_sfa;
  cutlass::HostTensor<
      WeightSFBType, cutlass::layout::PackedVectorLayout> block_sfb;
  block_sfa.reset(cutlass::make_Coord(int(size(filter_zeros(layout_sfa)))));
  block_sfb.reset(cutlass::make_Coord(int(size(filter_zeros(layout_sfb)))));
  auto tensor_sfa = make_tensor(block_sfa.host_data(), layout_sfa);
  auto tensor_sfb = make_tensor(block_sfb.host_data(), layout_sfb);
  for (int coordinate = 0; coordinate < expansion; ++coordinate) {
    for (int mm = 0; mm < opt.m; ++mm) {
      for (int kk = 0; kk < opt.k; kk += K_BLOCK) {
        WeightSFAType value;
        uint8_t raw =
            host.sfa_mma[size_t(mm) * size_t(k_blocks) + size_t(kk / K_BLOCK)];
        std::memcpy(&value, &raw, 1);
        tensor_sfa(mm, coordinate * opt.k + kk, 0) = value;
      }
    }
  }
  int b_coordinate_count = (concat || concat_n) ? 2 : 1;
  for (int coordinate = 0;
       coordinate < b_coordinate_count;
       ++coordinate) {
    for (int nn = 0; nn < opt.n; ++nn) {
      for (int kk = 0; kk < opt.k; kk += K_BLOCK) {
        WeightSFBType value;
        uint8_t raw =
            host.sfb_mma[size_t(nn) * size_t(k_blocks) + size_t(kk / K_BLOCK)];
        std::memcpy(&value, &raw, 1);
        int execution_column = concat_n ? coordinate * opt.n + nn : nn;
        int execution_reduction = concat ? coordinate * opt.k + kk : kk;
        tensor_sfb(
            execution_column, execution_reduction, 0) = value;
      }
    }
  }
  block_sfa.sync_device();
  block_sfb.sync_device();

  auto launch_a_duplicate = [&]() {
    int count = opt.m * k_blocks;
    weight_duplicate_A_concat<<<(count + 127) / 128, 128>>>(
        as_a(d_a_execution.get()), as_a(d_a_input.get()), opt.m, opt.k);
    CUDA_CHECK(cudaGetLastError());
  };
  auto launch_b_transform = [&]() {
    int count = opt.n * k_blocks;
    if (concat) {
      weight_build_B_concat<<<(count + 127) / 128, 128>>>(
          as_b(d_b_main.get()), as_b(d_b_input.get()), d_sfb_raw.get(),
          opt.n, opt.k, opt.second_magnitude);
    } else if (concat_n) {
      weight_build_B_concat_n<<<(count + 127) / 128, 128>>>(
          as_b(d_b_main.get()), as_b(d_b_input.get()), d_sfb_raw.get(),
          opt.n, opt.k, opt.second_magnitude);
    } else {
      weight_build_B_two_packed<<<(count + 127) / 128, 128>>>(
          as_b(d_b_main.get()), as_b(d_b_remainder.get()),
          as_b(d_b_input.get()), d_sfb_raw.get(),
          opt.n, opt.k, opt.second_magnitude);
    }
    CUDA_CHECK(cudaGetLastError());
  };

  WeightAPacked* a_execution =
      concat ? as_a(d_a_execution.get()) : as_a(d_a_input.get());
  Gemm gemm_main;
  typename Gemm::Arguments args_main{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {opt.m, execution_n, execution_k, 1},
    {
      a_execution, stride_a,
      as_b(d_b_main.get()), stride_b,
      block_sfa.device_data(), layout_sfa,
      block_sfb.device_data(), layout_sfb
    },
    {
      {1.0f, 0.0f},
      concat_n ? d_concat_n_output.get() : d_output.get(), stride_c,
      concat_n ? d_concat_n_output.get() : d_output.get(), stride_d
    }
  };
  weight_configure_scheduler(
      args_main, opt.scheduler_swizzle, opt.scheduler_raster);
#if defined(RAZER_STREAM_K)
  using StreamKParams =
      cutlass::gemm::kernel::detail::
          PersistentTileSchedulerSm90StreamKParams;
  if (opt.mode == "concat-splitk" ||
      opt.mode == "concat-splitk-atomic" ||
      opt.mode == "concat-splitk-graph" ||
      opt.mode == "concat-n-splitk" ||
      opt.mode == "concat-n-splitk-atomic" ||
      opt.mode == "concat-n-splitk-graph") {
    args_main.scheduler.splits = RAZER_SPLIT_K;
    args_main.scheduler.decomposition_mode =
        StreamKParams::DecompositionMode::SplitK;
    args_main.scheduler.reduction_mode =
        (opt.mode == "concat-splitk" ||
         opt.mode == "concat-splitk-graph" ||
         opt.mode == "concat-n-splitk" ||
         opt.mode == "concat-n-splitk-graph")
            ? StreamKParams::ReductionMode::Deterministic
            : StreamKParams::ReductionMode::Nondeterministic;
  } else {
    args_main.scheduler.splits = 1;
    args_main.scheduler.decomposition_mode =
        StreamKParams::DecompositionMode::DataParallel;
    args_main.scheduler.reduction_mode =
        StreamKParams::ReductionMode::Deterministic;
  }
#endif
  size_t workspace_bytes = Gemm::get_workspace_size(args_main);
  cutlass::device_memory::allocation<uint8_t> workspace_main(workspace_bytes);
  CUTLASS_CHECK(gemm_main.can_implement(args_main));
  CUTLASS_CHECK(gemm_main.initialize(args_main, workspace_main.get()));

  launch_b_transform();
  if (concat && !opt.online_a_duplicate) launch_a_duplicate();
  CUDA_CHECK(cudaDeviceSynchronize());

  if (concat_n) {
    size_t output_count = size_t(opt.m) * size_t(opt.n);
    size_t output_vector_count = output_count / 4;
    int n_vectors = opt.n / 4;
    int add_threads = 256;
    int add_blocks =
        int((output_vector_count + size_t(add_threads) - 1) /
            size_t(add_threads));
    cudaStream_t concat_n_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &concat_n_stream, cudaStreamNonBlocking));

    auto launch_add = [&]() {
      weight_add_n_halves_float4<<<
          add_blocks, add_threads, 0, concat_n_stream>>>(
          reinterpret_cast<float4*>(d_output.get()),
          reinterpret_cast<float4 const*>(d_concat_n_output.get()),
          n_vectors,
          output_vector_count);
      CUDA_CHECK(cudaGetLastError());
    };

    cudaGraph_t concat_n_graph_handle = nullptr;
    cudaGraphExec_t concat_n_graph_exec = nullptr;
    if (concat_n_graph) {
      CUDA_CHECK(cudaStreamBeginCapture(
          concat_n_stream, cudaStreamCaptureModeGlobal));
#if defined(RAZER_STREAM_K)
      if (opt.mode == "concat-n-splitk-graph") {
        CUTLASS_CHECK(GemmKernel::initialize_workspace(
            args_main, workspace_main.get(), concat_n_stream));
      }
#endif
      CUTLASS_CHECK(gemm_main.run(concat_n_stream));
      launch_add();
      CUDA_CHECK(cudaStreamEndCapture(
          concat_n_stream, &concat_n_graph_handle));
      CUDA_CHECK(cudaGraphInstantiate(
          &concat_n_graph_exec, concat_n_graph_handle,
          nullptr, nullptr, 0));
    }
    auto execute = [&]() {
#if defined(RAZER_STREAM_K)
      if (opt.mode == "concat-n-splitk" ||
          opt.mode == "concat-n-splitk-atomic") {
        CUTLASS_CHECK(GemmKernel::initialize_workspace(
            args_main, workspace_main.get(), concat_n_stream));
      }
#endif
      if (concat_n_graph) {
        CUDA_CHECK(cudaGraphLaunch(
            concat_n_graph_exec, concat_n_stream));
      } else {
        CUTLASS_CHECK(gemm_main.run(concat_n_stream));
        launch_add();
      }
    };

    if (opt.check) {
      execute();
      CUDA_CHECK(cudaStreamSynchronize(concat_n_stream));
      std::vector<float> output(output_count);
      CUDA_CHECK(cudaMemcpy(
          output.data(), d_output.get(), output.size() * sizeof(float),
          cudaMemcpyDeviceToHost));
      int status = validate_output(opt, host, output);
      if (status) return status;
    }

    size_t flush_bytes = size_t(opt.flush_mb) * 1024ull * 1024ull;
    unsigned char* flush_buffer = nullptr;
    if (flush_bytes) CUDA_CHECK(cudaMalloc(&flush_buffer, flush_bytes));
    auto flush = [&]() {
      if (flush_buffer) {
        flush_cache_kernel<<<1024, 256>>>(flush_buffer, flush_bytes);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    };
    for (int iteration = 0; iteration < opt.warmup; ++iteration) {
      flush();
      execute();
      CUDA_CHECK(cudaStreamSynchronize(concat_n_stream));
    }
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> samples;
    samples.reserve(size_t(opt.iters));
    for (int iteration = 0; iteration < opt.iters; ++iteration) {
      flush();
      CUDA_CHECK(cudaEventRecord(start, concat_n_stream));
      execute();
      CUDA_CHECK(cudaEventRecord(stop, concat_n_stream));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float elapsed = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
      samples.push_back(double(elapsed));
    }
    if (opt.check) {
      CUDA_CHECK(cudaStreamSynchronize(concat_n_stream));
      std::vector<float> repeated_output(output_count);
      CUDA_CHECK(cudaMemcpy(
          repeated_output.data(), d_output.get(),
          repeated_output.size() * sizeof(float),
          cudaMemcpyDeviceToHost));
      int status = validate_output(opt, host, repeated_output);
      if (status) return status;
      std::cout << "Repeated-launch correctness: PASS\n";
    }
    SampleSummary summary = summarize(samples);
    double logical_ops =
        2.0 * double(opt.m) * double(opt.n) * double(opt.k);
    std::cout << "METHOD=weight-razer-" << opt.mode << "\n"
              << "M,N,K = " << opt.m << "," << opt.n << "," << opt.k
              << "\n"
              << std::fixed << std::setprecision(6)
              << "Best: " << summary.best
              << " ms, Median: " << summary.median
              << " ms, P90: " << summary.p90
              << " ms, Avg: " << summary.average
              << " ms over " << opt.iters
              << " iters (warmup " << opt.warmup << ")\n"
              << "Timing includes: 1x GEMM(N'=2N) + FP32 half add"
              << (concat_n_graph ? " in one reusable CUDA graph" : "")
#if defined(RAZER_STREAM_K)
              << (opt.mode == "concat-n-splitk"
                  ? " with deterministic Split-K="
                  : (opt.mode == "concat-n-splitk-atomic"
                      ? " with atomic Split-K="
                      : (opt.mode == "concat-n-splitk-graph"
                          ? " with deterministic Split-K="
                          : "")))
              << ((opt.mode == "concat-n-splitk" ||
                   opt.mode == "concat-n-splitk-atomic" ||
                   opt.mode == "concat-n-splitk-graph")
                  ? std::to_string(RAZER_SPLIT_K) : std::string())
#endif
              << "; static B preprocessing excluded; flush_mb="
              << opt.flush_mb << "\n"
              << std::setprecision(2)
              << "Effective logical throughput: avg "
              << logical_ops / (summary.average * 1.0e9)
              << " TFLOPs (2*M*N*K logical ops; actual tensor work is 2x)\n";
    print_samples(samples);

    if (opt.breakdown) {
      cudaEvent_t e0, e1, e2;
      CUDA_CHECK(cudaEventCreate(&e0));
      CUDA_CHECK(cudaEventCreate(&e1));
      CUDA_CHECK(cudaEventCreate(&e2));
      CUDA_CHECK(cudaEventRecord(e0, concat_n_stream));
#if defined(RAZER_STREAM_K)
      if (opt.mode == "concat-n-splitk" ||
          opt.mode == "concat-n-splitk-atomic" ||
          opt.mode == "concat-n-splitk-graph") {
        CUTLASS_CHECK(GemmKernel::initialize_workspace(
            args_main, workspace_main.get(), concat_n_stream));
      }
#endif
      CUTLASS_CHECK(gemm_main.run(concat_n_stream));
      CUDA_CHECK(cudaEventRecord(e1, concat_n_stream));
      launch_add();
      CUDA_CHECK(cudaEventRecord(e2, concat_n_stream));
      CUDA_CHECK(cudaEventSynchronize(e2));
      float gemm_ms = 0.0f;
      float add_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&gemm_ms, e0, e1));
      CUDA_CHECK(cudaEventElapsedTime(&add_ms, e1, e2));
      std::cout << std::fixed << std::setprecision(6)
                << "BREAKDOWN_MS gemm_n2=" << gemm_ms
                << " add=" << add_ms << "\n";
      CUDA_CHECK(cudaEventDestroy(e0));
      CUDA_CHECK(cudaEventDestroy(e1));
      CUDA_CHECK(cudaEventDestroy(e2));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (flush_buffer) CUDA_CHECK(cudaFree(flush_buffer));
    if (concat_n_graph) {
      CUDA_CHECK(cudaGraphExecDestroy(concat_n_graph_exec));
      CUDA_CHECK(cudaGraphDestroy(concat_n_graph_handle));
    }
    CUDA_CHECK(cudaStreamDestroy(concat_n_stream));
    return 0;
  }

  if (concat) {
    bool concat_splitk_graph =
#if defined(RAZER_STREAM_K)
        opt.mode == "concat-splitk-graph";
#else
        false;
#endif
    cudaStream_t concat_stream = nullptr;
    cudaGraph_t concat_graph = nullptr;
    cudaGraphExec_t concat_graph_exec = nullptr;
    if (concat_splitk_graph) {
      CUDA_CHECK(cudaStreamCreateWithFlags(
          &concat_stream, cudaStreamNonBlocking));
      CUDA_CHECK(cudaStreamBeginCapture(
          concat_stream, cudaStreamCaptureModeGlobal));
      CUTLASS_CHECK(GemmKernel::initialize_workspace(
          args_main, workspace_main.get(), concat_stream));
      CUTLASS_CHECK(gemm_main.run(concat_stream));
      CUDA_CHECK(cudaStreamEndCapture(concat_stream, &concat_graph));
      CUDA_CHECK(cudaGraphInstantiate(
          &concat_graph_exec, concat_graph, nullptr, nullptr, 0));
    }
    auto execute = [&]() {
      if (opt.online_a_duplicate) launch_a_duplicate();
#if defined(RAZER_STREAM_K)
      if (opt.mode == "concat-splitk-graph") {
        CUDA_CHECK(cudaGraphLaunch(
            concat_graph_exec, concat_stream));
        return;
      }
      if (opt.mode == "concat-splitk" ||
          opt.mode == "concat-splitk-atomic") {
        CUTLASS_CHECK(GemmKernel::initialize_workspace(
            args_main, workspace_main.get(), concat_stream));
      }
#endif
      CUTLASS_CHECK(gemm_main.run(concat_stream));
    };
    if (opt.check) {
      execute();
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<float> output(size_t(opt.m) * size_t(opt.n));
      CUDA_CHECK(cudaMemcpy(
          output.data(), d_output.get(), output.size() * sizeof(float),
          cudaMemcpyDeviceToHost));
      int status = validate_output(opt, host, output);
      if (status) return status;
    }
    size_t flush_bytes = size_t(opt.flush_mb) * 1024ull * 1024ull;
    unsigned char* flush_buffer = nullptr;
    if (flush_bytes) CUDA_CHECK(cudaMalloc(&flush_buffer, flush_bytes));
    std::vector<double> samples =
        time_iterations(
            opt, execute, flush_buffer, flush_bytes, concat_stream);
    if (opt.check) {
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<float> repeated_output(
          size_t(opt.m) * size_t(opt.n));
      CUDA_CHECK(cudaMemcpy(
          repeated_output.data(), d_output.get(),
          repeated_output.size() * sizeof(float),
          cudaMemcpyDeviceToHost));
      int status = validate_output(opt, host, repeated_output);
      if (status) return status;
      std::cout << "Repeated-launch correctness: PASS\n";
    }
    SampleSummary summary = summarize(samples);
    double logical_ops = 2.0 * double(opt.m) * double(opt.n) * double(opt.k);
    std::cout << "METHOD=weight-razer-" << opt.mode << "\n"
              << "M,N,K = " << opt.m << "," << opt.n << "," << opt.k << "\n"
              << std::fixed << std::setprecision(6)
              << "Best: " << summary.best << " ms, Median: " << summary.median
              << " ms, P90: " << summary.p90 << " ms, Avg: " << summary.average
              << " ms over " << opt.iters << " iters (warmup " << opt.warmup << ")\n"
              << "Timing includes: "
              << (opt.online_a_duplicate ? "online A duplication + " : "")
              << "1x GEMM(K'=2K)"
              << (opt.mode == "concat-splitk"
                  ? " with deterministic Split-K="
                  : (opt.mode == "concat-splitk-atomic"
                      ? " with atomic Split-K="
                      : (opt.mode == "concat-splitk-graph"
                          ? " with deterministic Split-K="
                          : "")))
#if defined(RAZER_STREAM_K)
              << ((opt.mode == "concat-splitk" ||
                   opt.mode == "concat-splitk-atomic" ||
                   opt.mode == "concat-splitk-graph")
                  ? std::to_string(RAZER_SPLIT_K) : std::string())
#endif
              << (concat_splitk_graph
                  ? " in one reusable CUDA graph" : "")
              << "; static B preprocessing excluded; flush_mb="
              << opt.flush_mb << "\n"
              << std::setprecision(2)
              << "Effective logical throughput: avg "
              << logical_ops / (summary.average * 1.0e9)
              << " TFLOPs (2*M*N*K logical ops; actual tensor work is 2x)\n";
    print_samples(samples);
    if (opt.breakdown) {
      cudaEvent_t e0, e1, e2;
      CUDA_CHECK(cudaEventCreate(&e0));
      CUDA_CHECK(cudaEventCreate(&e1));
      CUDA_CHECK(cudaEventCreate(&e2));
      CUDA_CHECK(cudaEventRecord(e0));
      if (opt.online_a_duplicate) launch_a_duplicate();
      CUDA_CHECK(cudaEventRecord(e1));
#if defined(RAZER_STREAM_K)
      if (opt.mode == "concat-splitk" ||
          opt.mode == "concat-splitk-atomic") {
        CUTLASS_CHECK(GemmKernel::initialize_workspace(
            args_main, workspace_main.get()));
      }
#endif
      CUTLASS_CHECK(gemm_main.run());
      CUDA_CHECK(cudaEventRecord(e2));
      CUDA_CHECK(cudaEventSynchronize(e2));
      float duplicate_ms = 0.0f, gemm_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&duplicate_ms, e0, e1));
      CUDA_CHECK(cudaEventElapsedTime(&gemm_ms, e1, e2));
      std::cout << std::fixed << std::setprecision(6)
                << "BREAKDOWN_MS duplicate=" << duplicate_ms
                << " gemm=" << gemm_ms << "\n";
      CUDA_CHECK(cudaEventDestroy(e0));
      CUDA_CHECK(cudaEventDestroy(e1));
      CUDA_CHECK(cudaEventDestroy(e2));
    }
    if (flush_buffer) CUDA_CHECK(cudaFree(flush_buffer));
    if (concat_splitk_graph) {
      CUDA_CHECK(cudaGraphExecDestroy(concat_graph_exec));
      CUDA_CHECK(cudaGraphDestroy(concat_graph));
      CUDA_CHECK(cudaStreamDestroy(concat_stream));
    }
    return 0;
  }

  auto stride_a_original =
      cutlass::make_cute_packed_stride(StrideA{}, {opt.m, opt.k, 1});
  auto stride_b_original =
      cutlass::make_cute_packed_stride(StrideB{}, {opt.n, opt.k, 1});
  float* remainder_output =
      overlap_two_pass ? d_remainder_output.get() : d_output.get();
  float remainder_beta = overlap_two_pass ? 0.0f : 1.0f;
  Gemm gemm_remainder;
  typename Gemm::Arguments args_remainder{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {opt.m, opt.n, opt.k, 1},
    {
      as_a(d_a_input.get()), stride_a_original,
      as_b(d_b_remainder.get()), stride_b_original,
      block_sfa.device_data(), layout_sfa,
      block_sfb.device_data(), layout_sfb
    },
    {
      {1.0f, remainder_beta},
      remainder_output, stride_d,
      remainder_output, stride_d
    }
  };
  weight_configure_scheduler(
      args_remainder, opt.scheduler_swizzle, opt.scheduler_raster);
  size_t remainder_workspace_bytes = Gemm::get_workspace_size(args_remainder);
  cutlass::device_memory::allocation<uint8_t> workspace_remainder(
      remainder_workspace_bytes);
  CUTLASS_CHECK(gemm_remainder.can_implement(args_remainder));
  CUTLASS_CHECK(gemm_remainder.initialize(
      args_remainder, workspace_remainder.get()));

  if (overlap_two_pass) {
    size_t output_count = size_t(opt.m) * size_t(opt.n);
    size_t output_vector_count = output_count / 4;
    int add_threads = 256;
    int add_blocks =
        int((output_vector_count + size_t(add_threads) - 1) /
            size_t(add_threads));
    cudaStream_t main_stream, remainder_stream;
    cudaEvent_t remainder_done;
    CUDA_CHECK(cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &remainder_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(
        &remainder_done, cudaEventDisableTiming));

    auto execute_overlap = [&](cudaEvent_t gate) {
      CUDA_CHECK(cudaStreamWaitEvent(remainder_stream, gate));
      CUTLASS_CHECK(gemm_main.run(main_stream));
      CUTLASS_CHECK(gemm_remainder.run(remainder_stream));
      CUDA_CHECK(cudaEventRecord(remainder_done, remainder_stream));
      CUDA_CHECK(cudaStreamWaitEvent(main_stream, remainder_done));
      weight_add_output_float4<<<add_blocks, add_threads, 0, main_stream>>>(
          reinterpret_cast<float4*>(d_output.get()),
          reinterpret_cast<float4 const*>(d_remainder_output.get()),
          output_vector_count);
      CUDA_CHECK(cudaGetLastError());
    };

    cudaGraph_t overlap_graph = nullptr;
    cudaGraphExec_t overlap_graph_exec = nullptr;
    cudaEvent_t capture_gate = nullptr;
    if (graph_two_pass) {
      CUDA_CHECK(cudaEventCreateWithFlags(
          &capture_gate, cudaEventDisableTiming));
      CUDA_CHECK(cudaStreamBeginCapture(
          main_stream, cudaStreamCaptureModeGlobal));
      CUDA_CHECK(cudaEventRecord(capture_gate, main_stream));
      execute_overlap(capture_gate);
      CUDA_CHECK(cudaStreamEndCapture(main_stream, &overlap_graph));
      CUDA_CHECK(cudaGraphInstantiate(
          &overlap_graph_exec, overlap_graph, nullptr, nullptr, 0));
    }
    auto execute_selected = [&](cudaEvent_t gate) {
      if (graph_two_pass) {
        CUDA_CHECK(cudaGraphLaunch(overlap_graph_exec, main_stream));
      } else {
        execute_overlap(gate);
      }
    };

    int correctness_status = 0;
    if (opt.check) {
      cudaEvent_t gate;
      CUDA_CHECK(cudaEventCreateWithFlags(&gate, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventRecord(gate, main_stream));
      execute_selected(gate);
      CUDA_CHECK(cudaStreamSynchronize(main_stream));
      CUDA_CHECK(cudaEventDestroy(gate));
      std::vector<float> output(output_count);
      CUDA_CHECK(cudaMemcpy(
          output.data(), d_output.get(), output.size() * sizeof(float),
          cudaMemcpyDeviceToHost));
      correctness_status = validate_output(opt, host, output);
    }

    size_t flush_bytes = size_t(opt.flush_mb) * 1024ull * 1024ull;
    unsigned char* flush_buffer = nullptr;
    if (!correctness_status && flush_bytes) {
      CUDA_CHECK(cudaMalloc(&flush_buffer, flush_bytes));
    }
    auto flush = [&]() {
      if (flush_buffer) {
        flush_cache_kernel<<<1024, 256>>>(flush_buffer, flush_bytes);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    };
    std::vector<double> samples;
    if (!correctness_status) {
      cudaEvent_t gate, start, stop;
      CUDA_CHECK(cudaEventCreateWithFlags(&gate, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));
      for (int iteration = 0; iteration < opt.warmup; ++iteration) {
        flush();
        CUDA_CHECK(cudaEventRecord(gate, main_stream));
        execute_selected(gate);
        CUDA_CHECK(cudaStreamSynchronize(main_stream));
      }
      samples.reserve(size_t(opt.iters));
      for (int iteration = 0; iteration < opt.iters; ++iteration) {
        flush();
        CUDA_CHECK(cudaEventRecord(start, main_stream));
        execute_selected(start);
        CUDA_CHECK(cudaEventRecord(stop, main_stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        samples.push_back(double(elapsed));
      }
      CUDA_CHECK(cudaEventDestroy(gate));
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));
    }

    if (!correctness_status) {
      SampleSummary summary = summarize(samples);
      double logical_ops =
          2.0 * double(opt.m) * double(opt.n) * double(opt.k);
      std::cout << "METHOD=weight-razer-" << opt.mode << "\n"
                << "M,N,K = " << opt.m << "," << opt.n << "," << opt.k << "\n"
                << std::fixed << std::setprecision(6)
                << "Best: " << summary.best << " ms, Median: " << summary.median
                << " ms, P90: " << summary.p90 << " ms, Avg: "
                << summary.average << " ms over " << opt.iters
                << " iters (warmup " << opt.warmup << ")\n"
                << "Timing includes: concurrent 2x GEMM + FP32 output add"
                << (graph_two_pass ? " in one reusable CUDA graph" : "")
                << "; "
                   "static B preprocessing excluded; flush_mb="
                << opt.flush_mb << "\n"
                << std::setprecision(2)
                << "Effective logical throughput: avg "
                << logical_ops / (summary.average * 1.0e9)
                << " TFLOPs (2*M*N*K logical ops; actual tensor work is 2x)\n";
      print_samples(samples);
    }

    if (!correctness_status && opt.breakdown) {
      cudaEvent_t start, main_done, remainder_start, remainder_end;
      cudaEvent_t add_start, end;
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&main_done));
      CUDA_CHECK(cudaEventCreate(&remainder_start));
      CUDA_CHECK(cudaEventCreate(&remainder_end));
      CUDA_CHECK(cudaEventCreate(&add_start));
      CUDA_CHECK(cudaEventCreate(&end));
      CUDA_CHECK(cudaEventRecord(start, main_stream));
      CUDA_CHECK(cudaStreamWaitEvent(remainder_stream, start));
      CUTLASS_CHECK(gemm_main.run(main_stream));
      CUDA_CHECK(cudaEventRecord(main_done, main_stream));
      CUDA_CHECK(cudaEventRecord(remainder_start, remainder_stream));
      CUTLASS_CHECK(gemm_remainder.run(remainder_stream));
      CUDA_CHECK(cudaEventRecord(remainder_end, remainder_stream));
      CUDA_CHECK(cudaStreamWaitEvent(main_stream, remainder_end));
      CUDA_CHECK(cudaEventRecord(add_start, main_stream));
      weight_add_output_float4<<<add_blocks, add_threads, 0, main_stream>>>(
          reinterpret_cast<float4*>(d_output.get()),
          reinterpret_cast<float4 const*>(d_remainder_output.get()),
          output_vector_count);
      CUDA_CHECK(cudaEventRecord(end, main_stream));
      CUDA_CHECK(cudaEventSynchronize(end));
      float total_ms = 0.0f, main_ms = 0.0f;
      float remainder_ms = 0.0f, add_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, end));
      CUDA_CHECK(cudaEventElapsedTime(&main_ms, start, main_done));
      CUDA_CHECK(cudaEventElapsedTime(
          &remainder_ms, remainder_start, remainder_end));
      CUDA_CHECK(cudaEventElapsedTime(&add_ms, add_start, end));
      std::cout << std::fixed << std::setprecision(6)
                << (graph_two_pass
                    ? "BREAKDOWN_UNCAPTURED_DAG_MS "
                    : "BREAKDOWN_MS ")
                << "concurrent_total=" << total_ms
                << " main=" << main_ms
                << " remainder=" << remainder_ms
                << " add=" << add_ms << "\n";
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(main_done));
      CUDA_CHECK(cudaEventDestroy(remainder_start));
      CUDA_CHECK(cudaEventDestroy(remainder_end));
      CUDA_CHECK(cudaEventDestroy(add_start));
      CUDA_CHECK(cudaEventDestroy(end));
    }
    if (flush_buffer) CUDA_CHECK(cudaFree(flush_buffer));
    if (graph_two_pass) {
      CUDA_CHECK(cudaGraphExecDestroy(overlap_graph_exec));
      CUDA_CHECK(cudaGraphDestroy(overlap_graph));
      CUDA_CHECK(cudaEventDestroy(capture_gate));
    }
    CUDA_CHECK(cudaEventDestroy(remainder_done));
    CUDA_CHECK(cudaStreamDestroy(main_stream));
    CUDA_CHECK(cudaStreamDestroy(remainder_stream));
    return correctness_status;
  }

  auto execute = [&]() {
    CUTLASS_CHECK(gemm_main.run());
    CUTLASS_CHECK(gemm_remainder.run());
  };
  if (opt.check) {
    execute();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> output(size_t(opt.m) * size_t(opt.n));
    CUDA_CHECK(cudaMemcpy(
        output.data(), d_output.get(), output.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    int status = validate_output(opt, host, output);
    if (status) return status;
  }
  size_t flush_bytes = size_t(opt.flush_mb) * 1024ull * 1024ull;
  unsigned char* flush_buffer = nullptr;
  if (flush_bytes) CUDA_CHECK(cudaMalloc(&flush_buffer, flush_bytes));
  std::vector<double> samples =
      time_iterations(opt, execute, flush_buffer, flush_bytes);
  SampleSummary summary = summarize(samples);
  double logical_ops = 2.0 * double(opt.m) * double(opt.n) * double(opt.k);
  std::cout << "METHOD=weight-razer-two-pass\n"
            << "M,N,K = " << opt.m << "," << opt.n << "," << opt.k << "\n"
            << std::fixed << std::setprecision(6)
            << "Best: " << summary.best << " ms, Median: " << summary.median
            << " ms, P90: " << summary.p90 << " ms, Avg: " << summary.average
            << " ms over " << opt.iters << " iters (warmup " << opt.warmup << ")\n"
            << "Timing includes: 2x GEMM; static B preprocessing excluded; flush_mb="
            << opt.flush_mb << "\n"
            << std::setprecision(2)
            << "Effective logical throughput: avg "
            << logical_ops / (summary.average * 1.0e9)
            << " TFLOPs (2*M*N*K logical ops; actual tensor work is 2x)\n";
  print_samples(samples);
  if (opt.breakdown) {
    cudaEvent_t e0, e1, e2;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventRecord(e0));
    CUTLASS_CHECK(gemm_main.run());
    CUDA_CHECK(cudaEventRecord(e1));
    CUTLASS_CHECK(gemm_remainder.run());
    CUDA_CHECK(cudaEventRecord(e2));
    CUDA_CHECK(cudaEventSynchronize(e2));
    float main_ms = 0.0f, remainder_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&main_ms, e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&remainder_ms, e1, e2));
    std::cout << std::fixed << std::setprecision(6)
              << "BREAKDOWN_MS main=" << main_ms
              << " remainder=" << remainder_ms << "\n";
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(e2));
  }
  if (flush_buffer) CUDA_CHECK(cudaFree(flush_buffer));
  return 0;
}

int main(int argc, char const** argv) {
  WeightOptions opt;
  opt.parse(argc, argv);
  if (opt.help) {
    opt.usage();
    return 0;
  }
  if (!opt.valid()) {
    std::cerr << "Invalid or missing argument.\n";
    opt.usage();
    return 1;
  }
  if (opt.k % 64 != 0 || opt.m % 16 != 0 || opt.n % 8 != 0) {
    std::cerr
        << "Require M multiple of 16, N multiple of 8, "
           "and K multiple of 64.\n";
    return 1;
  }

  WeightHostData host = make_weight_data(opt);
  double actual_rate =
      double(host.special_count) / (double(opt.n) * double(opt.k));
  std::cout << "KERNEL_SHAPE tile=" << RAZER_TILE_M << "x"
            << RAZER_TILE_N << "x" << RAZER_TILE_K
            << " cluster=" << RAZER_CLUSTER_M << "x"
            << RAZER_CLUSTER_N
            << " output_alignment=" << RAZER_OUTPUT_ALIGNMENT << "\n"
            << std::setprecision(9)
            << "CONFIG mode=" << opt.mode
            << " m=" << opt.m << " n=" << opt.n << " k=" << opt.k
            << " second_magnitude=" << opt.second_magnitude
            << " requested_weight_special_rate=" << opt.weight_special_rate
            << " actual_weight_special_rate=" << actual_rate
            << " seed=" << opt.seed << "\n";
#if defined(RAZER_MAX_SWIZZLE) || defined(RAZER_RASTER_ALONG_M) || \
    defined(RAZER_RASTER_ALONG_N)
  std::cout << "SCHEDULER_OVERRIDE";
#if defined(RAZER_MAX_SWIZZLE)
  std::cout << " max_swizzle=" << RAZER_MAX_SWIZZLE;
#endif
#if defined(RAZER_RASTER_ALONG_M)
  std::cout << " raster=along-m";
#elif defined(RAZER_RASTER_ALONG_N)
  std::cout << " raster=along-n";
#endif
  std::cout << "\n";
#endif
  if (opt.scheduler_swizzle != -1) {
    std::cout << "SCHEDULER_OVERRIDE max_swizzle="
              << opt.scheduler_swizzle
              << " raster=" << opt.scheduler_raster << "\n";
  }
  return run_dense_mode(opt, host);
}
