// Activation-quantizer test.
//
// For every CTA tile height (16, 32, 64, 128) and both formats, the device output is compared
// byte for byte with a host reference: the same quantization arithmetic in host code, then
// packed with pack_a, the routine the GEMM drivers use to lay out their activations.
// The row-major reference is also checked for the expected RaZeR semantics on hand-built blocks.
#include "activation_quantization.h"
#include "razer_w4a4_host.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "cutlass/float8.h"

namespace {

bool check_cuda(cudaError_t status, char const* operation) {
  if (status == cudaSuccess) return true;
  std::cerr << operation << ": " << cudaGetErrorString(status) << "\n";
  return false;
}

std::uint8_t host_quantize_ue4m3(float value) {
  return std::uint8_t(cutlass::float_ue4m3_t(value).raw()) & 0x7Fu;
}

float host_decode_ue4m3(std::uint8_t raw) {
  return float(cutlass::float_ue4m3_t::bitcast(std::uint8_t(raw & 0x7Fu)));
}

std::uint8_t host_nearest_fp4(float value) {
  if (value <= -5.0f) return 0xFu;
  if (value <= -3.5f) return 0xEu;
  if (value <= -2.5f) return 0xDu;
  if (value <= -1.75f) return 0xCu;
  if (value <= -1.25f) return 0xBu;
  if (value <= -0.75f) return 0xAu;
  if (value <= -0.25f) return 0x9u;
  if (value <= 0.25f) return 0x8u;
  if (value <= 0.75f) return 0x1u;
  if (value <= 1.25f) return 0x2u;
  if (value <= 1.75f) return 0x3u;
  if (value <= 2.5f) return 0x4u;
  if (value <= 3.5f) return 0x5u;
  if (value <= 5.0f) return 0x6u;
  return 0x7u;
}

// Host copy of the device arithmetic, row-major outputs: codes [m][k] (one nibble per byte),
// scales [m][k/16].
void host_quantize(std::vector<float> const& input, int m, int k, bool razer,
                   std::vector<std::uint8_t>& codes, std::vector<std::uint8_t>& scales) {
  codes.assign(std::size_t(m) * k, 0);
  scales.assign(std::size_t(m) * k / 16, 0);
  float absolute_max = 0.0f;
  for (float v : input) absolute_max = std::max(absolute_max, std::fabs(v));
  for (int row = 0; row < m; ++row) {
    for (int kb = 0; kb < k / 16; ++kb) {
      float const* x = input.data() + std::size_t(row) * k + kb * 16;
      std::uint8_t* c = codes.data() + std::size_t(row) * k + kb * 16;
      std::uint8_t& s = scales[std::size_t(row) * (k / 16) + kb];
      if (absolute_max == 0.0f) {
        std::fill(c, c + 16, 0x8u);
        s = host_quantize_ue4m3(std::ldexp(1.0f, -9));
        continue;
      }
      float block_absmax = 0.0f;
      for (int e = 0; e < 16; ++e) block_absmax = std::max(block_absmax, std::fabs(x[e]));
      float global_scale = absolute_max / (6.0f * 448.0f);
      float scale_unquantized = (block_absmax / global_scale) / 6.0f;
      scale_unquantized = std::min(448.0f, std::max(std::ldexp(1.0f, -9), scale_unquantized));
      std::uint8_t scale_raw = host_quantize_ue4m3(scale_unquantized);
      float scale = global_scale * host_decode_ue4m3(scale_raw);
      // Each half of 8 values accumulates its own signed gain, as one device thread does.
      float half_gain[2] = {0.0f, 0.0f};
      bool positive[16] = {}, negative[16] = {};
      for (int e = 0; e < 16; ++e) {
        float normalized = x[e] / scale;
        c[e] = host_nearest_fp4(normalized);
        if (razer) {
          float magnitude = std::fabs(normalized);
          if (magnitude >= 4.5f && magnitude <= 5.5f) {
            float gain = 1.0f - 2.0f * std::fabs(magnitude - 5.0f);
            half_gain[e / 8] += std::copysign(gain, normalized);
            (normalized >= 0.0f ? positive : negative)[e] = true;
          }
        }
      }
      bool select_positive = half_gain[0] + half_gain[1] > 0.0f;
      if (razer)
        for (int e = 0; e < 16; ++e)
          if (select_positive ? positive[e] : negative[e]) c[e] = 0x0u;
      s = std::uint8_t(scale_raw | (razer && !select_positive ? 0x80u : 0x00u));
    }
  }
}

}  // namespace

int main() {
  // Row 0 holds three hand-built blocks (a positive special, a negative special, no special)
  // and one zero block; the other rows are random values that often fall near +-5.
  constexpr int kM = 37, kK = 256;
  std::vector<float> input(std::size_t(kM) * kK);
  constexpr std::array<float, 16> positive_block = {
      6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.5f, 1.0f, 0.5f,
      0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  constexpr std::array<float, 16> negative_block = {
      -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.5f, -1.0f, -0.5f,
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  constexpr std::array<float, 16> ordinary_block = {
      6.0f, 4.0f, 3.0f, 2.0f, 1.5f, 1.0f, 0.5f, 0.0f,
      -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f, 0.0f};
  std::copy(positive_block.begin(), positive_block.end(), input.begin());
  std::copy(negative_block.begin(), negative_block.end(), input.begin() + 16);
  std::copy(ordinary_block.begin(), ordinary_block.end(), input.begin() + 32);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> uniform(-6.0f, 6.0f);
  for (std::size_t i = kK; i < input.size(); ++i) input[i] = uniform(rng);

  std::size_t workspace_bytes = 0;
  if (!check_cuda(razer::w4a4::activation_quantization_workspace_size(kM, kK, &workspace_bytes),
                  "workspace query"))
    return 1;
  float* device_input = nullptr;
  float* device_absmax = nullptr;
  void* workspace = nullptr;
  std::size_t max_codes = 0, max_scales = 0;
  if (!check_cuda(razer::w4a4::activation_quantization_output_size(kM, kK, 128, &max_codes, &max_scales),
                  "output size query"))
    return 1;
  std::uint8_t* device_codes = nullptr;
  std::uint8_t* device_scales = nullptr;
  if (!check_cuda(cudaMalloc(&device_input, input.size() * sizeof(float)), "allocate input") ||
      !check_cuda(cudaMalloc(&device_absmax, sizeof(float)), "allocate maximum") ||
      !check_cuda(cudaMalloc(&workspace, workspace_bytes), "allocate workspace") ||
      !check_cuda(cudaMalloc(&device_codes, max_codes), "allocate codes") ||
      !check_cuda(cudaMalloc(&device_scales, max_scales), "allocate scales") ||
      !check_cuda(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice),
                  "copy input"))
    return 1;

  bool correct = true;
  std::vector<std::uint8_t> ref_codes[2], ref_scales[2];
  for (int razer = 0; razer < 2; ++razer) {
    host_quantize(input, kM, kK, razer, ref_codes[razer], ref_scales[razer]);
    auto format = razer ? razer::w4a4::ActivationQuantizationFormat::kRazer
                        : razer::w4a4::ActivationQuantizationFormat::kNvfp4;
    for (int block_m : {16, 32, 64, 128}) {
      std::size_t code_bytes = 0, scale_bytes = 0;
      razer::w4a4::activation_quantization_output_size(kM, kK, block_m, &code_bytes, &scale_bytes);
      if (!check_cuda(razer::w4a4::quantize_activations(device_input, kM, kK, block_m, format, device_codes,
                                                        device_scales, device_absmax, workspace, workspace_bytes,
                                                        nullptr),
                      "quantization") ||
          !check_cuda(cudaDeviceSynchronize(), "quantization synchronize"))
        return 1;
      std::vector<std::uint8_t> codes(code_bytes), scales(scale_bytes);
      float absmax = 0.0f;
      if (!check_cuda(cudaMemcpy(codes.data(), device_codes, code_bytes, cudaMemcpyDeviceToHost), "copy codes") ||
          !check_cuda(cudaMemcpy(scales.data(), device_scales, scale_bytes, cudaMemcpyDeviceToHost), "copy scales") ||
          !check_cuda(cudaMemcpy(&absmax, device_absmax, sizeof(float), cudaMemcpyDeviceToHost), "copy maximum"))
        return 1;
      PackedA expected = pack_a(ref_codes[razer], ref_scales[razer], kM, kK, block_m);
      if (absmax != 6.0f) {
        std::cerr << "unexpected tensor maximum " << absmax << "\n";
        correct = false;
      }
      if (codes != expected.codes || scales != expected.scales) {
        std::cerr << (razer ? "RaZeR" : "NVFP4") << " output differs from the packed host reference at block_m "
                  << block_m << "\n";
        correct = false;
      }
    }
  }

  // RaZeR semantics on the hand-built blocks of row 0 (row-major references, equal to the device
  // outputs after packing as checked above).
  auto const& nc = ref_codes[0];
  auto const& rc = ref_codes[1];
  auto const& ns = ref_scales[0];
  auto const& rs = ref_scales[1];
  for (int block = 0; block < 3; ++block)
    if ((ns[block] & 0x7Fu) != (rs[block] & 0x7Fu)) {
      std::cerr << "scale payload mismatch in block " << block << "\n";
      correct = false;
    }
  if (nc[1] == 0x0u || rc[1] != 0x0u || (rs[0] & 0x80u) != 0u) {
    std::cerr << "positive special was not remapped as expected\n";
    correct = false;
  }
  if (nc[17] == 0x0u || rc[17] != 0x0u || (rs[1] & 0x80u) == 0u) {
    std::cerr << "negative special was not remapped as expected\n";
    correct = false;
  }
  for (int e = 32; e < 48; ++e)
    if (nc[e] != rc[e]) {
      std::cerr << "ordinary block changed at element " << e << "\n";
      correct = false;
      break;
    }

  // Invalid arguments are rejected.
  if (razer::w4a4::quantize_activations(device_input, kM, kK - 16, 16,
                                        razer::w4a4::ActivationQuantizationFormat::kRazer, device_codes,
                                        device_scales, device_absmax, workspace, workspace_bytes,
                                        nullptr) != cudaErrorInvalidValue) {
    std::cerr << "k not a multiple of 64 was not rejected\n";
    correct = false;
  }
  if (razer::w4a4::quantize_activations(device_input, kM, kK, 48,
                                        razer::w4a4::ActivationQuantizationFormat::kRazer, device_codes,
                                        device_scales, device_absmax, workspace, workspace_bytes,
                                        nullptr) != cudaErrorInvalidValue) {
    std::cerr << "invalid block_m was not rejected\n";
    correct = false;
  }
  if (razer::w4a4::quantize_activations(device_input, kM, kK, 16,
                                        static_cast<razer::w4a4::ActivationQuantizationFormat>(-1), device_codes,
                                        device_scales, device_absmax, workspace, workspace_bytes,
                                        nullptr) != cudaErrorInvalidValue) {
    std::cerr << "invalid quantization format was not rejected\n";
    correct = false;
  }

  cudaFree(device_scales);
  cudaFree(device_codes);
  cudaFree(workspace);
  cudaFree(device_absmax);
  cudaFree(device_input);

  if (!correct) return 1;
  std::cout << "activation quantization test: PASS\n";
  return 0;
}
