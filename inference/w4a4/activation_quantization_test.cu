#include "activation_quantization.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool check_cuda(cudaError_t status, char const* operation) {
  if (status == cudaSuccess) return true;
  std::cerr << operation << ": " << cudaGetErrorString(status) << "\n";
  return false;
}

std::uint8_t nibble(
    std::vector<std::uint8_t> const& packed,
    std::size_t index) {
  std::uint8_t byte = packed[index / 2];
  return index % 2 == 0 ? std::uint8_t(byte & 0xFu)
                        : std::uint8_t(byte >> 4);
}

}  // namespace

int main() {
  constexpr std::size_t kBlocks = 3;
  constexpr std::size_t kElements = kBlocks * 16;
  std::array<float, kElements> input{};
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
  std::copy(
      negative_block.begin(), negative_block.end(), input.begin() + 16);
  std::copy(
      ordinary_block.begin(), ordinary_block.end(), input.begin() + 32);

  std::size_t workspace_bytes = 0;
  if (!check_cuda(
          razer::w4a4::activation_quantization_workspace_size(
              kElements, &workspace_bytes),
          "workspace query")) {
    return 1;
  }
  if (workspace_bytes == 0) {
    std::cerr << "workspace query returned zero bytes\n";
    return 1;
  }

  float* device_input = nullptr;
  std::uint8_t* device_native = nullptr;
  std::uint8_t* device_razer = nullptr;
  std::uint8_t* device_native_scales = nullptr;
  std::uint8_t* device_razer_scales = nullptr;
  float* device_native_absmax = nullptr;
  float* device_razer_absmax = nullptr;
  void* workspace = nullptr;

  if (!check_cuda(
          cudaMalloc(&device_input, kElements * sizeof(float)),
          "allocate input") ||
      !check_cuda(
          cudaMalloc(&device_native, kElements / 2),
          "allocate native output") ||
      !check_cuda(
          cudaMalloc(&device_razer, kElements / 2),
          "allocate RaZeR output") ||
      !check_cuda(
          cudaMalloc(&device_native_scales, kBlocks),
          "allocate native scales") ||
      !check_cuda(
          cudaMalloc(&device_razer_scales, kBlocks),
          "allocate RaZeR scales") ||
      !check_cuda(
          cudaMalloc(&device_native_absmax, sizeof(float)),
          "allocate native maximum") ||
      !check_cuda(
          cudaMalloc(&device_razer_absmax, sizeof(float)),
          "allocate RaZeR maximum") ||
      !check_cuda(
          cudaMalloc(&workspace, workspace_bytes),
          "allocate workspace") ||
      !check_cuda(
          cudaMemcpy(
              device_input,
              input.data(),
              kElements * sizeof(float),
              cudaMemcpyHostToDevice),
          "copy input")) {
    return 1;
  }

  if (!check_cuda(
          razer::w4a4::quantize_activations(
              device_input,
              kElements,
              razer::w4a4::ActivationQuantizationFormat::kNvfp4,
              device_native,
              device_native_scales,
              device_native_absmax,
              workspace,
              workspace_bytes,
              nullptr),
          "native quantization") ||
      !check_cuda(
          razer::w4a4::quantize_activations(
              device_input,
              kElements,
              razer::w4a4::ActivationQuantizationFormat::kRazer,
              device_razer,
              device_razer_scales,
              device_razer_absmax,
              workspace,
              workspace_bytes,
              nullptr),
          "RaZeR quantization") ||
      !check_cuda(cudaDeviceSynchronize(), "quantization synchronize")) {
    return 1;
  }

  std::vector<std::uint8_t> native(kElements / 2);
  std::vector<std::uint8_t> razer(kElements / 2);
  std::array<std::uint8_t, kBlocks> native_scales{};
  std::array<std::uint8_t, kBlocks> razer_scales{};
  float native_absmax = 0.0f;
  float razer_absmax = 0.0f;
  if (!check_cuda(
          cudaMemcpy(
              native.data(),
              device_native,
              native.size(),
              cudaMemcpyDeviceToHost),
          "copy native output") ||
      !check_cuda(
          cudaMemcpy(
              razer.data(),
              device_razer,
              razer.size(),
              cudaMemcpyDeviceToHost),
          "copy RaZeR output") ||
      !check_cuda(
          cudaMemcpy(
              native_scales.data(),
              device_native_scales,
              native_scales.size(),
              cudaMemcpyDeviceToHost),
          "copy native scales") ||
      !check_cuda(
          cudaMemcpy(
              razer_scales.data(),
              device_razer_scales,
              razer_scales.size(),
              cudaMemcpyDeviceToHost),
          "copy RaZeR scales") ||
      !check_cuda(
          cudaMemcpy(
              &native_absmax,
              device_native_absmax,
              sizeof(float),
              cudaMemcpyDeviceToHost),
          "copy native maximum") ||
      !check_cuda(
          cudaMemcpy(
              &razer_absmax,
              device_razer_absmax,
              sizeof(float),
              cudaMemcpyDeviceToHost),
          "copy RaZeR maximum")) {
    return 1;
  }

  bool correct = true;
  if (native_absmax != 6.0f || razer_absmax != 6.0f) {
    std::cerr << "unexpected tensor maximum\n";
    correct = false;
  }
  for (std::size_t block = 0; block < kBlocks; ++block) {
    if ((native_scales[block] & 0x7Fu) !=
        (razer_scales[block] & 0x7Fu)) {
      std::cerr << "scale payload mismatch in block " << block << "\n";
      correct = false;
    }
  }
  if (nibble(native, 1) == 0x0u || nibble(razer, 1) != 0x0u) {
    std::cerr << "positive special was not remapped as expected\n";
    correct = false;
  }
  if ((razer_scales[0] & 0x80u) != 0u) {
    std::cerr << "positive-special metadata has the wrong sign\n";
    correct = false;
  }
  if (nibble(native, 17) == 0x0u || nibble(razer, 17) != 0x0u) {
    std::cerr << "negative special was not remapped as expected\n";
    correct = false;
  }
  if ((razer_scales[1] & 0x80u) == 0u) {
    std::cerr << "negative-special metadata has the wrong sign\n";
    correct = false;
  }
  for (std::size_t element = 32; element < kElements; ++element) {
    if (nibble(native, element) != nibble(razer, element)) {
      std::cerr << "ordinary block changed at element " << element << "\n";
      correct = false;
      break;
    }
  }

  if (razer::w4a4::quantize_activations(
          device_input,
          kElements - 1,
          razer::w4a4::ActivationQuantizationFormat::kRazer,
          device_razer,
          device_razer_scales,
          device_razer_absmax,
          workspace,
          workspace_bytes,
          nullptr) != cudaErrorInvalidValue) {
    std::cerr << "invalid element count was not rejected\n";
    correct = false;
  }
  if (razer::w4a4::quantize_activations(
          device_input,
          kElements,
          static_cast<razer::w4a4::ActivationQuantizationFormat>(-1),
          device_razer,
          device_razer_scales,
          device_razer_absmax,
          workspace,
          workspace_bytes,
          nullptr) != cudaErrorInvalidValue) {
    std::cerr << "invalid quantization format was not rejected\n";
    correct = false;
  }

  cudaFree(workspace);
  cudaFree(device_razer_absmax);
  cudaFree(device_native_absmax);
  cudaFree(device_razer_scales);
  cudaFree(device_native_scales);
  cudaFree(device_razer);
  cudaFree(device_native);
  cudaFree(device_input);

  if (!correct) return 1;
  std::cout << "activation quantization test: PASS\n";
  return 0;
}
