// NVFP4 activations x RaZeR weights: W4A4 RaZeR emulation GEMM for SM120a.
//
// Weights are stored in the RaZeR format: 4-bit codes plus one scale byte per 16 values whose two
// spare bits select the special value. Right before the MMAs, each weight is split in registers
// into two ordinary E2M1 components W = Wm + Wc:
//
//   ordinary y: (Wm, Wc) = (y, 0)       +5: (4, 1)   +7: (4, 3)   +8: (4, 4)   +9: (6, 3)
//
// Negative specials negate both components. Both products accumulate in the same registers.
// Operands use the tile layout of razer_w4a4_kernels.cuh.
//
// All CTA configurations below are compiled in; --config selects one per run.
#include "razer_w4a4_host.cuh"

int main(int argc, char** argv) {
  // make<MODE, WM, WN, WARPS_M, WARPS_N, STAGES>: warp tile WM x WN MMA tiles (16 x 8 each),
  // WARPS_M x WARPS_N consumer warps, STAGES pipeline stages (clamped to the shared-memory budget).
  const std::vector<Launcher> reg = {
      make<MODE_WEIGHT, 1, 4, 1, 4, 10>(),
      make<MODE_WEIGHT, 1, 8, 1, 4, 8>(),
      make<MODE_WEIGHT, 1, 4, 1, 8, 8>(),
      make<MODE_WEIGHT, 1, 8, 1, 8, 5>(),
      make<MODE_WEIGHT, 2, 4, 1, 4, 8>(),
      make<MODE_WEIGHT, 2, 8, 1, 4, 8>(),
      make<MODE_WEIGHT, 2, 4, 1, 8, 8>(),
      make<MODE_WEIGHT, 4, 4, 1, 4, 8>(),
      make<MODE_WEIGHT, 4, 8, 1, 4, 6>(),
      make<MODE_WEIGHT, 2, 8, 2, 4, 6>(),
      make<MODE_WEIGHT, 4, 4, 1, 8, 6>(),
      make<MODE_WEIGHT, 4, 4, 2, 4, 6>(),
      make<MODE_WEIGHT, 4, 8, 2, 2, 6>(),
      make<MODE_WEIGHT, 2, 8, 4, 2, 6>(),
      make<MODE_WEIGHT, 4, 4, 4, 2, 6>(),
      make<MODE_WEIGHT, 8, 4, 2, 2, 6>(),
      make<MODE_WEIGHT, 4, 8, 4, 2, 6>(),
      make<MODE_WEIGHT, 4, 8, 2, 4, 6>(),
      make<MODE_WEIGHT, 8, 2, 1, 8, 6>(),
      make<MODE_WEIGHT, 8, 4, 1, 4, 6>(),
      make<MODE_WEIGHT, 8, 4, 1, 8, 4>(),
  };
  return run_driver(argc, argv, reg, /*full=*/false);
}
