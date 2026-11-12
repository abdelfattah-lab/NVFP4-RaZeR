// RaZeR activations x RaZeR weights: W4A4 RaZeR emulation GEMM for SM120a.
//
// Each RaZeR value is split into ordinary E2M1 components A = Am + Ac, B = Bm + Bc, and
//
//   A B = 2 Am Bm + 2 Ac Bc - (Am - Ac)(Bm - Bc)
//
// is evaluated as three block-scaled FP4 MMA passes into the same accumulators (the factor 2 is
// folded into the B scale exponent). Activation special 5 = 3 + 2; weight specials
// 5 = 3 + 2, 7 = 4 + 3, 8 = 6 + 2, 9 = 6 + 3; negative specials negate both components.
// Operands use the tile layout of razer_w4a4_kernels.cuh. Two kernels do the split:
//
//   BMxBN_w<..>_s<..>            every MMA warp splits the values it loads, in registers;
//   BMxBN_w<..>_c<..>p<..>e<..>  e extra warps split each shared-memory stage once into a second
//                                buffer that the MMA warps read.
//
// All CTA configurations below are compiled in; --config selects one per run.
#include "razer_w4a4_host.cuh"

int main(int argc, char** argv) {
  // make<MODE, WM, WN, WARPS_M, WARPS_N, STAGES>: warp tile WM x WN MMA tiles (16 x 8 each),
  // WARPS_M x WARPS_N consumer warps, STAGES pipeline stages (clamped to the shared-memory budget).
  const std::vector<Launcher> reg = {
      make<MODE_FULL3, 1, 4, 1, 4, 10>(),
      make<MODE_FULL3, 1, 8, 1, 4, 8>(),
      make<MODE_FULL3, 1, 4, 1, 8, 8>(),
      make<MODE_FULL3, 1, 8, 1, 8, 5>(),
      make<MODE_FULL3, 2, 4, 1, 4, 8>(),
      make<MODE_FULL3, 2, 8, 1, 4, 8>(),
      make<MODE_FULL3, 2, 4, 1, 8, 8>(),
      make<MODE_FULL3, 4, 4, 1, 4, 8>(),
      make<MODE_FULL3, 4, 8, 1, 4, 6>(),
      make<MODE_FULL3, 2, 8, 2, 4, 6>(),
      make<MODE_FULL3, 4, 4, 1, 8, 6>(),
      make<MODE_FULL3, 4, 4, 2, 4, 6>(),
      make<MODE_FULL3, 4, 8, 2, 2, 6>(),
      make<MODE_FULL3, 2, 8, 4, 2, 6>(),
      make<MODE_FULL3, 4, 4, 4, 2, 6>(),
      make<MODE_FULL3, 8, 4, 2, 2, 6>(),
      make<MODE_FULL3, 4, 8, 4, 2, 6>(),
      make<MODE_FULL3, 4, 8, 2, 4, 6>(),
      make<MODE_FULL3, 8, 2, 1, 8, 6>(),
      make<MODE_FULL3, 8, 4, 1, 4, 6>(),
      make<MODE_FULL3, 8, 4, 1, 8, 4>(),
      // make_exp<MODE, WM, WN, WARPS_M, WARPS_N, stages of loaded RaZeR tiles, stages of split
      //          components, splitting warps>
      make_exp<MODE_F3EXP, 4, 4, 2, 4, 3, 2, 1>(),
      make_exp<MODE_F3EXP, 4, 4, 2, 4, 3, 2, 2>(),
      make_exp<MODE_F3EXP, 8, 2, 1, 8, 3, 2, 2>(),
      make_exp<MODE_F3EXP, 4, 4, 2, 4, 2, 2, 3>(),
      make_exp<MODE_F3EXP, 2, 4, 4, 2, 3, 3, 2>(),
      make_exp<MODE_F3EXP, 4, 2, 2, 4, 3, 3, 2>(),
      make_exp<MODE_F3EXP, 4, 4, 2, 4, 2, 2, 4>(),
      make_exp<MODE_F3EXP, 4, 4, 2, 4, 2, 2, 6>(),
      make_exp<MODE_F3EXP, 8, 2, 1, 8, 2, 2, 4>(),
      make_exp<MODE_F3EXP, 4, 8, 2, 2, 2, 2, 4>(),
      make_exp<MODE_F3EXP, 4, 4, 2, 4, 3, 2, 4>(),
      make_exp<MODE_F3EXP, 4, 4, 2, 4, 3, 2, 5>(),
      make_exp<MODE_F3EXP, 2, 4, 4, 2, 3, 3, 4>(),
  };
  return run_driver(argc, argv, reg, /*full=*/true);
}
