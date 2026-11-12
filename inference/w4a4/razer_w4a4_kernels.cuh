// W4A4 RaZeR emulation GEMM kernels for SM120a: encodings, operand layout and kernels.
//
// Encoding:
//   * FP4 codes are E2M1 nibbles. In RaZeR operands the positive-zero code 0x0
//     is the special value; ordinary zero is 0x8 (-0).
//   * Weight (B) scale byte: bits[5:0] unsigned E3M3 payload (bias 3),
//     bit7 = special sign, bit6 = magnitude selector (0 -> 5, 1 -> the second
//     magnitude q2 in {7, 8, 9}). MMA payload = (raw & 0x3F) + 0x20, an exact
//     rebias to UE4M3.
//   * RaZeR activation (A) scale byte: bits[6:0] UE4M3 payload, bit7 = special
//     sign. The activation special magnitude is 5.
//   * Native NVFP4 activation scale byte: UE4M3 payload with bit7 clear.
//
// Tile layout of the operands in global memory. Every 16-row (A) or 8-column (B) by 64-wide K
// tile is stored contiguously, already in the register order of the mma.m16n8k64 operand
// fragments, so a CTA loads each pipeline stage with one bulk copy per array:
//   A codes : [mgroup][ks][GM m-tiles][32 lanes] uint4   (m16 x k64 tile in mma.m16n8k64 A-fragment order)
//   A scales: [mgroup][ks][GM m-tiles][16 rows]  uint32  (4 scale bytes for the 4 K-blocks of the slice)
//   B codes : [ngroup][ks][GN n-tiles][32 lanes] uint2   (n8 x k64 tile in B-fragment order)
//   B scales: [ngroup][ks][GN n-tiles][8 cols]   uint32
// with GM = BM / 16 and GN = BN / 8 for the CTA tile BM x BN; mgroup indexes groups of BM rows,
// ngroup groups of BN columns, ks the 64-wide k-slices; a K-block is 16 values sharing a scale.
// These are permutations of the row-major (A) and column-major (B) byte arrays, with rows and
// columns padded up to whole CTA tiles.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define RZF_CHECK(expr)                                                           \
  do {                                                                            \
    cudaError_t _e = (expr);                                                      \
    if (_e != cudaSuccess) {                                                      \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorString(_e),     \
              __FILE__, __LINE__, #expr);                                         \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

// MODE = FORM | APRE << 3 | BPRE << 4 | EXP << 5
//   FORM: which exact formulation the MMAs evaluate
//     FORM_W : native NVFP4 activations x RaZeR weights, W = Wm + Wc (two MMAs)
//     FORM_F3: RaZeR x RaZeR, AB = 2 Am Bm + 2 Ac Bc - (Am - Ac)(Bm - Bc) (three MMAs)
//   APRE / BPRE: the activation / weight components (A = Am + Ac etc.) are read as separate
//     E2M1 arrays, one per MMA product, instead of being split from the RaZeR codes. Operands in
//     global memory are always in the RaZeR format; the component arrays exist only in the
//     second shared-memory buffer of the EXP kernel.
//   EXP: dedicated warps split each loaded tile once into that buffer (rzf_exp_kernel).
enum RzfForm : int { FORM_W = 1, FORM_F3 = 2 };
__host__ __device__ constexpr int mode_form(int mode) { return mode & 7; }
__host__ __device__ constexpr bool mode_apre(int mode) { return (mode >> 3) & 1; }
__host__ __device__ constexpr bool mode_bpre(int mode) { return (mode >> 4) & 1; }
__host__ __device__ constexpr bool mode_exp(int mode) { return (mode >> 5) & 1; }
__host__ __device__ constexpr bool razer_act(int mode) { return mode_form(mode) == FORM_F3; }
__host__ __device__ constexpr int mma_passes(int mode) { return mode_form(mode) == FORM_F3 ? 3 : 2; }
// Number of activation / weight arrays per stage: 1 (RaZeR codes or native NVFP4), or one per product.
__host__ __device__ constexpr int a_planes(int mode) {
  return mode_apre(mode) && mode_form(mode) == FORM_F3 ? 3 : 1;
}
__host__ __device__ constexpr int b_planes(int mode) { return mode_bpre(mode) ? mma_passes(mode) : 1; }

enum RzfMode : int {
  MODE_WEIGHT = FORM_W,             // RaZeR weights split in registers by the MMA warps
  MODE_FULL3 = FORM_F3,             // RaZeR activations and weights split in registers by the MMA warps
  MODE_F3EXP = FORM_F3 | 32,        // RaZeR activations and weights split by dedicated warps
  MODE_F3PRE = FORM_F3 | 8 | 16,    // component arrays in shared memory, read by the MMA warps of F3EXP
};

// Per-tensor constants for the weight special decomposition, replicated nibble words.
struct RzfConsts {
  // FORM_W: W = Wm + Wc.  index 0 -> magnitude 5, index 1 -> second magnitude.
  uint32_t w_main[2];
  uint32_t w_comp[2];
  // FORM_F3: AB = 2 Am Bm + 2 Ac Bc - (Am - Ac)(Bm - Bc).
  uint32_t f_bm[2];
  uint32_t f_bc[2];
  uint32_t f_bd[2];
};

__host__ __device__ inline uint32_t rep8(uint32_t nib) { return nib * 0x11111111u; }

// Native E2M1 components of the weight specials for second magnitude q2.
inline RzfConsts make_consts(int q2) {
  RzfConsts c{};
  // E2M1 codes: 0.5->1 1->2 1.5->3 2->4 3->5 4->6 6->7
  c.w_main[0] = rep8(0x6);  // 5 = 4 + 1
  c.w_comp[0] = rep8(0x2);
  c.f_bm[0] = rep8(0x5);    // 5 = 3 + 2, diff 1
  c.f_bc[0] = rep8(0x4);
  c.f_bd[0] = rep8(0x2);
  switch (q2) {
    case 7:
      c.w_main[1] = rep8(0x6); c.w_comp[1] = rep8(0x5);  // 4 + 3
      c.f_bm[1] = rep8(0x6); c.f_bc[1] = rep8(0x5); c.f_bd[1] = rep8(0x2);  // 4 + 3, diff 1
      break;
    case 8:
      c.w_main[1] = rep8(0x6); c.w_comp[1] = rep8(0x6);  // 4 + 4
      c.f_bm[1] = rep8(0x7); c.f_bc[1] = rep8(0x4); c.f_bd[1] = rep8(0x6);  // 6 + 2, diff 4
      break;
    case 9:
      c.w_main[1] = rep8(0x7); c.w_comp[1] = rep8(0x5);  // 6 + 3
      c.f_bm[1] = rep8(0x7); c.f_bc[1] = rep8(0x5); c.f_bd[1] = rep8(0x5);  // 6 + 3, diff 3
      break;
    default:
      fprintf(stderr, "second magnitude must be 7, 8 or 9\n");
      std::exit(1);
  }
  return c;
}

// 0xF in every nibble equal to 0x0 (the RaZeR special code), else 0.
__device__ __forceinline__ uint32_t special_sel(uint32_t w) {
  uint32_t t = w | (w >> 1);
  t = t | (t >> 2);
  uint32_t zero = (~t) & 0x11111111u;
  return zero * 0xFu;
}

// One m16n8k64 block-scaled FP4 MMA with UE4M3 scales (4 scale bytes per operand register set).
__device__ __forceinline__ void mma_nvf4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
      "%10, {0, 0}, %11, {0, 0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}

// ------------------------------------------------------------------------------------------------
// Kernels. One producer warp loads operand tiles into a circular buffer
// of shared-memory stages with bulk copies (cp.async.bulk, completion tracked by mbarriers); MMA warps
// split the RaZeR special values into E2M1 components, either themselves in registers (rzf_kernel)
// or after dedicated warps have done it once per stage (rzf_exp_kernel), and issue the MMAs. With
// split-K, each CTA writes its partial tile to a workspace and the last CTA to finish a tile adds
// the partials and writes the output, so the output never needs to be zeroed.

struct RzfParams {
  const uint8_t* __restrict__ a;
  const uint8_t* __restrict__ sa;
  const uint8_t* __restrict__ b;
  const uint8_t* __restrict__ sb;
  float* __restrict__ d;
  float* __restrict__ ws;       // split-K partials
  int* __restrict__ counters;   // split-K tile arrival counters (self-resetting)
  int m, n;
  int mtiles, ntiles, kslices;
  int ks_per_split;
  int mgroups, ngroups, splits;  // tile grid
  int num_tiles;
  int group_m;                   // tile order: consecutive tile indices walk N within bands of GROUP_M row groups
  RzfConsts c;
};

// Linear tile index -> (mg, ng, split).  Split is the slowest index so that concurrently
// running CTAs share operand tiles in L2.
__device__ __forceinline__ void tile_coords(const RzfParams& p, int t, int& mg, int& ng, int& z) {
  const int per_split = p.mgroups * p.ngroups;
  z = t / per_split;
  t -= z * per_split;
  const int band = p.group_m * p.ngroups;
  const int b = t / band;
  const int r = t - b * band;
  const int gm = min(p.group_m, p.mgroups - b * p.group_m);
  mg = b * p.group_m + r % gm;
  ng = r / gm;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void bulk_copy(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void consumer_bar(int nthreads) {
  asm volatile("bar.sync 1, %0;" ::"r"(nthreads) : "memory");
}

// STAGES_REQ is clamped so the pipeline fits in the 99 KiB dynamic shared-memory budget.
template <int MODE, int WM, int WN, int WARPS_M, int WARPS_N, int STAGES_REQ>
struct RzfCfg {
  static constexpr int BM = WARPS_M * WM * 16;
  static constexpr int BN = WARPS_N * WN * 8;
  static constexpr int GM = BM / 16;
  static constexpr int GN = BN / 8;
  static constexpr int A_BYTES = a_planes(MODE) * GM * 512;   // [component array][GM][32 lanes] uint4
  static constexpr int SA_BYTES = GM * 64;
  static constexpr int B_BYTES = b_planes(MODE) * GN * 256;   // [component array][GN][32 lanes] uint2
  static constexpr int SB_BYTES = GN * 32;
  static constexpr int STAGE_BYTES = A_BYTES + SA_BYTES + B_BYTES + SB_BYTES;
  static constexpr int STAGES_FIT = (101376 - 256) / STAGE_BYTES;
  static constexpr int STAGES = STAGES_REQ < STAGES_FIT ? STAGES_REQ : STAGES_FIT;
  static constexpr int CONSUMERS = WARPS_M * WARPS_N;
  static constexpr int CTHREADS = CONSUMERS * 32;
  static constexpr int THREADS = CTHREADS + 32;
  static constexpr int SMEM = STAGES * STAGE_BYTES + 2 * STAGES * 8 + 16;
  static_assert(STAGES >= 2, "tile too large for shared memory");
  static constexpr int PARTIAL_FLOATS = CTHREADS * WM * WN * 4;
};

// One k-slice of one warp tile: load the A and B fragments from a shared-memory stage (splitting
// RaZeR codes into E2M1 components unless the stage already holds component arrays), then issue
// all MMAs of the slice.
template <int MODE, int WM, int WN, int WARPS_M, int WARPS_N, int STAGES_REQ>
__device__ __forceinline__ void rzf_compute_stage(float (&acc)[WM][WN][4], const uint8_t* pA, const uint8_t* pSA,
                                                  const uint8_t* pB, const uint8_t* pSB, const RzfParams& p,
                                                  int warp_m, int warp_n, int lane) {
  using C = RzfCfg<MODE, WM, WN, WARPS_M, WARPS_N, STAGES_REQ>;
  constexpr int FORM = mode_form(MODE);
  constexpr int NP = mma_passes(MODE);
  constexpr int NPA = FORM == FORM_F3 ? 3 : 1;
  const int g = lane >> 2;
  const int tig = lane & 3;
  const uint32_t sh_lo = 8u * uint32_t(tig >> 1);
  const uint32_t sh_hi = sh_lo + 16u;
  const uint4* sA = reinterpret_cast<const uint4*>(pA) + warp_m * WM * 32;
  const uint32_t* sSA = reinterpret_cast<const uint32_t*>(pSA) + warp_m * WM * 16;
  const uint2* sB = reinterpret_cast<const uint2*>(pB) + warp_n * WN * 32;
  const uint32_t* sSB = reinterpret_cast<const uint32_t*>(pSB) + warp_n * WN * 8;

  // ---- activation operands: aop[product][i][reg], sfa[i]
  uint32_t aop[NPA][WM][4], sfa[WM];
#pragma unroll
  for (int i = 0; i < WM; ++i) {
    if constexpr (FORM == FORM_F3 && !mode_apre(MODE)) {
      const uint4 araw = sA[i * 32 + lane];
      const uint32_t s0 = sSA[i * 16 + g], s1 = sSA[i * 16 + 8 + g];
      const uint32_t a[4] = {araw.x, araw.y, araw.z, araw.w};
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const uint32_t meta = ((r & 1) ? s1 : s0) >> ((r & 2) ? sh_hi : sh_lo);
        const uint32_t sel = special_sel(a[r]);
        const uint32_t sgn = (meta & 0x80u) ? 0x88888888u : 0u;
        aop[0][i][r] = a[r] | (sel & (0x55555555u ^ sgn));                   // Am: special -> +-3
        aop[1][i][r] = sel & (0x44444444u ^ sgn);                            // Ac: special -> +-2
        aop[2][i][r] = (a[r] | (sel & (0x22222222u ^ sgn))) ^ 0x88888888u;  // -(Am - Ac)
      }
      sfa[i] = ((tig & 1) ? s1 : s0) & 0x7F7F7F7Fu;
    } else {
      // Native NVFP4 activations (FORM_W) or activation component arrays (MODE_F3PRE).
#pragma unroll
      for (int pl = 0; pl < NPA; ++pl) {
        const uint4 araw = sA[(pl * C::GM + i) * 32 + lane];
        aop[pl][i][0] = araw.x; aop[pl][i][1] = araw.y; aop[pl][i][2] = araw.z; aop[pl][i][3] = araw.w;
      }
      sfa[i] = sSA[i * 16 + (tig & 1) * 8 + g];  // already plain UE4M3 (sign bits clear)
    }
  }

  // ---- weight operands: bop[pass][j][reg], sfb[j]
  uint32_t bop[NP][WN][2], sfb[WN];
#pragma unroll
  for (int j = 0; j < WN; ++j) {
    const uint32_t sbw = sSB[j * 8 + g];
    if constexpr (mode_bpre(MODE)) {
#pragma unroll
      for (int pl = 0; pl < NP; ++pl) {
        const uint2 braw = sB[(pl * C::GN + j) * 32 + lane];
        bop[pl][j][0] = braw.x; bop[pl][j][1] = braw.y;
      }
      sfb[j] = sbw;  // already rebiased to UE4M3 by the splitting warps
    } else {
      const uint2 braw = sB[j * 32 + lane];
      const uint32_t w[2] = {braw.x, braw.y};
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const uint32_t meta = sbw >> (r ? sh_hi : sh_lo);
        const uint32_t sel = special_sel(w[r]);
        const uint32_t sgn = (meta & 0x80u) ? 0x88888888u : 0u;
        const bool mag = meta & 0x40u;
        if constexpr (FORM == FORM_W) {
          bop[0][j][r] = w[r] | (sel & ((mag ? p.c.w_main[1] : p.c.w_main[0]) ^ sgn));
          bop[1][j][r] = sel & ((mag ? p.c.w_comp[1] : p.c.w_comp[0]) ^ sgn);
        } else {
          bop[0][j][r] = w[r] | (sel & ((mag ? p.c.f_bm[1] : p.c.f_bm[0]) ^ sgn));
          bop[1][j][r] = sel & ((mag ? p.c.f_bc[1] : p.c.f_bc[0]) ^ sgn);
          bop[2][j][r] = w[r] | (sel & ((mag ? p.c.f_bd[1] : p.c.f_bd[0]) ^ sgn));
        }
      }
      sfb[j] = (sbw & 0x3F3F3F3Fu) + 0x20202020u;
    }
  }

  // ---- MMA issue order: all MMAs of product 0, then of product 1, ..., so that consecutive MMAs
  // write different accumulators and do not wait on each other.
  // FORM_F3 passes 0 and 1 carry the factor 2 by adding 1 to the UE4M3 exponent of the B scales.
#pragma unroll
  for (int pass = 0; pass < NP; ++pass) {
    const uint32_t dbl = (FORM == FORM_F3 && pass < 2) ? 0x08080808u : 0u;
#pragma unroll
    for (int j = 0; j < WN; ++j)
#pragma unroll
      for (int i = 0; i < WM; ++i)
        mma_nvf4(acc[i][j], aop[NPA == 1 ? 0 : pass][i], bop[pass][j][0], bop[pass][j][1], sfa[i], sfb[j] + dbl);
  }
}

// Split-K: every split stores its partial tile; the last arriving CTA reduces and stores D.
template <int WM, int WN, int CTHREADS>
__device__ __forceinline__ void rzf_store_tile(float (&acc)[WM][WN][4], const RzfParams& p, int mg, int ng, int z,
                                               int GM, int GN, int warp_m, int warp_n, int lane, int* s_last) {
  const int g = lane >> 2;
  const int tig = lane & 3;
  const int ctid = threadIdx.x;
  if (p.splits > 1) {
    const int tile = mg * p.ngroups + ng;
    float4* mine = reinterpret_cast<float4*>(p.ws + (size_t(tile) * p.splits + z) * (CTHREADS * WM * WN * 4));
#pragma unroll
    for (int i = 0; i < WM; ++i)
#pragma unroll
      for (int j = 0; j < WN; ++j)
        mine[(i * WN + j) * CTHREADS + ctid] =
            make_float4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
    __threadfence();
    consumer_bar(CTHREADS);
    if (ctid == 0) *s_last = (atomicAdd(&p.counters[tile], 1) == p.splits - 1);
    consumer_bar(CTHREADS);
    const bool last = *s_last;
    consumer_bar(CTHREADS);  // s_last is reused by the next tile
    if (!last) return;
    __threadfence();
    for (int zz = 0; zz < p.splits; ++zz) {
      if (zz == z) continue;
      const float4* other =
          reinterpret_cast<const float4*>(p.ws + (size_t(tile) * p.splits + zz) * (CTHREADS * WM * WN * 4));
#pragma unroll
      for (int i = 0; i < WM; ++i)
#pragma unroll
        for (int j = 0; j < WN; ++j) {
          const float4 v = __ldcg(other + (i * WN + j) * CTHREADS + ctid);
          acc[i][j][0] += v.x; acc[i][j][1] += v.y; acc[i][j][2] += v.z; acc[i][j][3] += v.w;
        }
    }
    if (ctid == 0) p.counters[tile] = 0;
  }

  const int mt0 = mg * GM + warp_m * WM;
  const int nt0 = ng * GN + warp_n * WN;
#pragma unroll
  for (int i = 0; i < WM; ++i) {
#pragma unroll
    for (int j = 0; j < WN; ++j) {
      if (nt0 + j >= p.ntiles) continue;
      const int col = (nt0 + j) * 8 + tig * 2;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = (mt0 + i) * 16 + g + h * 8;
        if (row >= p.m) continue;
        *reinterpret_cast<float2*>(p.d + size_t(row) * p.n + col) =
            make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
      }
    }
  }
}

// Launched with at most one CTA per SM; each CTA loops over output tiles blockIdx.x,
// blockIdx.x + gridDim.x, ...
// Warp CONSUMERS is the producer (one lane issues the bulk copies); warps 0..CONSUMERS-1 compute.
template <int MODE, int WM, int WN, int WARPS_M, int WARPS_N, int STAGES_REQ>
__global__ void __launch_bounds__(RzfCfg<MODE, WM, WN, WARPS_M, WARPS_N, STAGES_REQ>::THREADS)
    rzf_kernel(const RzfParams p) {
  using C = RzfCfg<MODE, WM, WN, WARPS_M, WARPS_N, STAGES_REQ>;
  extern __shared__ __align__(128) uint8_t smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + C::STAGES * C::STAGE_BYTES);
  uint64_t* empty = full + C::STAGES;
  int* s_last = reinterpret_cast<int*>(empty + C::STAGES);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  if (threadIdx.x == 0) {
    for (int s = 0; s < C::STAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], C::CONSUMERS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == C::CONSUMERS) {
    if (lane == 0) {
      int it = 0;
      for (int t = blockIdx.x; t < p.num_tiles; t += gridDim.x) {
        int mg, ng, z;
        tile_coords(p, t, mg, ng, z);
        const int ks_begin = z * p.ks_per_split;
        const int ks_end = min(p.kslices, ks_begin + p.ks_per_split);
        const uint8_t* ga = p.a + (size_t(mg) * p.kslices) * C::A_BYTES;
        const uint8_t* gsa = p.sa + (size_t(mg) * p.kslices) * C::SA_BYTES;
        const uint8_t* gb = p.b + (size_t(ng) * p.kslices) * C::B_BYTES;
        const uint8_t* gsb = p.sb + (size_t(ng) * p.kslices) * C::SB_BYTES;
        for (int ks = ks_begin; ks < ks_end; ++ks, ++it) {
          const int s = it % C::STAGES;
          if (it >= C::STAGES) mbar_wait(&empty[s], ((it / C::STAGES) - 1) & 1);
          uint8_t* st = smem + s * C::STAGE_BYTES;
          mbar_expect_tx(&full[s], C::STAGE_BYTES);
          bulk_copy(st, ga + size_t(ks) * C::A_BYTES, C::A_BYTES, &full[s]);
          bulk_copy(st + C::A_BYTES, gsa + size_t(ks) * C::SA_BYTES, C::SA_BYTES, &full[s]);
          bulk_copy(st + C::A_BYTES + C::SA_BYTES, gb + size_t(ks) * C::B_BYTES, C::B_BYTES, &full[s]);
          bulk_copy(st + C::A_BYTES + C::SA_BYTES + C::B_BYTES, gsb + size_t(ks) * C::SB_BYTES,
                    C::SB_BYTES, &full[s]);
        }
      }
    }
    return;
  }

  const int warp_m = warp / WARPS_N;
  const int warp_n = warp % WARPS_N;
  int it = 0;
  for (int t = blockIdx.x; t < p.num_tiles; t += gridDim.x) {
    int mg, ng, z;
    tile_coords(p, t, mg, ng, z);
    const int ks_begin = z * p.ks_per_split;
    const int ks_end = min(p.kslices, ks_begin + p.ks_per_split);

    float acc[WM][WN][4];
#pragma unroll
    for (int i = 0; i < WM; ++i)
#pragma unroll
      for (int j = 0; j < WN; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

    for (int ks = ks_begin; ks < ks_end; ++ks, ++it) {
      const int s = it % C::STAGES;
      mbar_wait(&full[s], (it / C::STAGES) & 1);
      rzf_compute_stage<MODE, WM, WN, WARPS_M, WARPS_N, STAGES_REQ>(
          acc, smem + s * C::STAGE_BYTES, smem + s * C::STAGE_BYTES + C::A_BYTES,
          smem + s * C::STAGE_BYTES + C::A_BYTES + C::SA_BYTES,
          smem + s * C::STAGE_BYTES + C::A_BYTES + C::SA_BYTES + C::B_BYTES, p, warp_m, warp_n, lane);
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);
    }

    rzf_store_tile<WM, WN, C::CTHREADS>(acc, p, mg, ng, z, C::GM, C::GN, warp_m, warp_n, lane, s_last);
  }
}

// ------------------------------------------------------------------------------------------------
// Kernel with dedicated splitting warps (full RaZeR): operands stay in the RaZeR format in global
// memory. A producer warp bulk-copies RaZeR tiles into a first circular buffer; EXPANDERS warps
// split each stage once into separate E2M1 component arrays in a second circular buffer; the MMA
// warps read those arrays (MODE_F3PRE), so no two MMA warps split the same values.
// ------------------------------------------------------------------------------------------------
template <int MODE, int WM, int WN, int WARPS_M, int WARPS_N, int CSTAGES, int PSTAGES, int EXPANDERS>
struct RzfExpCfg {
  static constexpr int FORM = mode_form(MODE);
  static_assert(FORM == FORM_F3, "the kernel with splitting warps implements full RaZeR only");
  static constexpr int PRE = MODE_F3PRE;
  using P = RzfCfg<PRE, WM, WN, WARPS_M, WARPS_N, 2>;
  using Cc = RzfCfg<FORM, WM, WN, WARPS_M, WARPS_N, 2>;
  static constexpr int BM = P::BM, BN = P::BN, GM = P::GM, GN = P::GN;
  static constexpr int C_STAGE = Cc::A_BYTES + Cc::SA_BYTES + Cc::B_BYTES + Cc::SB_BYTES;
  static constexpr int P_STAGE = P::A_BYTES + P::SA_BYTES + P::B_BYTES + P::SB_BYTES;
  static constexpr int CONSUMERS = WARPS_M * WARPS_N;
  static constexpr int CTHREADS = CONSUMERS * 32;
  static constexpr int XTHREADS = EXPANDERS * 32;
  static constexpr int THREADS = CTHREADS + XTHREADS + 32;
  static constexpr int BAR_OFF = CSTAGES * C_STAGE + PSTAGES * P_STAGE;
  static constexpr int SMEM = BAR_OFF + 2 * (CSTAGES + PSTAGES) * 8 + 16;
};

template <int MODE, int WM, int WN, int WARPS_M, int WARPS_N, int CSTAGES, int PSTAGES, int EXPANDERS>
__global__ void __launch_bounds__(RzfExpCfg<MODE, WM, WN, WARPS_M, WARPS_N, CSTAGES, PSTAGES, EXPANDERS>::THREADS)
    rzf_exp_kernel(const RzfParams p) {
  using X = RzfExpCfg<MODE, WM, WN, WARPS_M, WARPS_N, CSTAGES, PSTAGES, EXPANDERS>;
  using Cc = typename X::Cc;
  using P = typename X::P;
  extern __shared__ __align__(128) uint8_t smem[];
  uint64_t* cfull = reinterpret_cast<uint64_t*>(smem + X::BAR_OFF);
  uint64_t* cempty = cfull + CSTAGES;
  uint64_t* pfull = cempty + CSTAGES;
  uint64_t* pempty = pfull + PSTAGES;
  int* s_last = reinterpret_cast<int*>(pempty + PSTAGES);
  auto cstage = [&](int s) { return smem + s * X::C_STAGE; };
  auto pstage = [&](int s) { return smem + CSTAGES * X::C_STAGE + s * X::P_STAGE; };

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) {
    for (int s = 0; s < CSTAGES; ++s) { mbar_init(&cfull[s], 1); mbar_init(&cempty[s], EXPANDERS); }
    for (int s = 0; s < PSTAGES; ++s) { mbar_init(&pfull[s], EXPANDERS); mbar_init(&pempty[s], X::CONSUMERS); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == X::CONSUMERS + EXPANDERS) {  // producer
    if (lane == 0) {
      int it = 0;
      for (int t = blockIdx.x; t < p.num_tiles; t += gridDim.x) {
        int mg, ng, z;
        tile_coords(p, t, mg, ng, z);
        const int ks_begin = z * p.ks_per_split;
        const int ks_end = min(p.kslices, ks_begin + p.ks_per_split);
        const uint8_t* ga = p.a + (size_t(mg) * p.kslices) * Cc::A_BYTES;
        const uint8_t* gsa = p.sa + (size_t(mg) * p.kslices) * Cc::SA_BYTES;
        const uint8_t* gb = p.b + (size_t(ng) * p.kslices) * Cc::B_BYTES;
        const uint8_t* gsb = p.sb + (size_t(ng) * p.kslices) * Cc::SB_BYTES;
        for (int ks = ks_begin; ks < ks_end; ++ks, ++it) {
          const int s = it % CSTAGES;
          if (it >= CSTAGES) mbar_wait(&cempty[s], ((it / CSTAGES) - 1) & 1);
          uint8_t* st = cstage(s);
          mbar_expect_tx(&cfull[s], X::C_STAGE);
          bulk_copy(st, ga + size_t(ks) * Cc::A_BYTES, Cc::A_BYTES, &cfull[s]);
          bulk_copy(st + Cc::A_BYTES, gsa + size_t(ks) * Cc::SA_BYTES, Cc::SA_BYTES, &cfull[s]);
          bulk_copy(st + Cc::A_BYTES + Cc::SA_BYTES, gb + size_t(ks) * Cc::B_BYTES, Cc::B_BYTES, &cfull[s]);
          bulk_copy(st + Cc::A_BYTES + Cc::SA_BYTES + Cc::B_BYTES, gsb + size_t(ks) * Cc::SB_BYTES, Cc::SB_BYTES,
                    &cfull[s]);
        }
      }
    }
    return;
  }

  if (warp >= X::CONSUMERS) {  // splitting warps
    const int xt = (warp - X::CONSUMERS) * 32 + lane;
    int it = 0;
    for (int t = blockIdx.x; t < p.num_tiles; t += gridDim.x) {
      int mg, ng, z;
      tile_coords(p, t, mg, ng, z);
      const int ks_begin = z * p.ks_per_split;
      const int ks_end = min(p.kslices, ks_begin + p.ks_per_split);
      for (int ks = ks_begin; ks < ks_end; ++ks, ++it) {
        const int cs = it % CSTAGES, ps = it % PSTAGES;
        mbar_wait(&cfull[cs], (it / CSTAGES) & 1);
        if (it >= PSTAGES) mbar_wait(&pempty[ps], ((it / PSTAGES) - 1) & 1);
        const uint8_t* c = cstage(cs);
        uint8_t* q = pstage(ps);
        const uint4* cA = reinterpret_cast<const uint4*>(c);
        const uint32_t* cSA = reinterpret_cast<const uint32_t*>(c + Cc::A_BYTES);
        const uint2* cB = reinterpret_cast<const uint2*>(c + Cc::A_BYTES + Cc::SA_BYTES);
        const uint32_t* cSB = reinterpret_cast<const uint32_t*>(c + Cc::A_BYTES + Cc::SA_BYTES + Cc::B_BYTES);
        uint4* qA = reinterpret_cast<uint4*>(q);
        uint32_t* qSA = reinterpret_cast<uint32_t*>(q + P::A_BYTES);
        uint2* qB = reinterpret_cast<uint2*>(q + P::A_BYTES + P::SA_BYTES);
        uint32_t* qSB = reinterpret_cast<uint32_t*>(q + P::A_BYTES + P::SA_BYTES + P::B_BYTES);
        // activations
        for (int idx = xt; idx < X::GM * 32; idx += X::XTHREADS) {
          const int i = idx >> 5, l = idx & 31, g = l >> 2, tig = l & 3;
          const uint32_t sh_lo = 8u * uint32_t(tig >> 1), sh_hi = sh_lo + 16u;
          const uint4 araw = cA[idx];
          const uint32_t s0 = cSA[i * 16 + g], s1 = cSA[i * 16 + 8 + g];
          const uint32_t a[4] = {araw.x, araw.y, araw.z, araw.w};
          uint32_t o[3][4];
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            const uint32_t meta = ((r & 1) ? s1 : s0) >> ((r & 2) ? sh_hi : sh_lo);
            const uint32_t sel = special_sel(a[r]);
            const uint32_t sgn = (meta & 0x80u) ? 0x88888888u : 0u;
            o[0][r] = a[r] | (sel & (0x55555555u ^ sgn));
            o[1][r] = sel & (0x44444444u ^ sgn);
            o[2][r] = (a[r] | (sel & (0x22222222u ^ sgn))) ^ 0x88888888u;
          }
#pragma unroll
          for (int pl = 0; pl < 3; ++pl) qA[(pl * X::GM + i) * 32 + l] = make_uint4(o[pl][0], o[pl][1], o[pl][2], o[pl][3]);
        }
        for (int idx = xt; idx < X::GM * 16; idx += X::XTHREADS)
          qSA[idx] = cSA[idx] & 0x7F7F7F7Fu;
        // weights
        for (int idx = xt; idx < X::GN * 32; idx += X::XTHREADS) {
          const int j = idx >> 5, l = idx & 31, g = l >> 2, tig = l & 3;
          const uint32_t sh_lo = 8u * uint32_t(tig >> 1), sh_hi = sh_lo + 16u;
          const uint2 braw = cB[idx];
          const uint32_t sbw = cSB[j * 8 + g];
          const uint32_t w[2] = {braw.x, braw.y};
          uint32_t o[3][2];
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const uint32_t meta = sbw >> (r ? sh_hi : sh_lo);
            const uint32_t sel = special_sel(w[r]);
            const uint32_t sgn = (meta & 0x80u) ? 0x88888888u : 0u;
            const bool mag = meta & 0x40u;
            o[0][r] = w[r] | (sel & ((mag ? p.c.f_bm[1] : p.c.f_bm[0]) ^ sgn));
            o[1][r] = sel & ((mag ? p.c.f_bc[1] : p.c.f_bc[0]) ^ sgn);
            o[2][r] = w[r] | (sel & ((mag ? p.c.f_bd[1] : p.c.f_bd[0]) ^ sgn));
          }
#pragma unroll
          for (int pl = 0; pl < mma_passes(X::PRE); ++pl) qB[(pl * X::GN + j) * 32 + l] = make_uint2(o[pl][0], o[pl][1]);
        }
        for (int idx = xt; idx < X::GN * 8; idx += X::XTHREADS) qSB[idx] = (cSB[idx] & 0x3F3F3F3Fu) + 0x20202020u;
        __syncwarp();
        if (lane == 0) {
          mbar_arrive(&cempty[cs]);
          mbar_arrive(&pfull[ps]);
        }
      }
    }
    return;
  }

  const int warp_m = warp / WARPS_N;
  const int warp_n = warp % WARPS_N;
  int it = 0;
  for (int t = blockIdx.x; t < p.num_tiles; t += gridDim.x) {
    int mg, ng, z;
    tile_coords(p, t, mg, ng, z);
    const int ks_begin = z * p.ks_per_split;
    const int ks_end = min(p.kslices, ks_begin + p.ks_per_split);
    float acc[WM][WN][4];
#pragma unroll
    for (int i = 0; i < WM; ++i)
#pragma unroll
      for (int j = 0; j < WN; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;
    for (int ks = ks_begin; ks < ks_end; ++ks, ++it) {
      const int s = it % PSTAGES;
      mbar_wait(&pfull[s], (it / PSTAGES) & 1);
      const uint8_t* q = pstage(s);
      rzf_compute_stage<X::PRE, WM, WN, WARPS_M, WARPS_N, 2>(acc, q, q + P::A_BYTES, q + P::A_BYTES + P::SA_BYTES,
                                                              q + P::A_BYTES + P::SA_BYTES + P::B_BYTES, p, warp_m,
                                                              warp_n, lane);
      __syncwarp();
      if (lane == 0) mbar_arrive(&pempty[s]);
    }
    rzf_store_tile<WM, WN, X::CTHREADS>(acc, p, mg, ng, z, X::GM, X::GN, warp_m, warp_n, lane, s_last);
  }
}
