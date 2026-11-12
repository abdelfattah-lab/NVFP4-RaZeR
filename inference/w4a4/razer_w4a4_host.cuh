// Host-side helpers for the W4A4 RaZeR emulation GEMMs: synthetic operand generation, the exact
// fp64 reference, and rearrangement of row-major activations / column-major weights into the
// tile layout described in razer_w4a4_kernels.cuh, and the command-line driver of the two
// GEMM programs.
#pragma once
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <cstring>
#include <functional>
#include <string>
#include "razer_w4a4_kernels.cuh"

static double e2m1(uint8_t c) {
  static const double mag[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
  return (c & 8) ? -mag[c & 7] : mag[c & 7];
}
static double ue4m3(uint8_t p) {
  int e = (p >> 3) & 0xF, m = p & 7;
  if (e == 0) return std::ldexp(m / 8.0, -6);
  return std::ldexp(1.0 + m / 8.0, e - 7);
}
static double e3m3(uint8_t p) { return ue4m3(uint8_t((p & 0x3F) + 0x20)); }

// Unpacked operands: A codes / scales row-major [m][k] / [m][k/16]; B column-major [n][k] / [n][k/16].
struct HostProblem {
  int m, n, k, kb, q2;
  std::vector<uint8_t> a, b, sa, sb;
};

// Dequantized activation A[r][kk] (native NVFP4 for FORM_W, RaZeR for FORM_F3).
static double a_value(const HostProblem& h, int mode, int r, int kk) {
  uint8_t c = h.a[size_t(r) * h.k + kk];
  uint8_t s = h.sa[size_t(r) * h.kb + kk / 16];
  if (razer_act(mode) && c == 0) return ((s & 0x80) ? -5.0 : 5.0) * ue4m3(s & 0x7F);
  return e2m1(c) * ue4m3(s & 0x7F);
}
// Dequantized RaZeR weight B[col][kk].
static double b_value(const HostProblem& h, int col, int kk) {
  uint8_t c = h.b[size_t(col) * h.k + kk];
  uint8_t s = h.sb[size_t(col) * h.kb + kk / 16];
  double sc = e3m3(s);
  if (c == 0) {
    double mag = (s & 0x40) ? h.q2 : 5.0;
    return ((s & 0x80) ? -mag : mag) * sc;
  }
  return e2m1(c) * sc;
}

// Random operands: special codes at the given rates, ordinary codes uniform over the 15 non-special
// codes (all 16 for native NVFP4 activations), and scales from a small set of exact values so that
// every product and partial sum is exactly representable in fp32.
static HostProblem make_problem(int mode, int m, int n, int k, int q2, double a_rate, double w_rate,
                                uint64_t seed) {
  HostProblem h;
  h.m = m; h.n = n; h.k = k; h.kb = k / 16; h.q2 = q2;
  h.a.resize(size_t(m) * k); h.b.resize(size_t(n) * k);
  h.sa.resize(size_t(m) * h.kb); h.sb.resize(size_t(n) * h.kb);
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> nib_all(0, 15), nib_ord(1, 15), bit(0, 1), choice(0, 3);
  std::bernoulli_distribution a_sp(a_rate), w_sp(w_rate);
  const uint8_t ue4m3_set[4] = {0x30, 0x38, 0x3C, 0x40};
  const uint8_t e3m3_set[4] = {0x10, 0x18, 0x1C, 0x20};
  for (auto& v : h.a) v = razer_act(mode) ? (a_sp(rng) ? 0 : nib_ord(rng)) : nib_all(rng);
  for (auto& v : h.b) v = w_sp(rng) ? 0 : nib_ord(rng);
  for (auto& s : h.sa) s = ue4m3_set[choice(rng)] | ((razer_act(mode) && bit(rng)) ? 0x80 : 0);
  for (auto& s : h.sb) s = e3m3_set[choice(rng)] | (bit(rng) ? 0x80 : 0) | (bit(rng) ? 0x40 : 0);
  return h;
}

// Activations in the tile layout for CTA tile height BM: rows are padded to whole BM groups with code
// 0x88 (two -0 nibbles) and scale 0x38 (1.0).
struct PackedA {
  std::vector<uint8_t> codes, scales;
  int mgroups, mtiles, kslices;
};
static PackedA pack_a(const std::vector<uint8_t>& codes, const std::vector<uint8_t>& scales, int m, int k, int BM) {
  PackedA P;
  const int GM = BM / 16, kb = k / 16;
  P.kslices = k / 64;
  P.mgroups = (m + BM - 1) / BM;
  P.mtiles = P.mgroups * GM;
  P.codes.assign(size_t(P.mtiles) * P.kslices * 512, 0x88);
  P.scales.assign(size_t(P.mtiles) * P.kslices * 64, 0x38);
  auto nib = [&](int r, int kk) -> uint32_t { return r < m ? codes[size_t(r) * k + kk] : 0x8u; };
  for (int mt = 0; mt < P.mtiles; ++mt) {
    const int grp = mt / GM, in = mt % GM;
    for (int ks = 0; ks < P.kslices; ++ks) {
      const size_t tile = (size_t(grp) * P.kslices + ks) * GM + in;
      uint32_t* acodes = reinterpret_cast<uint32_t*>(P.codes.data() + tile * 512);
      for (int lane = 0; lane < 32; ++lane) {
        // mma.m16n8k64 A fragment: lane (g, t) holds rows g and g + 8, k = t*8 .. t*8+7 and +32.
        int g = lane >> 2, t = lane & 3;
        uint32_t r[4] = {0, 0, 0, 0};
        for (int i = 0; i < 8; ++i) {
          int k0 = ks * 64 + t * 8 + i;
          r[0] |= nib(mt * 16 + g, k0) << (4 * i);
          r[1] |= nib(mt * 16 + g + 8, k0) << (4 * i);
          r[2] |= nib(mt * 16 + g, k0 + 32) << (4 * i);
          r[3] |= nib(mt * 16 + g + 8, k0 + 32) << (4 * i);
        }
        for (int q = 0; q < 4; ++q) acodes[lane * 4 + q] = r[q];
      }
      uint32_t* asc = reinterpret_cast<uint32_t*>(P.scales.data() + tile * 64);
      for (int row = 0; row < 16; ++row) {
        int rr = mt * 16 + row;
        uint32_t w = 0;
        for (int q = 0; q < 4; ++q)
          w |= uint32_t(rr < m ? scales[size_t(rr) * kb + ks * 4 + q] : 0x38) << (8 * q);
        asc[row] = w;
      }
    }
  }
  return P;
}

// Weights in the tile layout for CTA tile width BN (done once, offline): columns are padded to
// whole BN groups with code 0x88 and scale 0x18 (E3M3 1.0).
struct PackedB {
  std::vector<uint8_t> codes, scales;
  int ngroups, ntiles, kslices;
};
static PackedB pack_b(const HostProblem& h, int BN) {
  PackedB P;
  const int GN = BN / 8;
  P.kslices = h.k / 64;
  P.ngroups = (h.n + BN - 1) / BN;
  P.ntiles = h.n / 8;
  P.codes.assign(size_t(P.ngroups) * GN * P.kslices * 256, 0x88);
  P.scales.assign(size_t(P.ngroups) * GN * P.kslices * 32, 0x18);
  auto nib = [&](int c, int kk) -> uint32_t { return c < h.n ? h.b[size_t(c) * h.k + kk] : 0x8u; };
  for (int nt = 0; nt < P.ngroups * GN; ++nt) {
    const int grp = nt / GN, in = nt % GN;
    for (int ks = 0; ks < P.kslices; ++ks) {
      const size_t tile = (size_t(grp) * P.kslices + ks) * GN + in;
      uint32_t* bcodes = reinterpret_cast<uint32_t*>(P.codes.data() + tile * 256);
      for (int lane = 0; lane < 32; ++lane) {
        // mma.m16n8k64 B fragment: lane (g, t) holds column g, k = t*8 .. t*8+7 and +32.
        int g = lane >> 2, t = lane & 3;
        uint32_t r[2] = {0, 0};
        for (int i = 0; i < 8; ++i) {
          int k0 = ks * 64 + t * 8 + i;
          r[0] |= nib(nt * 8 + g, k0) << (4 * i);
          r[1] |= nib(nt * 8 + g, k0 + 32) << (4 * i);
        }
        bcodes[lane * 2] = r[0]; bcodes[lane * 2 + 1] = r[1];
      }
      uint32_t* bsc = reinterpret_cast<uint32_t*>(P.scales.data() + tile * 32);
      for (int col = 0; col < 8; ++col) {
        int cc = nt * 8 + col;
        uint32_t w = 0;
        for (int q = 0; q < 4; ++q)
          w |= uint32_t(cc < h.n ? h.sb[size_t(cc) * h.kb + ks * 4 + q] : 0x18) << (8 * q);
        bsc[col] = w;
      }
    }
  }
  return P;
}

// ------------------------------------------------------------------------------------------------
// Command-line driver shared by razer_weight_w4a4.cu and razer_full_w4a4.cu: generates synthetic
// operands with the requested special-value rates, lays them out for the selected configuration,
// optionally compares sampled outputs with an exact fp64 reference, and times the GEMM.
//
// Timing: a cache flush of --flush-mb MiB precedes every timed launch; each launch is bracketed
// by CUDA events on one stream; the mean, median and minimum over --iters launches are reported.
// Operand generation, packing and the reference are outside the timed region.


// One compiled CTA configuration of a kernel.
struct Launcher {
  std::string name;
  int mode, bm, bn;
  std::function<void(const RzfParams&, dim3, cudaStream_t)> launch;
};

// rzf_kernel configuration: name BMxBN_w<warps_m>x<warps_n>_s<stages>.
template <int MODE, int WM, int WN, int WARPS_M, int WARPS_N, int STAGES>
Launcher make() {
  using C = RzfCfg<MODE, WM, WN, WARPS_M, WARPS_N, STAGES>;
  static_assert(C::SMEM <= 101376, "shared memory budget");
  Launcher l;
  l.name = std::to_string(C::BM) + "x" + std::to_string(C::BN) + "_w" + std::to_string(WARPS_M) + "x" +
           std::to_string(WARPS_N) + "_s" + std::to_string(C::STAGES);
  l.mode = MODE; l.bm = C::BM; l.bn = C::BN;
  auto k = rzf_kernel<MODE, WM, WN, WARPS_M, WARPS_N, STAGES>;
  RZF_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));  // once, outside timing
  l.launch = [k](const RzfParams& p, dim3 grid, cudaStream_t st) { k<<<grid, C::THREADS, C::SMEM, st>>>(p); };
  return l;
}

// rzf_exp_kernel configuration: name BMxBN_w<warps_m>x<warps_n>_c<RaZeR stages>p<component stages>e<warps>.
template <int MODE, int WM, int WN, int WARPS_M, int WARPS_N, int CS, int PS, int E>
Launcher make_exp() {
  using X = RzfExpCfg<MODE, WM, WN, WARPS_M, WARPS_N, CS, PS, E>;
  static_assert(X::SMEM <= 101376, "shared memory budget");
  Launcher l;
  l.name = std::to_string(X::BM) + "x" + std::to_string(X::BN) + "_w" + std::to_string(WARPS_M) + "x" +
           std::to_string(WARPS_N) + "_c" + std::to_string(CS) + "p" + std::to_string(PS) + "e" + std::to_string(E);
  l.mode = MODE; l.bm = X::BM; l.bn = X::BN;
  auto k = rzf_exp_kernel<MODE, WM, WN, WARPS_M, WARPS_N, CS, PS, E>;
  RZF_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, X::SMEM));  // once, outside timing
  l.launch = [k](const RzfParams& p, dim3 grid, cudaStream_t st) { k<<<grid, X::THREADS, X::SMEM, st>>>(p); };
  return l;
}

__global__ void flush_kernel(unsigned char* buf, size_t n) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  size_t stride = size_t(blockDim.x) * gridDim.x;
  for (; i < n; i += stride) buf[i] = (unsigned char)i;
}

// Parses --key=value arguments; every measurement parameter is required (no defaults).
struct DriverArgs {
  std::string config;
  long long split_k = -1, m = -1, n = -1, k = -1, warmup = -1, iters = -1, flush_mb = -1, seed = -1, q2 = -1,
            check_samples = -1;
  double a_rate = -1, b_rate = -1, max_err = -1;
  bool list = false;
};

static int usage(const char* prog, bool full) {
  fprintf(stderr,
          "Usage: %s --config=<name> --split-k=<n> --m=<m> --n=<n> --k=<k> --warmup=<n> --iters=<n>\n"
          "          --flush-mb=<MiB> --seed=<n> --b-special-rate=<p> --b-second-magnitude=<7|8|9>%s\n"
          "          [--check-samples=<n> --max-normalized-error=<e>]\n"
          "       %s --list-configs\n",
          prog, full ? " --a-special-rate=<p>" : "", prog);
  return 2;
}

static int run_driver(int argc, char** argv, const std::vector<Launcher>& reg, bool full) {
  const char* prog = argv[0];
  DriverArgs a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto val = [&](const char* key) -> const char* {
      size_t l = strlen(key);
      return (s.compare(0, l, key) == 0 && s.size() > l && s[l] == '=') ? s.c_str() + l + 1 : nullptr;
    };
    if (s == "--list-configs") a.list = true;
    else if (const char* v = val("--config")) a.config = v;
    else if (const char* v = val("--split-k")) a.split_k = atoll(v);
    else if (const char* v = val("--m")) a.m = atoll(v);
    else if (const char* v = val("--n")) a.n = atoll(v);
    else if (const char* v = val("--k")) a.k = atoll(v);
    else if (const char* v = val("--warmup")) a.warmup = atoll(v);
    else if (const char* v = val("--iters")) a.iters = atoll(v);
    else if (const char* v = val("--flush-mb")) a.flush_mb = atoll(v);
    else if (const char* v = val("--seed")) a.seed = atoll(v);
    else if (const char* v = val("--b-second-magnitude")) a.q2 = atoll(v);
    else if (const char* v = val("--b-special-rate")) a.b_rate = atof(v);
    else if (full && val("--a-special-rate")) a.a_rate = atof(val("--a-special-rate"));
    else if (const char* v = val("--check-samples")) a.check_samples = atoll(v);
    else if (const char* v = val("--max-normalized-error")) a.max_err = atof(v);
    else { fprintf(stderr, "unknown argument %s\n", s.c_str()); return usage(prog, full); }
  }
  if (a.list) {
    for (auto& l : reg) printf("%s\n", l.name.c_str());
    return 0;
  }
  if (a.config.empty() || a.split_k < 1 || a.m < 1 || a.n < 1 || a.k < 1 || a.warmup < 0 || a.iters < 1 ||
      a.flush_mb < 0 || a.seed < 0 || a.b_rate < 0 || a.b_rate > 1 || (full && (a.a_rate < 0 || a.a_rate > 1)))
    return usage(prog, full);
  if (a.q2 != 7 && a.q2 != 8 && a.q2 != 9) { fprintf(stderr, "--b-second-magnitude must be 7, 8 or 9\n"); return 2; }
  if ((a.check_samples < 0) != (a.max_err < 0)) {
    fprintf(stderr, "--check-samples and --max-normalized-error must be given together\n");
    return 2;
  }
  if (a.k % 64 || a.n % 8) { fprintf(stderr, "K must be a multiple of 64 and N a multiple of 8\n"); return 2; }
  const Launcher* lp = nullptr;
  for (auto& l : reg) if (l.name == a.config) lp = &l;
  if (!lp) { fprintf(stderr, "unknown --config %s (see --list-configs)\n", a.config.c_str()); return 2; }
  const Launcher& l = *lp;
  const int m = int(a.m), n = int(a.n), k = int(a.k), mode = l.mode;

  HostProblem h = make_problem(mode, m, n, k, int(a.q2), full ? a.a_rate : 0.0, a.b_rate, uint64_t(a.seed));
  PackedA PA = pack_a(h.a, h.sa, m, k, l.bm);
  PackedB PB = pack_b(h, l.bn);

  // Split-K: at most one split per k-slice; empty trailing splits are dropped.
  int sp = int(std::min<long long>(a.split_k, PA.kslices));
  const int ks_per_split = (PA.kslices + sp - 1) / sp;
  sp = (PA.kslices + ks_per_split - 1) / ks_per_split;

  uint8_t *da, *dsa, *db, *dsb;
  RZF_CHECK(cudaMalloc(&da, PA.codes.size())); RZF_CHECK(cudaMalloc(&dsa, PA.scales.size()));
  RZF_CHECK(cudaMalloc(&db, PB.codes.size())); RZF_CHECK(cudaMalloc(&dsb, PB.scales.size()));
  RZF_CHECK(cudaMemcpy(da, PA.codes.data(), PA.codes.size(), cudaMemcpyHostToDevice));
  RZF_CHECK(cudaMemcpy(dsa, PA.scales.data(), PA.scales.size(), cudaMemcpyHostToDevice));
  RZF_CHECK(cudaMemcpy(db, PB.codes.data(), PB.codes.size(), cudaMemcpyHostToDevice));
  RZF_CHECK(cudaMemcpy(dsb, PB.scales.data(), PB.scales.size(), cudaMemcpyHostToDevice));
  float* d = nullptr;
  RZF_CHECK(cudaMalloc(&d, size_t(m) * n * 4));
  // Split-K partials: one BM x BN fp32 tile per (tile, split); one arrival counter per tile.
  const size_t tiles = size_t(PA.mgroups) * PB.ngroups;
  float* ws = nullptr;
  int* counters = nullptr;
  if (sp > 1) RZF_CHECK(cudaMalloc(&ws, tiles * sp * l.bm * l.bn * 4));
  RZF_CHECK(cudaMalloc(&counters, tiles * 4));
  RZF_CHECK(cudaMemset(counters, 0, tiles * 4));

  RzfParams p;
  p.a = da; p.sa = dsa; p.b = db; p.sb = dsb; p.d = d; p.ws = ws; p.counters = counters;
  p.m = m; p.n = n; p.mtiles = PA.mtiles; p.ntiles = PB.ntiles; p.kslices = PA.kslices;
  p.c = make_consts(int(a.q2));
  p.ks_per_split = ks_per_split;
  p.mgroups = PA.mgroups; p.ngroups = PB.ngroups; p.splits = sp;
  p.num_tiles = int(tiles) * sp;
  p.group_m = 8;
  int num_sms = 0;
  RZF_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  dim3 grid(std::min(p.num_tiles, num_sms));  // at most one CTA per SM; each CTA loops over tiles

  cudaStream_t st;
  RZF_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  auto run = [&]() { l.launch(p, grid, st); };

  int status = 0;
  double max_rel = 0;
  if (a.check_samples >= 0) {
    // Sampled outputs against the exact fp64 dot product; error normalized by sum |a_i b_i|.
    RZF_CHECK(cudaMemsetAsync(d, 0xFF, size_t(m) * n * 4, st));
    run();
    RZF_CHECK(cudaStreamSynchronize(st));
    RZF_CHECK(cudaGetLastError());
    std::vector<float> hd(size_t(m) * n);
    RZF_CHECK(cudaMemcpy(hd.data(), d, hd.size() * 4, cudaMemcpyDeviceToHost));
    std::mt19937_64 rng(uint64_t(a.seed) ^ 0x9e3779b97f4a7c15ull);
    int bad = 0;
    for (long long t = 0; t < a.check_samples; ++t) {
      int r = int(rng() % m), c = int(rng() % n);
      double s = 0, sabs = 0;
      for (int kk = 0; kk < k; ++kk) {
        double v = a_value(h, mode, r, kk) * b_value(h, c, kk);
        s += v; sabs += std::fabs(v);
      }
      double err = std::fabs(double(hd[size_t(r) * n + c]) - s) / std::max(sabs, 1e-30);
      max_rel = std::max(max_rel, err);
      if (!(err <= a.max_err)) ++bad;
    }
    printf("CHECK samples=%lld max_normalized_error=%.3g threshold=%.3g %s\n", a.check_samples, max_rel, a.max_err,
           bad ? "FAIL" : "PASS");
    if (bad) status = 1;
  }

  unsigned char* flush = nullptr;
  const size_t flush_bytes = size_t(a.flush_mb) << 20;
  if (flush_bytes) RZF_CHECK(cudaMalloc(&flush, flush_bytes));
  cudaEvent_t e0, e1;
  RZF_CHECK(cudaEventCreate(&e0)); RZF_CHECK(cudaEventCreate(&e1));
  for (long long w = 0; w < a.warmup; ++w) run();
  std::vector<float> ms(a.iters);
  for (long long it = 0; it < a.iters; ++it) {
    if (flush) flush_kernel<<<1024, 256, 0, st>>>(flush, flush_bytes);
    RZF_CHECK(cudaEventRecord(e0, st));
    run();
    RZF_CHECK(cudaEventRecord(e1, st));
    RZF_CHECK(cudaEventSynchronize(e1));
    RZF_CHECK(cudaEventElapsedTime(&ms[it], e0, e1));
  }
  RZF_CHECK(cudaGetLastError());
  double mean = 0;
  for (float v : ms) mean += v;
  mean /= double(a.iters);
  std::sort(ms.begin(), ms.end());
  // Logical throughput: 2 M N K of the original GEMM, not the 2 or 3 tensor-core passes.
  const double tflops = 2.0 * m * n * k / (mean * 1e-3) / 1e12;
  printf("RESULT config=%s m=%d n=%d k=%d splits=%d mean_ms=%.6f median_ms=%.6f min_ms=%.6f tflops=%.2f\n",
         l.name.c_str(), m, n, k, sp, mean, ms[a.iters / 2], ms[0], tflops);
  return status;
}
