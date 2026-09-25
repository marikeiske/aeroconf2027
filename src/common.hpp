// common.hpp - shared harness for the AeroConf 2027 benchmark suite
// Kernels: K1 SGEMM (tiled), K2 Conv2D 5x5 (perception pre-processing),
//          K3 batched two-sensor Kalman filter (track-level sensor fusion).
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace bench {

using clk = std::chrono::steady_clock;
inline double now_ns() {
  return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
             clk::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- CLI
struct Args {
  std::string kernel = "gemm";   // gemm | conv | kalman | null
  int size = 256;                // gemm: N ; conv: image side ; kalman: #tracks
  int iters = 1000;              // measured iterations
  int warmup = 20;               // discarded iterations (after the first)
  std::string platform = "unknown";
  std::string scenario = "iso";  // iso | intf
  std::string out = "results.csv";
  std::string device = "";       // model-specific device hint
};
inline Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "-k") a.kernel = v;
    else if (k == "-n") a.size = std::atoi(v.c_str());
    else if (k == "-i") a.iters = std::atoi(v.c_str());
    else if (k == "-w") a.warmup = std::atoi(v.c_str());
    else if (k == "-p") a.platform = v;
    else if (k == "-s") a.scenario = v;
    else if (k == "-o") a.out = v;
    else if (k == "-d") a.device = v;
  }
  return a;
}

// ------------------------------------------------------ deterministic data
struct Lcg {
  uint32_t s;
  explicit Lcg(uint32_t seed) : s(seed) {}
  float next() { s = s * 1664525u + 1013904223u; return (float)((s >> 8) & 0xFFFF) / 65536.0f - 0.5f; }
};

constexpr int TS = 16;        // GEMM tile
constexpr int KR = 2;         // conv radius -> 5x5
constexpr int NX = 6, NZ = 3; // Kalman state / measurement dims

inline std::vector<float> gauss5() {
  const float g[5] = {1, 4, 6, 4, 1};
  std::vector<float> w(25);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j) w[i * 5 + j] = g[i] * g[j] / 256.0f;
  return w;
}

// Kalman track layout (structure-of-arrays would be faster on GPU; we keep
// array-of-structures to mirror typical avionics track tables): per track
// x[6], P[36], z1[3] (radar), z2[3] (EO/IR) -> 48 floats.
constexpr int TRK = NX + NX * NX + 2 * NZ;  // 48
struct KfParams { float dt, q, r1, r2; };
inline KfParams kf_params() { return {0.02f, 0.05f, 4.0f, 1.0f}; }  // 50 Hz

inline void init_tracks(std::vector<float>& t, int n) {
  t.assign((size_t)n * TRK, 0.f);
  Lcg r(7);
  for (int i = 0; i < n; ++i) {
    float* p = &t[(size_t)i * TRK];
    for (int k = 0; k < NX; ++k) p[k] = 100.f * r.next();
    float* P = p + NX;
    for (int k = 0; k < NX; ++k) P[k * NX + k] = 10.f;
    float* z1 = p + NX + NX * NX; float* z2 = z1 + NZ;
    for (int k = 0; k < NZ; ++k) { z1[k] = p[k] + r.next(); z2[k] = p[k] + 0.5f * r.next(); }
  }
}

// Reference (and baseline) single-track step, written in plain C so the very
// same arithmetic sequence is reproduced in OpenCL C and SYCL.
inline void kf_track_step(float* p, KfParams k) {
  float* x = p; float* P = p + NX; const float* zs[2] = {p + NX + NX * NX, p + NX + NX * NX + NZ};
  const float rs[2] = {k.r1, k.r2};
  // predict: x = F x ; P = F P F' + Q  (constant velocity, F = [I dtI; 0 I])
  for (int i = 0; i < 3; ++i) x[i] += k.dt * x[i + 3];
  float FP[36];
  for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j)
    FP[i * 6 + j] = P[i * 6 + j] + (i < 3 ? k.dt * P[(i + 3) * 6 + j] : 0.f);
  for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j)
    P[i * 6 + j] = FP[i * 6 + j] + (j < 3 ? k.dt * FP[i * 6 + j + 3] : 0.f) + (i == j ? k.q : 0.f);
  // two sequential updates (radar, EO/IR), H = [I3 0]
  for (int s = 0; s < 2; ++s) {
    const float* z = zs[s];
    float S[9];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) S[i * 3 + j] = P[i * 6 + j] + (i == j ? rs[s] : 0.f);
    float det = S[0] * (S[4] * S[8] - S[5] * S[7]) - S[1] * (S[3] * S[8] - S[5] * S[6]) + S[2] * (S[3] * S[7] - S[4] * S[6]);
    float id = 1.0f / det, Si[9];
    Si[0] = (S[4] * S[8] - S[5] * S[7]) * id; Si[1] = (S[2] * S[7] - S[1] * S[8]) * id; Si[2] = (S[1] * S[5] - S[2] * S[4]) * id;
    Si[3] = (S[5] * S[6] - S[3] * S[8]) * id; Si[4] = (S[0] * S[8] - S[2] * S[6]) * id; Si[5] = (S[2] * S[3] - S[0] * S[5]) * id;
    Si[6] = (S[3] * S[7] - S[4] * S[6]) * id; Si[7] = (S[1] * S[6] - S[0] * S[7]) * id; Si[8] = (S[0] * S[4] - S[1] * S[3]) * id;
    float K[18];  // K = P H' S^-1  (6x3), P H' = first 3 columns of P
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 3; ++j) {
      float a = 0.f; for (int m = 0; m < 3; ++m) a += P[i * 6 + m] * Si[m * 3 + j]; K[i * 3 + j] = a; }
    float y[3]; for (int i = 0; i < 3; ++i) y[i] = z[i] - x[i];
    for (int i = 0; i < 6; ++i) x[i] += K[i * 3 + 0] * y[0] + K[i * 3 + 1] * y[1] + K[i * 3 + 2] * y[2];
    float HP[18]; for (int i = 0; i < 3; ++i) for (int j = 0; j < 6; ++j) HP[i * 6 + j] = P[i * 6 + j];
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j)
      P[i * 6 + j] -= K[i * 3 + 0] * HP[0 * 6 + j] + K[i * 3 + 1] * HP[1 * 6 + j] + K[i * 3 + 2] * HP[2 * 6 + j];
  }
}

// Nominal floating-point operation counts (for throughput reporting)
inline double flops(const std::string& k, int n) {
  if (k == "gemm") return 2.0 * n * (double)n * n;
  if (k == "conv") return 2.0 * 25 * (double)n * n;
  if (k == "kalman") return 1000.0 * n;  // ~985 FLOP counted by hand in kf_track_step
  return 0;
}

// ------------------------------------------------------------ validation
inline double max_rel_err(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    double d = std::fabs((double)a[i] - b[i]) / (std::fabs((double)b[i]) + 1e-3);
    m = std::max(m, d);
  }
  return m;
}

// ------------------------------------------------------------ reference
inline void ref_gemm(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int n) {
  std::fill(C.begin(), C.end(), 0.f);
  for (int i = 0; i < n; ++i) for (int k = 0; k < n; ++k) {
    float a = A[(size_t)i * n + k];
    for (int j = 0; j < n; ++j) C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
  }
}
inline void ref_conv(const std::vector<float>& in, std::vector<float>& out, const std::vector<float>& w, int n) {
  for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) {
    float acc = 0.f;
    for (int dy = -KR; dy <= KR; ++dy) for (int dx = -KR; dx <= KR; ++dx) {
      int yy = std::min(std::max(y + dy, 0), n - 1), xx = std::min(std::max(x + dx, 0), n - 1);
      acc += w[(dy + KR) * 5 + dx + KR] * in[(size_t)yy * n + xx];
    }
    out[(size_t)y * n + x] = acc;
  }
}

// ------------------------------------------------------------ CSV output
struct Recorder {
  Args a; std::string model, impl, dev;
  std::vector<double> lat;   // kernel latency (data resident), ns
  std::vector<double> e2e;   // end-to-end incl. H2D + D2H, ns
  double t_init = 0, t_build = 0, t_first = 0, err = -1;
  void write() {
    FILE* f = std::fopen(a.out.c_str(), "a");
    if (!f) { std::perror("csv"); return; }
    std::fseek(f, 0, SEEK_END);
    if (std::ftell(f) == 0)
      std::fprintf(f, "platform,model,impl,device,kernel,size,scenario,iter,lat_ns,e2e_ns\n");
    for (size_t i = 0; i < lat.size(); ++i)
      std::fprintf(f, "%s,%s,%s,%s,%s,%d,%s,%zu,%.0f,%.0f\n", a.platform.c_str(), model.c_str(), impl.c_str(),
                   dev.c_str(), a.kernel.c_str(), a.size, a.scenario.c_str(), i, lat[i], i < e2e.size() ? e2e[i] : -1.0);
    std::fclose(f);
    std::string mo = a.out + ".meta.csv";
    FILE* g = std::fopen(mo.c_str(), "a");
    std::fseek(g, 0, SEEK_END);
    if (std::ftell(g) == 0)
      std::fprintf(g, "platform,model,impl,device,kernel,size,scenario,t_init_ns,t_build_ns,t_first_ns,max_rel_err\n");
    std::fprintf(g, "%s,%s,%s,%s,%s,%d,%s,%.0f,%.0f,%.0f,%.3e\n", a.platform.c_str(), model.c_str(), impl.c_str(),
                 dev.c_str(), a.kernel.c_str(), a.size, a.scenario.c_str(), t_init, t_build, t_first, err);
    std::fclose(g);
    std::vector<double> s = lat; std::sort(s.begin(), s.end());
    if (!s.empty())
      std::printf("[%s/%s/%s] %s n=%d %s: median=%.1f us p99=%.1f us max=%.1f us err=%.1e\n", model.c_str(), impl.c_str(),
                  dev.c_str(), a.kernel.c_str(), a.size, a.scenario.c_str(), s[s.size() / 2] / 1e3,
                  s[(size_t)(0.99 * (s.size() - 1))] / 1e3, s.back() / 1e3, err);
  }
};

}  // namespace bench
