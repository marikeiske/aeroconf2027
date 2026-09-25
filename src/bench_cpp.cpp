// bench_cpp.cpp - native baseline: ISO C++ (serial) and C++/OpenMP
#include <omp.h>
#include "common.hpp"
using namespace bench;

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  const bool par = a.device != "serial";
  Recorder rec; rec.a = a; rec.model = "cpp"; rec.impl = par ? "openmp" : "serial"; rec.dev = "cpu";
  if (!par) omp_set_num_threads(1);
  const int n = a.size;
  std::vector<float> A, B, C, ref, work, w = gauss5();
  KfParams kp = kf_params();

  if (a.kernel == "gemm") {
    A.resize((size_t)n * n); B.resize((size_t)n * n); C.resize((size_t)n * n); ref.resize((size_t)n * n);
    Lcg r(1); for (auto& v : A) v = r.next(); for (auto& v : B) v = r.next();
    ref_gemm(A, B, ref, n);
  } else if (a.kernel == "conv") {
    A.resize((size_t)n * n); C.resize((size_t)n * n); ref.resize((size_t)n * n);
    Lcg r(2); for (auto& v : A) v = r.next(); ref_conv(A, ref, w, n);
  } else if (a.kernel == "kalman") {
    init_tracks(A, n); ref = A; for (int i = 0; i < n; ++i) kf_track_step(&ref[(size_t)i * TRK], kp);
    C = A;
  }
  float* Ap = A.data(); float* Bp = B.data(); float* Cp = C.data(); const float* wp = w.data();

  auto compute = [&]() {
    if (a.kernel == "gemm") {
#pragma omp parallel for schedule(static) if (par)
      for (int i = 0; i < n; ++i) {
        float* c = Cp + (size_t)i * n;
        for (int j = 0; j < n; ++j) c[j] = 0.f;
        for (int k = 0; k < n; ++k) { float av = Ap[(size_t)i * n + k]; const float* b = Bp + (size_t)k * n;
          for (int j = 0; j < n; ++j) c[j] += av * b[j]; }
      }
    } else if (a.kernel == "conv") {
#pragma omp parallel for schedule(static) if (par)
      for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) {
        float acc = 0.f;
        for (int dy = -KR; dy <= KR; ++dy) for (int dx = -KR; dx <= KR; ++dx) {
          int yy = std::min(std::max(y + dy, 0), n - 1), xx = std::min(std::max(x + dx, 0), n - 1);
          acc += wp[(dy + KR) * 5 + dx + KR] * Ap[(size_t)yy * n + xx]; }
        Cp[(size_t)y * n + x] = acc;
      }
    } else if (a.kernel == "kalman") {
#pragma omp parallel for schedule(static) if (par)
      for (int i = 0; i < n; ++i) kf_track_step(Cp + (size_t)i * TRK, kp);
    } else {
#pragma omp parallel if (par)
      { if (omp_get_thread_num() == 12345) Cp = nullptr; }
    }
  };
  auto stage = [&]() { if (a.kernel == "kalman") std::copy(A.begin(), A.end(), C.begin()); };

  double t0 = now_ns();
  compute(); rec.t_first = now_ns() - t0;
  if (a.kernel != "null") rec.err = max_rel_err(C, ref);
  for (int i = 0; i < a.warmup; ++i) compute();
  for (int i = 0; i < a.iters; ++i) {
    t0 = now_ns(); compute(); rec.lat.push_back(now_ns() - t0);
    if (a.kernel != "null") { t0 = now_ns(); stage(); compute(); rec.e2e.push_back(now_ns() - t0); }
  }
  rec.write();
  return 0;
}
