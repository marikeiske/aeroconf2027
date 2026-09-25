// bench_sycl.cpp - SYCL 2020 host+device program. Memory model selected at run
// time: "-d usm" (explicit device USM + memcpy, closest to OpenCL semantics) or
// "-d buf" (buffers/accessors, runtime-managed dependencies and transfers).
#include <sycl/sycl.hpp>
#include "common.hpp"
using namespace bench;

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  const bool usm = a.device != "buf";
  Recorder rec; rec.a = a; rec.model = "sycl";
  const int n = a.size;
  KfParams kp = kf_params();

  double t0 = now_ns();
  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
  rec.t_init = now_ns() - t0;
  auto d = q.get_device();
  std::string be = d.get_platform().get_info<sycl::info::platform::name>();
  #ifndef ACPP_FLAVOR
#define ACPP_FLAVOR "jit"
#endif
  const bool ocl_be = be.find("OpenCL") != std::string::npos || be.find("Intel") != std::string::npos || be.find("Portable") != std::string::npos;
  rec.impl = std::string("acpp-") + (ocl_be ? "ocl" : "omp") + "-" + ACPP_FLAVOR + "-" + (usm ? "usm" : "buf");
  rec.dev = d.is_gpu() ? "gpu" : "cpu";
  std::fprintf(stderr, "platform=%s device=%s\n", be.c_str(), d.get_info<sycl::info::device::name>().c_str());

  std::vector<float> h1, h2, ho, ref, w = gauss5();
  size_t n1 = 0, n2 = 0, no = 0;
  if (a.kernel == "gemm") {
    n1 = n2 = no = (size_t)n * n; h1.resize(n1); h2.resize(n2); ho.resize(no); ref.resize(no);
    Lcg r(1); for (auto& v : h1) v = r.next(); for (auto& v : h2) v = r.next(); ref_gemm(h1, h2, ref, n);
  } else if (a.kernel == "conv") {
    n1 = no = (size_t)n * n; h1.resize(n1); ho.resize(no); ref.resize(no);
    Lcg r(2); for (auto& v : h1) v = r.next(); ref_conv(h1, ref, w, n);
  } else if (a.kernel == "kalman") {
    init_tracks(h1, n); ref = h1; for (int i = 0; i < n; ++i) kf_track_step(&ref[(size_t)i * TRK], kp);
    n1 = no = h1.size(); ho.resize(no);
  } else { n1 = 16; h1.assign(16, 0.f); }

  // ---- device storage
  float *d1 = nullptr, *d2 = nullptr, *d3 = nullptr, *dw = nullptr;
  sycl::buffer<float, 1>* B1 = nullptr; sycl::buffer<float, 1>* B2 = nullptr;
  sycl::buffer<float, 1>* B3 = nullptr; sycl::buffer<float, 1>* BW = nullptr;
  if (usm) {
    d1 = sycl::malloc_device<float>(n1, q);
    if (n2) d2 = sycl::malloc_device<float>(n2, q);
    d3 = (a.kernel == "gemm" || a.kernel == "conv") ? sycl::malloc_device<float>(no, q) : d1;
    dw = sycl::malloc_device<float>(25, q); q.memcpy(dw, w.data(), 25 * sizeof(float)).wait();
  } else {
    B1 = new sycl::buffer<float, 1>(sycl::range<1>(n1));
    if (n2) B2 = new sycl::buffer<float, 1>(sycl::range<1>(n2));
    B3 = (a.kernel == "gemm" || a.kernel == "conv") ? new sycl::buffer<float, 1>(sycl::range<1>(no)) : B1;
    BW = new sycl::buffer<float, 1>(w.data(), sycl::range<1>(25));
  }

  auto upload = [&]() {
    if (usm) { q.memcpy(d1, h1.data(), n1 * sizeof(float)); if (n2) q.memcpy(d2, h2.data(), n2 * sizeof(float)); q.wait(); }
    else {
      q.submit([&](sycl::handler& h) { sycl::accessor acc{*B1, h, sycl::write_only, sycl::no_init}; h.copy(h1.data(), acc); });
      if (n2) q.submit([&](sycl::handler& h) { sycl::accessor acc{*B2, h, sycl::write_only, sycl::no_init}; h.copy(h2.data(), acc); });
      q.wait();
    }
  };
  auto download = [&]() {
    if (!no) return;
    if (usm) q.memcpy(ho.data(), d3, no * sizeof(float)).wait();
    else { q.submit([&](sycl::handler& h) { sycl::accessor acc{*B3, h, sycl::read_only}; h.copy(acc, ho.data()); }); q.wait(); }
  };

  // ---- kernels (bodies identical to kernels.cl)
  auto launch = [&]() {
    if (a.kernel == "gemm") {
      q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 2> As{sycl::range<2>(TS, TS), h}, Bs{sycl::range<2>(TS, TS), h};
        const float *A = d1, *B = d2; float* C = d3;
        auto body = [=](sycl::nd_item<2> it, const float* A, const float* B, float* C) {
          const int lr = it.get_local_id(0), lc = it.get_local_id(1);
          const int row = it.get_global_id(0), col = it.get_global_id(1);
          float acc = 0.f;
          for (int t = 0; t < n / TS; ++t) {
            As[lr][lc] = A[row * n + t * TS + lc];
            Bs[lr][lc] = B[(t * TS + lr) * n + col];
            sycl::group_barrier(it.get_group());
            for (int k = 0; k < TS; ++k) acc += As[lr][k] * Bs[k][lc];
            sycl::group_barrier(it.get_group());
          }
          C[row * n + col] = acc;
        };
        sycl::nd_range<2> r{{(size_t)n, (size_t)n}, {TS, TS}};
        if (usm) h.parallel_for(r, [=](sycl::nd_item<2> it) { body(it, A, B, C); });
        else {
          sycl::accessor a1{*B1, h, sycl::read_only}, a2{*B2, h, sycl::read_only};
          sycl::accessor a3{*B3, h, sycl::write_only, sycl::no_init};
          h.parallel_for(r, [=](sycl::nd_item<2> it) {
            body(it, a1.get_multi_ptr<sycl::access::decorated::no>().get(), a2.get_multi_ptr<sycl::access::decorated::no>().get(),
                 a3.get_multi_ptr<sycl::access::decorated::no>().get()); });
        }
      });
    } else if (a.kernel == "conv") {
      q.submit([&](sycl::handler& h) {
        auto body = [=](sycl::nd_item<2> it, const float* in, float* out, const float* w) {
          const int y = it.get_global_id(0), x = it.get_global_id(1);
          float acc = 0.f;
          for (int dy = -KR; dy <= KR; ++dy)
            for (int dx = -KR; dx <= KR; ++dx) {
              int yy = sycl::min(sycl::max(y + dy, 0), n - 1), xx = sycl::min(sycl::max(x + dx, 0), n - 1);
              acc += w[(dy + KR) * 5 + dx + KR] * in[yy * n + xx];
            }
          out[y * n + x] = acc;
        };
        sycl::nd_range<2> r{{(size_t)n, (size_t)n}, {16, 16}};
        if (usm) { const float* in = d1; float* out = d3; const float* ww = dw;
          h.parallel_for(r, [=](sycl::nd_item<2> it) { body(it, in, out, ww); }); }
        else {
          sycl::accessor a1{*B1, h, sycl::read_only}, aw{*BW, h, sycl::read_only};
          sycl::accessor a3{*B3, h, sycl::write_only, sycl::no_init};
          h.parallel_for(r, [=](sycl::nd_item<2> it) {
            body(it, a1.get_multi_ptr<sycl::access::decorated::no>().get(), a3.get_multi_ptr<sycl::access::decorated::no>().get(),
                 aw.get_multi_ptr<sycl::access::decorated::no>().get()); });
        }
      });
    } else if (a.kernel == "kalman") {
      q.submit([&](sycl::handler& h) {
        const size_t g = ((size_t)(n + 63) / 64) * 64;
        auto body = [=](sycl::nd_item<1> it, float* T) {
          const int id = it.get_global_id(0);
          if (id >= n) return;
          float x[6], P[36];
          float* p = T + (size_t)id * TRK;
          for (int i = 0; i < 6; ++i) x[i] = p[i];
          for (int i = 0; i < 36; ++i) P[i] = p[NX + i];
          const float dt = kp.dt, q_ = kp.q;
          for (int i = 0; i < 3; ++i) x[i] += dt * x[i + 3];
          float FP[36];
          for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j)
            FP[i * 6 + j] = P[i * 6 + j] + (i < 3 ? dt * P[(i + 3) * 6 + j] : 0.f);
          for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j)
            P[i * 6 + j] = FP[i * 6 + j] + (j < 3 ? dt * FP[i * 6 + j + 3] : 0.f) + (i == j ? q_ : 0.f);
          for (int s = 0; s < 2; ++s) {
            const float* z = p + NX + NX * NX + s * NZ;
            const float r = (s == 0) ? kp.r1 : kp.r2;
            float S[9];
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) S[i * 3 + j] = P[i * 6 + j] + (i == j ? r : 0.f);
            float det = S[0] * (S[4] * S[8] - S[5] * S[7]) - S[1] * (S[3] * S[8] - S[5] * S[6]) + S[2] * (S[3] * S[7] - S[4] * S[6]);
            float id_ = 1.0f / det, Si[9];
            Si[0] = (S[4] * S[8] - S[5] * S[7]) * id_; Si[1] = (S[2] * S[7] - S[1] * S[8]) * id_; Si[2] = (S[1] * S[5] - S[2] * S[4]) * id_;
            Si[3] = (S[5] * S[6] - S[3] * S[8]) * id_; Si[4] = (S[0] * S[8] - S[2] * S[6]) * id_; Si[5] = (S[2] * S[3] - S[0] * S[5]) * id_;
            Si[6] = (S[3] * S[7] - S[4] * S[6]) * id_; Si[7] = (S[1] * S[6] - S[0] * S[7]) * id_; Si[8] = (S[0] * S[4] - S[1] * S[3]) * id_;
            float K[18];
            for (int i = 0; i < 6; ++i) for (int j = 0; j < 3; ++j) {
              float acc = 0.f; for (int m = 0; m < 3; ++m) acc += P[i * 6 + m] * Si[m * 3 + j]; K[i * 3 + j] = acc; }
            float y[3]; for (int i = 0; i < 3; ++i) y[i] = z[i] - x[i];
            for (int i = 0; i < 6; ++i) x[i] += K[i * 3 + 0] * y[0] + K[i * 3 + 1] * y[1] + K[i * 3 + 2] * y[2];
            float HP[18]; for (int i = 0; i < 3; ++i) for (int j = 0; j < 6; ++j) HP[i * 6 + j] = P[i * 6 + j];
            for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j)
              P[i * 6 + j] -= K[i * 3 + 0] * HP[0 * 6 + j] + K[i * 3 + 1] * HP[1 * 6 + j] + K[i * 3 + 2] * HP[2 * 6 + j];
          }
          for (int i = 0; i < 6; ++i) p[i] = x[i];
          for (int i = 0; i < 36; ++i) p[NX + i] = P[i];
        };
        sycl::nd_range<1> r{g, 64};
        if (usm) { float* T = d1; h.parallel_for(r, [=](sycl::nd_item<1> it) { body(it, T); }); }
        else { sycl::accessor a1{*B1, h, sycl::read_write};
          h.parallel_for(r, [=](sycl::nd_item<1> it) { body(it, a1.get_multi_ptr<sycl::access::decorated::no>().get()); }); }
      });
    } else {  // null kernel
      if (usm) { float* p = d1; q.submit([&](sycl::handler& h) { h.single_task([=]() { if (p[0] == 12345.f) p[0] = 0.f; }); }); }
      else q.submit([&](sycl::handler& h) { sycl::accessor a1{*B1, h, sycl::read_write}; h.single_task([=]() { if (a1[0] == 12345.f) a1[0] = 0.f; }); });
    }
    q.wait();
  };

  upload();
  t0 = now_ns(); launch(); rec.t_first = now_ns() - t0;  // includes JIT (generic SSCP target)
  if (a.kernel != "null") { download(); rec.err = max_rel_err(ho, ref); }
  for (int i = 0; i < a.warmup; ++i) launch();
  for (int i = 0; i < a.iters; ++i) {
    t0 = now_ns(); launch(); rec.lat.push_back(now_ns() - t0);
    if (a.kernel != "null") { t0 = now_ns(); upload(); launch(); download(); rec.e2e.push_back(now_ns() - t0); }
  }
  rec.write();
  if (usm) { sycl::free(d1, q); if (d2) sycl::free(d2, q); if (d3 != d1) sycl::free(d3, q); sycl::free(dw, q); }
  return 0;
}
