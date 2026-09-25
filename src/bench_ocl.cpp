// bench_ocl.cpp - OpenCL 1.2 host program (C API), portable Linux/Windows
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <fstream>
#include <sstream>
#include "common.hpp"
using namespace bench;

#define CK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { std::fprintf(stderr, "OpenCL error %d at %s:%d\n", e_, __FILE__, __LINE__); std::exit(1); } } while (0)

static std::string read_file(const char* p) {
  std::ifstream f(p); std::stringstream s; s << f.rdbuf(); return s.str();
}

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  // device hint "<platform-substring>:<cpu|gpu>"
  std::string want_plat = a.device.substr(0, a.device.find(':'));
  bool want_gpu = a.device.find(":gpu") != std::string::npos;
  Recorder rec; rec.a = a; rec.model = "opencl";

  double t0 = now_ns();
  cl_uint np = 0; CK(clGetPlatformIDs(0, nullptr, &np));
  std::vector<cl_platform_id> plats(np); CK(clGetPlatformIDs(np, plats.data(), nullptr));
  cl_platform_id plat = nullptr; cl_device_id dev = nullptr; char pname[256] = {0}, dname[256] = {0};
  for (auto p : plats) {
    clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof pname, pname, nullptr);
    if (std::string(pname).find(want_plat) == std::string::npos) continue;
    if (clGetDeviceIDs(p, want_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &dev, nullptr) == CL_SUCCESS) { plat = p; break; }
  }
  if (!plat) { std::fprintf(stderr, "no matching platform/device for '%s'\n", a.device.c_str()); return 2; }
  clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof dname, dname, nullptr);
  cl_int err;
  cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CK(err);
  cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err); CK(err);
  rec.t_init = now_ns() - t0;
  std::string pn(pname);
  rec.impl = pn.find("Portable") != std::string::npos ? "pocl" : pn.find("Intel") != std::string::npos ? "intel-ocl"
           : pn.find("AMD") != std::string::npos ? "amd-ocl" : pn;
  rec.dev = want_gpu ? "gpu" : "cpu";
  std::fprintf(stderr, "platform=%s device=%s\n", pname, dname);

  // build (source -> device binary, i.e., online/JIT compilation)
  std::string src = read_file("kernels.cl");
  const char* s = src.c_str(); size_t sl = src.size();
  t0 = now_ns();
  cl_program prog = clCreateProgramWithSource(ctx, 1, &s, &sl, &err); CK(err);
  err = clBuildProgram(prog, 1, &dev, "-cl-std=CL1.2", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t ls; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ls);
    std::string log(ls, 0); clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, ls, &log[0], nullptr);
    std::fprintf(stderr, "%s\n", log.c_str()); return 3;
  }
  const char* kname = a.kernel == "gemm" ? "gemm_tiled" : a.kernel == "conv" ? "conv5" : a.kernel == "kalman" ? "kalman" : "null_kernel";
  cl_kernel k = clCreateKernel(prog, kname, &err); CK(err);
  rec.t_build = now_ns() - t0;

  const int n = a.size;
  std::vector<float> hin1, hin2, hout, ref;
  cl_mem b1 = nullptr, b2 = nullptr, b3 = nullptr, bw = nullptr;
  size_t gws[2] = {1, 1}, lws[2] = {1, 1}; cl_uint dims = 1;
  size_t in1_bytes = 0, in2_bytes = 0, out_bytes = 0;
  KfParams kp = kf_params();

  if (a.kernel == "gemm") {
    hin1.resize((size_t)n * n); hin2.resize((size_t)n * n); hout.resize((size_t)n * n); ref.resize((size_t)n * n);
    Lcg r(1); for (auto& v : hin1) v = r.next(); for (auto& v : hin2) v = r.next();
    ref_gemm(hin1, hin2, ref, n);
    in1_bytes = in2_bytes = out_bytes = sizeof(float) * n * n;
    b1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, in1_bytes, nullptr, &err); CK(err);
    b2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, in2_bytes, nullptr, &err); CK(err);
    b3 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_bytes, nullptr, &err); CK(err);
    CK(clSetKernelArg(k, 0, sizeof(int), &n)); CK(clSetKernelArg(k, 1, sizeof(cl_mem), &b1));
    CK(clSetKernelArg(k, 2, sizeof(cl_mem), &b2)); CK(clSetKernelArg(k, 3, sizeof(cl_mem), &b3));
    dims = 2; gws[0] = gws[1] = n; lws[0] = lws[1] = TS;
  } else if (a.kernel == "conv") {
    hin1.resize((size_t)n * n); hout.resize((size_t)n * n); ref.resize((size_t)n * n);
    Lcg r(2); for (auto& v : hin1) v = r.next();
    std::vector<float> w = gauss5(); ref_conv(hin1, ref, w, n);
    in1_bytes = out_bytes = sizeof(float) * n * n;
    b1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, in1_bytes, nullptr, &err); CK(err);
    b3 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_bytes, nullptr, &err); CK(err);
    bw = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 25 * sizeof(float), w.data(), &err); CK(err);
    CK(clSetKernelArg(k, 0, sizeof(int), &n)); CK(clSetKernelArg(k, 1, sizeof(cl_mem), &b1));
    CK(clSetKernelArg(k, 2, sizeof(cl_mem), &b3)); CK(clSetKernelArg(k, 3, sizeof(cl_mem), &bw));
    dims = 2; gws[0] = gws[1] = n; lws[0] = lws[1] = 16;
  } else if (a.kernel == "kalman") {
    init_tracks(hin1, n); ref = hin1;
    for (int i = 0; i < n; ++i) kf_track_step(&ref[(size_t)i * TRK], kp);
    hout.resize(hin1.size());
    in1_bytes = out_bytes = sizeof(float) * hin1.size();
    b1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, in1_bytes, nullptr, &err); CK(err);
    b3 = b1;
    CK(clSetKernelArg(k, 0, sizeof(int), &n)); CK(clSetKernelArg(k, 1, sizeof(cl_mem), &b1));
    CK(clSetKernelArg(k, 2, sizeof(float), &kp.dt)); CK(clSetKernelArg(k, 3, sizeof(float), &kp.q));
    CK(clSetKernelArg(k, 4, sizeof(float), &kp.r1)); CK(clSetKernelArg(k, 5, sizeof(float), &kp.r2));
    dims = 1; gws[0] = ((n + 63) / 64) * 64; lws[0] = 64;
  } else {  // null kernel: pure launch + synchronisation overhead
    b1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 64, nullptr, &err); CK(err);
    CK(clSetKernelArg(k, 0, sizeof(cl_mem), &b1));
    dims = 1; gws[0] = 1; lws[0] = 1;
  }

  auto upload = [&]() {
    if (in1_bytes) CK(clEnqueueWriteBuffer(q, b1, CL_TRUE, 0, in1_bytes, hin1.data(), 0, nullptr, nullptr));
    if (in2_bytes) CK(clEnqueueWriteBuffer(q, b2, CL_TRUE, 0, in2_bytes, hin2.data(), 0, nullptr, nullptr));
  };
  auto launch = [&]() {
    CK(clEnqueueNDRangeKernel(q, k, dims, nullptr, gws, lws, 0, nullptr, nullptr));
    CK(clFinish(q));
  };
  auto download = [&]() {
    if (out_bytes) CK(clEnqueueReadBuffer(q, b3, CL_TRUE, 0, out_bytes, hout.data(), 0, nullptr, nullptr));
  };

  upload();
  t0 = now_ns(); launch(); rec.t_first = now_ns() - t0;
  if (a.kernel != "null") { download(); rec.err = max_rel_err(hout, ref); }
  for (int i = 0; i < a.warmup; ++i) launch();
  rec.lat.reserve(a.iters); rec.e2e.reserve(a.iters);
  for (int i = 0; i < a.iters; ++i) {
    t0 = now_ns(); launch(); rec.lat.push_back(now_ns() - t0);
    if (a.kernel != "null") { t0 = now_ns(); upload(); launch(); download(); rec.e2e.push_back(now_ns() - t0); }
  }
  rec.write();
  clReleaseKernel(k); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx);
  return 0;
}
