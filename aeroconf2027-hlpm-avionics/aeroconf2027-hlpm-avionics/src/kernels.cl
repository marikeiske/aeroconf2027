// kernels.cl - OpenCL C 1.2 kernels (identical arithmetic to common.hpp)
#define TS 16
#define KR 2
#define NX 6
#define NZ 3
#define TRK 48

__kernel void null_kernel(__global float* d) { if (get_global_id(0) == 0xFFFFFFFF) d[0] = 0.f; }

__kernel void gemm_tiled(const int n, __global const float* A, __global const float* B, __global float* C) {
  __local float As[TS][TS];
  __local float Bs[TS][TS];
  const int lr = get_local_id(1), lc = get_local_id(0);
  const int row = get_global_id(1), col = get_global_id(0);
  float acc = 0.f;
  for (int t = 0; t < n / TS; ++t) {
    As[lr][lc] = A[row * n + t * TS + lc];
    Bs[lr][lc] = B[(t * TS + lr) * n + col];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < TS; ++k) acc += As[lr][k] * Bs[k][lc];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  C[row * n + col] = acc;
}

__kernel void conv5(const int n, __global const float* in, __global float* out, __constant float* w) {
  const int x = get_global_id(0), y = get_global_id(1);
  float acc = 0.f;
  for (int dy = -KR; dy <= KR; ++dy)
    for (int dx = -KR; dx <= KR; ++dx) {
      int yy = min(max(y + dy, 0), n - 1), xx = min(max(x + dx, 0), n - 1);
      acc += w[(dy + KR) * 5 + dx + KR] * in[yy * n + xx];
    }
  out[y * n + x] = acc;
}

__kernel void kalman(const int n, __global float* T, const float dt, const float q, const float r1, const float r2) {
  const int id = get_global_id(0);
  if (id >= n) return;
  __global float* p = T + id * TRK;
  float x[6], P[36];
  for (int i = 0; i < 6; ++i) x[i] = p[i];
  for (int i = 0; i < 36; ++i) P[i] = p[NX + i];
  for (int i = 0; i < 3; ++i) x[i] += dt * x[i + 3];
  float FP[36];
  for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j)
    FP[i * 6 + j] = P[i * 6 + j] + (i < 3 ? dt * P[(i + 3) * 6 + j] : 0.f);
  for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j)
    P[i * 6 + j] = FP[i * 6 + j] + (j < 3 ? dt * FP[i * 6 + j + 3] : 0.f) + (i == j ? q : 0.f);
  for (int s = 0; s < 2; ++s) {
    __global const float* z = p + NX + NX * NX + s * NZ;
    const float r = (s == 0) ? r1 : r2;
    float S[9];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) S[i * 3 + j] = P[i * 6 + j] + (i == j ? r : 0.f);
    float det = S[0] * (S[4] * S[8] - S[5] * S[7]) - S[1] * (S[3] * S[8] - S[5] * S[6]) + S[2] * (S[3] * S[7] - S[4] * S[6]);
    float id_ = 1.0f / det, Si[9];
    Si[0] = (S[4] * S[8] - S[5] * S[7]) * id_; Si[1] = (S[2] * S[7] - S[1] * S[8]) * id_; Si[2] = (S[1] * S[5] - S[2] * S[4]) * id_;
    Si[3] = (S[5] * S[6] - S[3] * S[8]) * id_; Si[4] = (S[0] * S[8] - S[2] * S[6]) * id_; Si[5] = (S[2] * S[3] - S[0] * S[5]) * id_;
    Si[6] = (S[3] * S[7] - S[4] * S[6]) * id_; Si[7] = (S[1] * S[6] - S[0] * S[7]) * id_; Si[8] = (S[0] * S[4] - S[1] * S[3]) * id_;
    float K[18];
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 3; ++j) {
      float a = 0.f; for (int m = 0; m < 3; ++m) a += P[i * 6 + m] * Si[m * 3 + j]; K[i * 3 + j] = a; }
    float y[3]; for (int i = 0; i < 3; ++i) y[i] = z[i] - x[i];
    for (int i = 0; i < 6; ++i) x[i] += K[i * 3 + 0] * y[0] + K[i * 3 + 1] * y[1] + K[i * 3 + 2] * y[2];
    float HP[18]; for (int i = 0; i < 3; ++i) for (int j = 0; j < 6; ++j) HP[i * 6 + j] = P[i * 6 + j];
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j)
      P[i * 6 + j] -= K[i * 3 + 0] * HP[0 * 6 + j] + K[i * 3 + 1] * HP[1 * 6 + j] + K[i * 3 + 2] * HP[2 * 6 + j];
  }
  for (int i = 0; i < 6; ++i) p[i] = x[i];
  for (int i = 0; i < 36; ++i) p[NX + i] = P[i];
}
