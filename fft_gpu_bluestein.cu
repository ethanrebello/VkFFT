// Bluestein's algorithm on GPU: FFT for arbitrary N (not just powers of 2).
// Turns a length-N DFT into a length-M cyclic convolution
// (M = next power of 2 >= 2N-1), then reuses our hybrid radix-2 FFT.
//
// nvcc -O2 -std=c++17 -arch=sm_80 fft_gpu_bluestein.cu -o fft_gpu_bluestein

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#define CHECK(x)                                                               \
  {                                                                            \
    cudaError_t err = x;                                                       \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error: %s\n", cudaGetErrorString(err));                     \
      exit(1);                                                                 \
    }                                                                          \
  }

using cuFC = cuFloatComplex;
const float PI = 3.14159265358979f;
const int TPB = 256;
const int CHUNK_MAX = 2048;

// ---------- radix-2 FFT kernels (hybrid shared+global, fwd/inv) ----------

__global__ void bitrev_kernel(cuFC *data, int n, int log_n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int rev = 0, x = i;
  for (int b = 0; b < log_n; b++) {
    rev = (rev << 1) | (x & 1);
    x >>= 1;
  }
  if (i < rev) {
    cuFC tmp = data[i];
    data[i] = data[rev];
    data[rev] = tmp;
  }
}

__global__ void fft_local(cuFC *data, int chunk, bool inv) {
  extern __shared__ cuFC shared[];
  int base = blockIdx.x * chunk;
  int tid = threadIdx.x;

  shared[tid] = data[base + tid];
  shared[tid + chunk / 2] = data[base + tid + chunk / 2];
  __syncthreads();

  for (int len = 2; len <= chunk; len *= 2) {
    int half = len / 2;
    int group = tid / half;
    int k = tid - group * half;
    int lo = group * len + k;
    int hi = lo + half;
    float ang = (inv ? 2.0f : -2.0f) * PI * k / len;
    cuFC w = make_cuFloatComplex(cosf(ang), sinf(ang));
    cuFC u = shared[lo];
    cuFC v = cuCmulf(shared[hi], w);
    shared[lo] = cuCaddf(u, v);
    shared[hi] = cuCsubf(u, v);
    __syncthreads();
  }

  data[base + tid] = shared[tid];
  data[base + tid + chunk / 2] = shared[tid + chunk / 2];
}

__global__ void fft_global_stage(cuFC *data, int n, int len, bool inv) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n / 2)
    return;
  int half = len / 2;
  int group = tid / half;
  int k = tid - group * half;
  int lo = group * len + k;
  int hi = lo + half;
  float ang = (inv ? 2.0f : -2.0f) * PI * k / len;
  cuFC w = make_cuFloatComplex(cosf(ang), sinf(ang));
  cuFC u = data[lo];
  cuFC v = cuCmulf(data[hi], w);
  data[lo] = cuCaddf(u, v);
  data[hi] = cuCsubf(u, v);
}

__global__ void scale_kernel(cuFC *data, int n, float factor) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n)
    return;
  data[tid].x *= factor;
  data[tid].y *= factor;
}

// Hybrid FFT on device (shared + global), forward or inverse.
void fft(cuFC *d_data, int n, bool inv) {
  int log_n = 0;
  while ((1 << log_n) < n)
    log_n++;

  int br_blocks = (n + TPB - 1) / TPB;
  bitrev_kernel<<<br_blocks, TPB>>>(d_data, n, log_n);

  int chunk = (n < CHUNK_MAX) ? n : CHUNK_MAX;
  fft_local<<<n / chunk, chunk / 2, chunk * sizeof(cuFC)>>>(d_data, chunk, inv);

  int gblks = (n / 2 + TPB - 1) / TPB;
  for (int len = chunk * 2; len <= n; len *= 2)
    fft_global_stage<<<gblks, TPB>>>(d_data, n, len, inv);

  if (inv)
    scale_kernel<<<br_blocks, TPB>>>(d_data, n, 1.0f / n);
}

// ---------- Bluestein kernels ----------
//
// X[k] = chirp(k) * (a * b)[k], where
//   chirp(k) = exp(-i * pi * k^2 / N)
//   a[k]     = x[k] * chirp(k)
//   b[k]     = conj(chirp(k)), mirrored at M-k
// Cyclic convolution is computed as: IFFT( FFT(a) .* FFT(b) )

// A[k] = x[k] * chirp(k) for k<n, zero otherwise.
__global__ void build_a(const cuFC *x, cuFC *A, int n, int m) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= m)
    return;
  if (k < n) {
    long long ksq = (long long)k * k % (2LL * n); // keep phase small
    float ang = -PI * ksq / n;
    cuFC chirp = make_cuFloatComplex(cosf(ang), sinf(ang));
    A[k] = cuCmulf(x[k], chirp);
  } else {
    A[k] = make_cuFloatComplex(0.0f, 0.0f);
  }
}

// B[k] = conj(chirp(k)) for k<n or k>m-n, zero otherwise.
__global__ void build_b(cuFC *B, int n, int m) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= m)
    return;
  int j = -1;
  if (k < n)
    j = k;
  else if (k > m - n)
    j = m - k;
  if (j >= 0) {
    long long jsq = (long long)j * j % (2LL * n);
    float ang = PI * jsq / n;
    B[k] = make_cuFloatComplex(cosf(ang), sinf(ang));
  } else {
    B[k] = make_cuFloatComplex(0.0f, 0.0f);
  }
}

__global__ void pointwise_mul(cuFC *A, const cuFC *B, int m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m)
    return;
  A[i] = cuCmulf(A[i], B[i]);
}

// X[k] = A[k] * chirp(k)
__global__ void chirp_post(const cuFC *A, cuFC *X, int n) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n)
    return;
  long long ksq = (long long)k * k % (2LL * n);
  float ang = -PI * ksq / n;
  cuFC chirp = make_cuFloatComplex(cosf(ang), sinf(ang));
  X[k] = cuCmulf(A[k], chirp);
}

// ---------- main ----------

int main() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1, 1);

  // mix of primes (7, 13, 257, 2003) and composites (100, 1000)
  int sizes[] = {7, 13, 100, 257, 1000, 2003};
  for (int si = 0; si < (int)(sizeof(sizes) / sizeof(sizes[0])); si++) {
    int N = sizes[si];
    int M = 1;
    while (M < 2 * N - 1)
      M *= 2;

    // generate input
    std::vector<cuFC> h_x(N);
    for (int i = 0; i < N; i++)
      h_x[i] = make_cuFloatComplex(dist(rng), dist(rng));

    // CPU reference: O(N^2) naive DFT
    std::vector<std::complex<float>> ref(N);
    for (int k = 0; k < N; k++) {
      std::complex<float> sum = 0;
      for (int j = 0; j < N; j++) {
        float ang = -2.0f * PI * k * j / N;
        std::complex<float> xj(h_x[j].x, h_x[j].y);
        sum += xj * std::complex<float>(cosf(ang), sinf(ang));
      }
      ref[k] = sum;
    }

    // allocate device buffers
    cuFC *d_x, *d_X, *d_A, *d_B;
    CHECK(cudaMalloc(&d_x, N * sizeof(cuFC)));
    CHECK(cudaMalloc(&d_X, N * sizeof(cuFC)));
    CHECK(cudaMalloc(&d_A, M * sizeof(cuFC)));
    CHECK(cudaMalloc(&d_B, M * sizeof(cuFC)));
    CHECK(
        cudaMemcpy(d_x, h_x.data(), N * sizeof(cuFC), cudaMemcpyHostToDevice));

    // run Bluestein
    cudaEvent_t t_start, t_end;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);

    int blocks_m = (M + TPB - 1) / TPB;
    int blocks_n = (N + TPB - 1) / TPB;

    build_a<<<blocks_m, TPB>>>(d_x, d_A, N, M);
    build_b<<<blocks_m, TPB>>>(d_B, N, M);

    fft(d_A, M, false);
    fft(d_B, M, false);

    pointwise_mul<<<blocks_m, TPB>>>(d_A, d_B, M);

    fft(d_A, M, true);

    chirp_post<<<blocks_n, TPB>>>(d_A, d_X, N);

    cudaEventRecord(t_end);
    cudaEventSynchronize(t_end);
    float ms = 0;
    cudaEventElapsedTime(&ms, t_start, t_end);

    // check error
    std::vector<cuFC> h_out(N);
    CHECK(cudaMemcpy(h_out.data(), d_X, N * sizeof(cuFC),
                     cudaMemcpyDeviceToHost));

    float err = 0;
    for (int i = 0; i < N; i++) {
      float dr = h_out[i].x - ref[i].real();
      float di = h_out[i].y - ref[i].imag();
      float e = sqrtf(dr * dr + di * di);
      if (e > err)
        err = e;
    }

    printf("N=%4d (M=%5d)  %.3f ms  err %.2e\n", N, M, ms, err);

    cudaFree(d_x);
    cudaFree(d_X);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
  }
  return 0;
}
