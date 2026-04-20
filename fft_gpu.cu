// CUDA radix-2 FFT. Uses shared memory for the first log2(chunk) stages, then
// falls back to global memory for the rest. Works for any power-of-2 N.
//
// nvcc -O2 -std=c++17 -arch=sm_75 fft_gpu.cu -o fft_gpu

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
const int CHUNK_MAX = 2048; // 2 * max threads per block

// Host-side bit-reversal (done before uploading to GPU).
void bitrev_host(cuFC *data, int n) {
  int j = 0;
  for (int i = 1; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j) {
      cuFC tmp = data[i];
      data[i] = data[j];
      data[j] = tmp;
    }
  }
}

// Phase 1: each block loads `chunk` elements into shared memory, runs
// stages len=2..chunk there, then writes back. Input must be bit-reversed.
__global__ void fft_local(cuFC *data, int chunk) {
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
    float ang = -2.0f * PI * k / len;
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

// Phase 2: one butterfly stage in global memory (used when len > chunk).
__global__ void fft_global_stage(cuFC *data, int n, int len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n / 2)
    return;
  int half = len / 2;
  int group = tid / half;
  int k = tid - group * half;
  int lo = group * len + k;
  int hi = lo + half;
  float ang = -2.0f * PI * k / len;
  cuFC w = make_cuFloatComplex(cosf(ang), sinf(ang));
  cuFC u = data[lo];
  cuFC v = cuCmulf(data[hi], w);
  data[lo] = cuCaddf(u, v);
  data[hi] = cuCsubf(u, v);
}

// Hybrid FFT: phase 1 (shared) + phase 2 (global).
// Caller must bit-reverse the input before calling.
void fft(cuFC *d_data, int n) {
  int chunk = (n < CHUNK_MAX) ? n : CHUNK_MAX;
  int threads = chunk / 2;
  int blocks = n / chunk;
  fft_local<<<blocks, threads, chunk * sizeof(cuFC)>>>(d_data, chunk);

  int gblks = (n / 2 + TPB - 1) / TPB;
  for (int len = chunk * 2; len <= n; len *= 2)
    fft_global_stage<<<gblks, TPB>>>(d_data, n, len);
}

// CPU reference FFT for correctness check.
void fft_ref(std::vector<std::complex<float>> &data) {
  int n = data.size();
  int j = 0;
  for (int i = 1; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j)
      std::swap(data[i], data[j]);
  }
  for (int len = 2; len <= n; len *= 2) {
    float ang = -2.0f * PI / len;
    std::complex<float> wlen(cosf(ang), sinf(ang));
    int half = len / 2;
    for (int base = 0; base < n; base += len) {
      std::complex<float> w(1, 0);
      for (int k = 0; k < half; k++) {
        std::complex<float> u = data[base + k];
        std::complex<float> v = data[base + k + half] * w;
        data[base + k] = u + v;
        data[base + k + half] = u - v;
        w *= wlen;
      }
    }
  }
}

float max_err(const std::vector<cuFC> &got,
              const std::vector<std::complex<float>> &ref) {
  float e = 0;
  for (size_t i = 0; i < got.size(); i++) {
    float dr = got[i].x - ref[i].real();
    float di = got[i].y - ref[i].imag();
    e = std::max(e, std::sqrt(dr * dr + di * di));
  }
  return e;
}

int main() {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1, 1);

  int sizes[] = {1024, 4096, 16384, 65536, 262144};
  for (int si = 0; si < (int)(sizeof(sizes) / sizeof(sizes[0])); si++) {
    int N = sizes[si];

    std::vector<std::complex<float>> input(N), ref(N);
    for (int i = 0; i < N; i++) {
      float re = dist(rng), im = dist(rng);
      input[i] = {re, im};
      ref[i] = {re, im};
    }
    fft_ref(ref);

    cuFC *d_data;
    CHECK(cudaMalloc(&d_data, N * sizeof(cuFC)));
    cudaEvent_t t_start, t_end;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);

    std::vector<cuFC> buf(N), output(N);
    for (int i = 0; i < N; i++)
      buf[i] = make_cuFloatComplex(input[i].real(), input[i].imag());
    bitrev_host(buf.data(), N);
    CHECK(cudaMemcpy(d_data, buf.data(), N * sizeof(cuFC),
                     cudaMemcpyHostToDevice));

    cudaEventRecord(t_start);
    fft(d_data, N);
    cudaEventRecord(t_end);
    cudaEventSynchronize(t_end);
    float ms = 0;
    cudaEventElapsedTime(&ms, t_start, t_end);
    CHECK(cudaMemcpy(output.data(), d_data, N * sizeof(cuFC),
                     cudaMemcpyDeviceToHost));
    float err = max_err(output, ref);

    printf("N=%7d  %.3f ms  err %.2e\n", N, ms, err);

    cudaFree(d_data);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
  }
  return 0;
}
