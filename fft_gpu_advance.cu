// Bailey's 4-step FFT on GPU. Works up to N = 2^20.
// Benchmarks against the simpler hybrid (shared+global) baseline.
//
// nvcc -O2 -std=c++17 -arch=sm_75 fft_gpu_advance.cu -o fft_gpu_advance

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
const int TILE = 32;
const int TPB = 256;

int log2i(int n) {
  int r = 0;
  while ((1 << r) < n)
    r++;
  return r;
}

// Reverse the lower `bits` bits of x (loop version — simple but clear).
__host__ __device__ int bit_reverse(int x, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) {
    r = (r << 1) | (x & 1);
    x >>= 1;
  }
  return r;
}

// Permute the whole length-N array to bit-reversed order.
__global__ void bitrev_kernel(cuFC *data, int n, int log_n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int j = bit_reverse(i, log_n);
  if (i < j) {
    cuFC tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
  }
}

// One butterfly stage, in global memory. Each thread = one butterfly.
__global__ void stage_kernel(cuFC *data, int n, int len) {
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

// Run one chunk's worth of FFT stages entirely in shared memory.
// One block per chunk, each block handles `chunk` consecutive elements.
__global__ void local_fft_kernel(cuFC *data, int chunk) {
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

// ---------- hybrid baseline ----------

void fft_hybrid(cuFC *d, int n) {
  int log_n = log2i(n);
  int br_blocks = (n + TPB - 1) / TPB;
  bitrev_kernel<<<br_blocks, TPB>>>(d, n, log_n);

  int chunk = (n < 2048) ? n : 2048;
  int threads = chunk / 2;
  int blocks = n / chunk;
  local_fft_kernel<<<blocks, threads, chunk * sizeof(cuFC)>>>(d, chunk);

  int gblocks = (n / 2 + TPB - 1) / TPB;
  for (int len = chunk * 2; len <= n; len *= 2) {
    stage_kernel<<<gblocks, TPB>>>(d, n, len);
  }
}

// ---------- Bailey's 4-step kernels ----------

// Batched FFT: each block does one length-n_sub FFT in shared memory.
__global__ void batched_fft_kernel(cuFC *data, int n_sub, int log_nsub) {
  extern __shared__ cuFC shared[];
  int row_base = blockIdx.x * n_sub;
  int tid = threadIdx.x;

  // load with bit-reversal, so later stages can just go forward
  int br1 = bit_reverse(tid, log_nsub);
  int br2 = bit_reverse(tid + n_sub / 2, log_nsub);
  shared[br1] = data[row_base + tid];
  shared[br2] = data[row_base + tid + n_sub / 2];
  __syncthreads();

  for (int len = 2; len <= n_sub; len *= 2) {
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

  data[row_base + tid] = shared[tid];
  data[row_base + tid + n_sub / 2] = shared[tid + n_sub / 2];
}

// Tiled matrix transpose. +1 padding avoids shared-memory bank conflicts.
__global__ void transpose_kernel(const cuFC *src, cuFC *dst, int rows,
                                 int cols) {
  __shared__ cuFC tile[TILE][TILE + 1];

  int x_in = blockIdx.x * TILE + threadIdx.x;
  int y_in = blockIdx.y * TILE + threadIdx.y;
  if (x_in < cols && y_in < rows)
    tile[threadIdx.y][threadIdx.x] = src[y_in * cols + x_in];
  __syncthreads();

  int x_out = blockIdx.y * TILE + threadIdx.x;
  int y_out = blockIdx.x * TILE + threadIdx.y;
  if (x_out < rows && y_out < cols)
    dst[y_out * rows + x_out] = tile[threadIdx.x][threadIdx.y];
}

// Element-wise twiddle: data[r][c] *= exp(-2*pi*i * r*c / N)
__global__ void twiddle_kernel(cuFC *data, int rows, int cols, int N) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= rows || c >= cols)
    return;
  long long rc = (long long)r * c % N;
  float ang = -2.0f * PI * rc / (float)N;
  cuFC w = make_cuFloatComplex(cosf(ang), sinf(ang));
  int idx = r * cols + c;
  data[idx] = cuCmulf(data[idx], w);
}

void fft_bailey(cuFC *d_data, cuFC *d_work, int N, int N1, int N2) {
  dim3 tblk(TILE, TILE);
  dim3 g_N1xN2((N2 + TILE - 1) / TILE, (N1 + TILE - 1) / TILE);
  dim3 g_N2xN1((N1 + TILE - 1) / TILE, (N2 + TILE - 1) / TILE);

  // 1. transpose N1 x N2 -> N2 x N1
  transpose_kernel<<<g_N1xN2, tblk>>>(d_data, d_work, N1, N2);

  // 2. batched FFT of length N1 (N2 rows)
  batched_fft_kernel<<<N2, N1 / 2, N1 * sizeof(cuFC)>>>(d_work, N1, log2i(N1));

  // 3. twiddle multiply
  dim3 tw_blk(16, 16);
  dim3 tw_grd((N1 + 15) / 16, (N2 + 15) / 16);
  twiddle_kernel<<<tw_grd, tw_blk>>>(d_work, N2, N1, N);

  // 4. transpose back
  transpose_kernel<<<g_N2xN1, tblk>>>(d_work, d_data, N2, N1);

  // 5. batched FFT of length N2 (N1 rows)
  batched_fft_kernel<<<N1, N2 / 2, N2 * sizeof(cuFC)>>>(d_data, N2, log2i(N2));

  // 6. final transpose to put output in natural order
  transpose_kernel<<<g_N1xN2, tblk>>>(d_data, d_work, N1, N2);
  CHECK(cudaMemcpy(d_data, d_work, N * sizeof(cuFC), cudaMemcpyDeviceToDevice));
}

// ---------- CPU reference (for correctness only) ----------

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
        auto u = data[base + k];
        auto v = data[base + k + half] * w;
        data[base + k] = u + v;
        data[base + k + half] = u - v;
        w *= wlen;
      }
    }
  }
}

int main() {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1, 1);

  const int REPEAT = 20;

  printf("%-6s %-8s %-10s %11s %11s   %7s\n", "log2N", "N", "N1xN2",
         "hybrid(ms)", "bailey(ms)", "h/b");
  printf("-----------------------------------------------------------------\n");

  int sizes[] = {21, 22};
  for (int si = 0; si < (int)(sizeof(sizes) / sizeof(sizes[0])); si++) {
    int log_N = sizes[si];
    int N = 1 << log_N;

    // balanced decomposition: N1 >= N2, both powers of 2
    int p1 = (log_N + 1) / 2;
    int p2 = log_N - p1;
    int N1 = 1 << p1;
    int N2 = 1 << p2;

    std::vector<std::complex<float>> h_in(N);
    for (int i = 0; i < N; i++) {
      float re = dist(rng), im = dist(rng);
      h_in[i] = {re, im};
    }

    cuFC *d_data, *d_work;
    CHECK(cudaMalloc(&d_data, N * sizeof(cuFC)));
    CHECK(cudaMalloc(&d_work, N * sizeof(cuFC)));

    std::vector<cuFC> h_buf(N);
    for (int i = 0; i < N; i++)
      h_buf[i] = make_cuFloatComplex(h_in[i].real(), h_in[i].imag());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // --- time hybrid ---
    CHECK(cudaMemcpy(d_data, h_buf.data(), N * sizeof(cuFC),
                     cudaMemcpyHostToDevice));
    fft_hybrid(d_data, N);
    cudaDeviceSynchronize();

    float ms_h = 0;
    for (int r = 0; r < REPEAT; r++) {
      CHECK(cudaMemcpy(d_data, h_buf.data(), N * sizeof(cuFC),
                       cudaMemcpyHostToDevice));
      cudaEventRecord(t0);
      fft_hybrid(d_data, N);
      cudaEventRecord(t1);
      cudaEventSynchronize(t1);
      float tmp;
      cudaEventElapsedTime(&tmp, t0, t1);
      ms_h += tmp;
    }
    ms_h /= REPEAT;

    // --- time Bailey ---
    CHECK(cudaMemcpy(d_data, h_buf.data(), N * sizeof(cuFC),
                     cudaMemcpyHostToDevice));
    fft_bailey(d_data, d_work, N, N1, N2);
    cudaDeviceSynchronize();

    float ms_b = 0;
    for (int r = 0; r < REPEAT; r++) {
      CHECK(cudaMemcpy(d_data, h_buf.data(), N * sizeof(cuFC),
                       cudaMemcpyHostToDevice));
      cudaEventRecord(t0);
      fft_bailey(d_data, d_work, N, N1, N2);
      cudaEventRecord(t1);
      cudaEventSynchronize(t1);
      float tmp;
      cudaEventElapsedTime(&tmp, t0, t1);
      ms_b += tmp;
    }
    ms_b /= REPEAT;

    printf("%-6d %-8d %4dx%-5d %11.3f %11.3f   %7.2f\n", log_N, N, N1, N2, ms_h,
           ms_b, ms_h / ms_b);

    cudaFree(d_data);
    cudaFree(d_work);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
  }

  return 0;
}
