// benchmark_all.cu — Comprehensive FFT Benchmarking Suite
// 
// Benchmarks:
//   - Graph 1: CPU radix-2, CPU radix-4, GPU global, GPU hybrid, GPU Bailey.
//              Sweeps power-of-2 sizes from 2^1 to 2^22.
//   - Graph 2: Bluestein (arbitrary-N) vs GPU global baseline (pow2 only).
//              Sweeps neighborhood N±1, N±3 around each power of 2.
//
// Compilation:
//   nvcc -O2 -std=c++17 -arch=sm_80 benchmark_all.cu -o benchmark_all

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <numeric>
#include <random>
#include <vector>

#define CK(e)                                                                  \
  do {                                                                         \
    cudaError_t _e = (e);                                                      \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d  %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(_e));                                          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

using fc  = cuFloatComplex;
using cpx = std::complex<double>;

static constexpr float  PI_F = 3.14159265358979f;
static constexpr double PI_D = 3.14159265358979323846;
static constexpr int TPB  = 256;
static constexpr int TILE = 32;

// =============================================================================
//  Utility and Helper Functions
// =============================================================================
static int ceilpow2(int v) { int m=1; while(m<v) m<<=1; return m; }
static int ilog2(int n)    { int r=0; while((1<<r)<n) r++; return r; }

static int gpu_reps(int logn) {
  if (logn <= 12) return 500;
  if (logn <= 16) return 200;
  if (logn <= 18) return  50;
  return 20;
}
static int cpu_reps(int logn) {
  if (logn <= 14) return 100;
  if (logn <= 18) return  20;
  if (logn <= 20) return   5;
  return 2;
}

// =============================================================================
//  CPU: Radix-2 Implementation (Double Precision)
// =============================================================================
static void cpu_bitrev(std::vector<cpx> &d) {
  int n=d.size(), j=0;
  for(int i=1;i<n;i++){
    int bit=n>>1;
    for(;j&bit;bit>>=1) j^=bit;
    j^=bit;
    if(i<j) std::swap(d[i],d[j]);
  }
}
static void cpu_fft2(std::vector<cpx> &d) {
  int n=d.size();
  cpu_bitrev(d);
  for(int len=2;len<=n;len<<=1){
    double ang=-2.0*PI_D/len;
    cpx wlen(std::cos(ang),std::sin(ang));
    int half=len/2;
    for(int base=0;base<n;base+=len){
      cpx w=1;
      for(int k=0;k<half;k++){
        cpx u=d[base+k], v=d[base+k+half]*w;
        d[base+k]=u+v; d[base+k+half]=u-v;
        w*=wlen;
      }
    }
  }
}

// =============================================================================
//  CPU: Radix-4 Implementation (Double Precision, N must be power of 4)
// =============================================================================
static void cpu_digitrev4(std::vector<cpx> &d) {
  int n=d.size(), log4=0;
  while((1<<(2*log4))<n) log4++;
  for(int i=0;i<n;i++){
    int src=i,rev=0;
    for(int k=0;k<log4;k++){rev=(rev<<2)|(src&3);src>>=2;}
    if(i<rev) std::swap(d[i],d[rev]);
  }
}
static void cpu_fft4(std::vector<cpx> &d) {
  int n=d.size();
  cpu_digitrev4(d);
  for(int len=4;len<=n;len<<=2){
    int q=len/4;
    double ang=-2.0*PI_D/len;
    cpx w1(std::cos(ang),std::sin(ang));
    cpx w2(std::cos(2*ang),std::sin(2*ang));
    cpx w3(std::cos(3*ang),std::sin(3*ang));
    for(int base=0;base<n;base+=len){
      cpx wa=1,wb=1,wc=1;
      for(int k=0;k<q;k++){
        cpx a0=d[base+k],      a1=d[base+k+q]*wa;
        cpx a2=d[base+k+2*q]*wb, a3=d[base+k+3*q]*wc;
        cpx s02=a0+a2, d02=a0-a2, s13=a1+a3, d13=a1-a3;
        cpx d13r=cpx(d13.imag(),-d13.real());
        d[base+k]=s02+s13; d[base+k+q]=d02+d13r;
        d[base+k+2*q]=s02-s13; d[base+k+3*q]=d02-d13r;
        wa*=w1; wb*=w2; wc*=w3;
      }
    }
  }
}

// =============================================================================
//  GPU: Global Memory Radix-2 (Baseline)
// =============================================================================
__global__ void gpu_stage_global(fc *data, int n, int len) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  if(tid>=n/2) return;
  int half=len>>1, group=tid/half, k=tid-group*half;
  int lo=group*len+k, hi=lo+half;
  float ang=-2.f*PI_F*k/len;
  fc w=make_cuFloatComplex(cosf(ang),sinf(ang));
  fc u=data[lo], v=cuCmulf(data[hi],w);
  data[lo]=cuCaddf(u,v); data[hi]=cuCsubf(u,v);
}
static void gpu_bitrev_host(fc *d, int n) {
  int j=0;
  for(int i=1;i<n;i++){
    int bit=n>>1;
    for(;j&bit;bit>>=1) j^=bit;
    j^=bit;
    if(i<j){fc t=d[i];d[i]=d[j];d[j]=t;}
  }
}
static void fft_global_launch(fc *d, int n) {
  int blocks=(n/2+TPB-1)/TPB;
  for(int len=2;len<=n;len<<=1)
    gpu_stage_global<<<blocks,TPB>>>(d,n,len);
}

// =============================================================================
//  GPU: Hybrid FFT (Shared + Global Memory)
// =============================================================================
__host__ __device__ int bit_reverse_adv(int x, int bits) {
  int r=0;
  for(int i=0;i<bits;i++){r=(r<<1)|(x&1);x>>=1;}
  return r;
}
__global__ void bitrev_kernel_adv(fc *data, int n, int log_n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  int j=bit_reverse_adv(i,log_n);
  if(i<j){fc t=data[i];data[i]=data[j];data[j]=t;}
}
__global__ void stage_kernel_adv(fc *data, int n, int len) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  if(tid>=n/2) return;
  int half=len/2, group=tid/half, k=tid-group*half;
  int lo=group*len+k, hi=lo+half;
  float ang=-2.f*PI_F*k/len;
  fc w=make_cuFloatComplex(cosf(ang),sinf(ang));
  fc u=data[lo], v=cuCmulf(data[hi],w);
  data[lo]=cuCaddf(u,v); data[hi]=cuCsubf(u,v);
}
__global__ void local_fft_kernel_adv(fc *data, int chunk) {
  extern __shared__ fc shared[];
  int base=blockIdx.x*chunk, tid=threadIdx.x;
  shared[tid]=data[base+tid];
  shared[tid+chunk/2]=data[base+tid+chunk/2];
  __syncthreads();
  for(int len=2;len<=chunk;len<<=1){
    int half=len/2, group=tid/half, k=tid-group*half;
    int lo=group*len+k, hi=lo+half;
    float ang=-2.f*PI_F*k/len;
    fc w=make_cuFloatComplex(cosf(ang),sinf(ang));
    fc u=shared[lo], v=cuCmulf(shared[hi],w);
    shared[lo]=cuCaddf(u,v); shared[hi]=cuCsubf(u,v);
    __syncthreads();
  }
  data[base+tid]=shared[tid];
  data[base+tid+chunk/2]=shared[tid+chunk/2];
}
static void fft_hybrid_launch(fc *d, int n) {
  int log_n=ilog2(n);
  bitrev_kernel_adv<<<(n+TPB-1)/TPB,TPB>>>(d,n,log_n);
  int chunk=(n<2048)?n:2048;
  local_fft_kernel_adv<<<n/chunk,chunk/2,chunk*sizeof(fc)>>>(d,chunk);
  int gblocks=(n/2+TPB-1)/TPB;
  for(int len=chunk*2;len<=n;len<<=1)
    stage_kernel_adv<<<gblocks,TPB>>>(d,n,len);
}

// =============================================================================
//  GPU: Bailey 4-Step FFT (Small N blocks + Twiddles)
// =============================================================================
__global__ void batched_fft_kernel(fc *data, int n_sub, int log_nsub) {
  extern __shared__ fc shared[];
  int row_base=blockIdx.x*n_sub, tid=threadIdx.x;
  int br1=bit_reverse_adv(tid,log_nsub);
  int br2=bit_reverse_adv(tid+n_sub/2,log_nsub);
  shared[br1]=data[row_base+tid];
  shared[br2]=data[row_base+tid+n_sub/2];
  __syncthreads();
  for(int len=2;len<=n_sub;len<<=1){
    int half=len/2, group=tid/half, k=tid-group*half;
    int lo=group*len+k, hi=lo+half;
    float ang=-2.f*PI_F*k/len;
    fc w=make_cuFloatComplex(cosf(ang),sinf(ang));
    fc u=shared[lo], v=cuCmulf(shared[hi],w);
    shared[lo]=cuCaddf(u,v); shared[hi]=cuCsubf(u,v);
    __syncthreads();
  }
  data[row_base+tid]=shared[tid];
  data[row_base+tid+n_sub/2]=shared[tid+n_sub/2];
}
__global__ void transpose_kernel(const fc *src, fc *dst, int rows, int cols) {
  __shared__ fc tile[TILE][TILE+1];
  int x_in=blockIdx.x*TILE+threadIdx.x, y_in=blockIdx.y*TILE+threadIdx.y;
  if(x_in<cols && y_in<rows) tile[threadIdx.y][threadIdx.x]=src[y_in*cols+x_in];
  __syncthreads();
  int x_out=blockIdx.y*TILE+threadIdx.x, y_out=blockIdx.x*TILE+threadIdx.y;
  if(x_out<rows && y_out<cols) dst[y_out*rows+x_out]=tile[threadIdx.x][threadIdx.y];
}
__global__ void twiddle_kernel(fc *data, int rows, int cols, int N) {
  int c=blockIdx.x*blockDim.x+threadIdx.x;
  int r=blockIdx.y*blockDim.y+threadIdx.y;
  if(r>=rows || c>=cols) return;
  long long rc=(long long)r*c%N;
  float ang=-2.f*PI_F*rc/(float)N;
  fc w=make_cuFloatComplex(cosf(ang),sinf(ang));
  data[r*cols+c]=cuCmulf(data[r*cols+c],w);
}
static void fft_bailey_launch(fc *d_data, fc *d_work, int N, int N1, int N2) {
  dim3 tblk(TILE,TILE);
  dim3 g_N1xN2((N2+TILE-1)/TILE,(N1+TILE-1)/TILE);
  dim3 g_N2xN1((N1+TILE-1)/TILE,(N2+TILE-1)/TILE);
  transpose_kernel<<<g_N1xN2,tblk>>>(d_data,d_work,N1,N2);
  batched_fft_kernel<<<N2,N1/2,N1*sizeof(fc)>>>(d_work,N1,ilog2(N1));
  dim3 tw_blk(16,16);
  dim3 tw_grd((N1+15)/16,(N2+15)/16);
  twiddle_kernel<<<tw_grd,tw_blk>>>(d_work,N2,N1,N);
  transpose_kernel<<<g_N2xN1,tblk>>>(d_work,d_data,N2,N1);
  batched_fft_kernel<<<N1,N2/2,N2*sizeof(fc)>>>(d_data,N2,ilog2(N2));
  transpose_kernel<<<g_N1xN2,tblk>>>(d_data,d_work,N1,N2);
  CK(cudaMemcpyAsync(d_data,d_work,N*sizeof(fc),cudaMemcpyDeviceToDevice));
}

// =============================================================================
//  GPU: Bluestein's Algorithm (Arbitrary N via cyclic convolution)
// =============================================================================
__global__ void bitrev_blue(fc *data, int n, int lg) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  unsigned x=(unsigned)i;
  x=(x>>16)|(x<<16);
  x=((x&0xff00ff00u)>>8)|((x&0x00ff00ffu)<<8);
  x=((x&0xf0f0f0f0u)>>4)|((x&0x0f0f0f0fu)<<4);
  x=((x&0xccccccccu)>>2)|((x&0x33333333u)<<2);
  x=((x&0xaaaaaaaau)>>1)|((x&0x55555555u)<<1);
  int j=x>>(32-lg);
  if(i<j){fc t=data[i];data[i]=data[j];data[j]=t;}
}
__global__ void butterfly_blue(fc *data, int n, int len) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  if(tid>=n/2) return;
  int half=len>>1, group=tid/half, k=tid-group*half;
  int lo=group*len+k, hi=lo+half;
  float ang=-2.f*PI_F*k/len;
  fc w=make_cuFloatComplex(cosf(ang),sinf(ang));
  fc u=data[lo], v=cuCmulf(data[hi],w);
  data[lo]=cuCaddf(u,v); data[hi]=cuCsubf(u,v);
}
__global__ void scale_blue(fc *data, int n, float f) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){data[i].x*=f;data[i].y*=f;}
}
__global__ void zerofill_blue(fc *data, int m) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<m) data[i]=make_cuFloatComplex(0.f,0.f);
}
__global__ void pre_blue(const fc *x, fc *A, int n, float sign) {
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  if(k>=n) return;
  long long ksq=(long long)k*k%(2LL*n);
  float ang=sign*PI_F*ksq/n;
  A[k]=cuCmulf(x[k],make_cuFloatComplex(cosf(ang),sinf(ang)));
}
__global__ void build_b_blue(fc *B, int n, int m, float sign) {
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  if(k>=n) return;
  long long ksq=(long long)k*k%(2LL*n);
  float ang=-sign*PI_F*ksq/n;
  fc chirp=make_cuFloatComplex(cosf(ang),sinf(ang));
  B[k]=chirp;
  if(k) B[m-k]=chirp;
}
__global__ void hadamard_blue(fc *A, const fc *B, int m) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<m) A[i]=cuCmulf(A[i],B[i]);
}
__global__ void post_blue(const fc *A, fc *y, int n, float sign) {
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  if(k>=n) return;
  long long ksq=(long long)k*k%(2LL*n);
  float ang=sign*PI_F*ksq/n;
  y[k]=cuCmulf(A[k],make_cuFloatComplex(cosf(ang),sinf(ang)));
}

// inner radix-2 with normalization (used as the inverse inside Bluestein)
static void fft2_blue_inv(fc *d, int m) {
  int bf=(m+TPB-1)/TPB, bh=(m/2+TPB-1)/TPB;
  bitrev_blue<<<bf,TPB>>>(d,m,ilog2(m));
  for(int len=2;len<=m;len<<=1) butterfly_blue<<<bh,TPB>>>(d,m,len);
  scale_blue<<<bf,TPB>>>(d,m,1.f/m);
}
// forward (no normalization)
static void fft2_blue_fwd(fc *d, int m) {
  int bf=(m+TPB-1)/TPB, bh=(m/2+TPB-1)/TPB;
  bitrev_blue<<<bf,TPB>>>(d,m,ilog2(m));
  for(int len=2;len<=m;len<<=1) butterfly_blue<<<bh,TPB>>>(d,m,len);
}

static void bluestein_launch(const fc *d_x, fc *d_y, fc *d_A, fc *d_B,
                              int n, int m) {
  int bm=(m+TPB-1)/TPB, bn=(n+TPB-1)/TPB;
  float sign=-1.f;
  zerofill_blue<<<bm,TPB>>>(d_A,m);
  zerofill_blue<<<bm,TPB>>>(d_B,m);
  pre_blue<<<bn,TPB>>>(d_x,d_A,n,sign);
  build_b_blue<<<bn,TPB>>>(d_B,n,m,sign);
  fft2_blue_fwd(d_A,m);
  fft2_blue_fwd(d_B,m);
  hadamard_blue<<<bm,TPB>>>(d_A,d_B,m);
  fft2_blue_inv(d_A,m);
  post_blue<<<bn,TPB>>>(d_A,d_y,n,sign);
}

// =============================================================================
//  Benchmarking Framework and Timing Helpers
// =============================================================================
struct Stats { float mean_us, std_us, gflops; };

template<typename Fn>
static Stats time_gpu(Fn fn, int reps, int n) {
  cudaEvent_t t0,t1;
  cudaEventCreate(&t0); cudaEventCreate(&t1);
  for(int i=0;i<std::min(reps/5+1,20);i++) fn();
  CK(cudaDeviceSynchronize());
  std::vector<float> s(reps);
  for(int r=0;r<reps;r++){
    cudaEventRecord(t0); fn(); cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&s[r],t0,t1);
    s[r]*=1000.f;
  }
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  float sum=std::accumulate(s.begin(),s.end(),0.f);
  float mean=sum/reps, var=0;
  for(float v:s) var+=(v-mean)*(v-mean);
  float std_=std::sqrt(var/reps);
  double log2n=std::log2((double)n);
  float gf=(float)(5.0*n*log2n/(mean*1e-6)/1e9);
  return {mean,std_,gf};
}

template<typename Fn>
static Stats time_cpu(Fn fn, int reps, int n) {
  using clk=std::chrono::high_resolution_clock;
  fn(); // warmup
  std::vector<float> s(reps);
  for(int r=0;r<reps;r++){
    auto t0=clk::now(); fn(); auto t1=clk::now();
    s[r]=(float)std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
  }
  float sum=std::accumulate(s.begin(),s.end(),0.f);
  float mean=sum/reps, var=0;
  for(float v:s) var+=(v-mean)*(v-mean);
  float std_=std::sqrt(var/reps);
  double log2n=std::log2((double)n);
  float gf=(float)(5.0*n*log2n/(mean*1e-6)/1e9);
  return {mean,std_,gf};
}

// =============================================================================
//  Main Benchmark Execution Entry Point
// =============================================================================
int main() {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> fdist(-1.f,1.f);

  // print GPU info
  {
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# GPU: %s, %.1f GB\n", p.name, p.totalGlobalMem/1e9);
  }
  printf("# GRAPH 1: power-of-2 sizes, all kernels, log2N = 1..22\n");
  printf("# graph,kernel,log2N,N,mean_us,std_us,gflops\n");
  fflush(stdout);

  // ===========================================================================
  //  GRAPH 1: Power-of-2 Sizes Sweep
  // ===========================================================================
  for(int logn=1; logn<=22; logn++){
    int n=1<<logn;
    int greps=gpu_reps(logn);
    int creps=cpu_reps(logn);

    // CPU radix-2
    {
      std::vector<cpx> x(n);
      for(auto &z:x) z={fdist(rng),fdist(rng)};
      auto fn=[&]{ auto tmp=x; cpu_fft2(tmp); };
      Stats s=time_cpu(fn,creps,n);
      printf("graph1,cpu_radix2,%d,%d,%.2f,%.2f,%.4f\n",
             logn,n,s.mean_us,s.std_us,s.gflops);
      fflush(stdout);
    }

    // CPU radix-4 (only when logn is even, so N is a power of 4)
    if(logn%2==0){
      std::vector<cpx> x(n);
      for(auto &z:x) z={fdist(rng),fdist(rng)};
      auto fn=[&]{ auto tmp=x; cpu_fft4(tmp); };
      Stats s=time_cpu(fn,creps,n);
      printf("graph1,cpu_radix4,%d,%d,%.2f,%.2f,%.4f\n",
             logn,n,s.mean_us,s.std_us,s.gflops);
      fflush(stdout);
    }

    // GPU global
    {
      std::vector<fc> h(n);
      for(auto &z:h) z=make_cuFloatComplex(fdist(rng),fdist(rng));
      gpu_bitrev_host(h.data(),n);
      fc *d; CK(cudaMalloc(&d,n*sizeof(fc)));
      CK(cudaMemcpy(d,h.data(),n*sizeof(fc),cudaMemcpyHostToDevice));
      auto fn=[&]{
        CK(cudaMemcpyAsync(d,h.data(),n*sizeof(fc),cudaMemcpyHostToDevice));
        fft_global_launch(d,n);
      };
      Stats s=time_gpu(fn,greps,n);
      printf("graph1,gpu_global,%d,%d,%.2f,%.2f,%.4f\n",
             logn,n,s.mean_us,s.std_us,s.gflops);
      fflush(stdout);
      cudaFree(d);
    }

    // GPU hybrid
    {
      std::vector<fc> h(n);
      for(auto &z:h) z=make_cuFloatComplex(fdist(rng),fdist(rng));
      fc *d; CK(cudaMalloc(&d,n*sizeof(fc)));
      CK(cudaMemcpy(d,h.data(),n*sizeof(fc),cudaMemcpyHostToDevice));
      auto fn=[&]{
        CK(cudaMemcpyAsync(d,h.data(),n*sizeof(fc),cudaMemcpyHostToDevice));
        fft_hybrid_launch(d,n);
      };
      Stats s=time_gpu(fn,greps,n);
      printf("graph1,gpu_hybrid,%d,%d,%.2f,%.2f,%.4f\n",
             logn,n,s.mean_us,s.std_us,s.gflops);
      fflush(stdout);
      cudaFree(d);
    }

    // GPU Bailey (requires logn >= 2)
    if(logn>=2){
      int p1=(logn+1)/2, p2=logn-p1;
      int N1=1<<p1, N2=1<<p2;
      std::vector<fc> h(n);
      for(auto &z:h) z=make_cuFloatComplex(fdist(rng),fdist(rng));
      fc *d_data,*d_work;
      CK(cudaMalloc(&d_data,n*sizeof(fc)));
      CK(cudaMalloc(&d_work,n*sizeof(fc)));
      CK(cudaMemcpy(d_data,h.data(),n*sizeof(fc),cudaMemcpyHostToDevice));
      auto fn=[&]{
        CK(cudaMemcpyAsync(d_data,h.data(),n*sizeof(fc),cudaMemcpyHostToDevice));
        fft_bailey_launch(d_data,d_work,n,N1,N2);
      };
      Stats s=time_gpu(fn,greps,n);
      printf("graph1,gpu_bailey,%d,%d,%.2f,%.2f,%.4f\n",
             logn,n,s.mean_us,s.std_us,s.gflops);
      fflush(stdout);
      cudaFree(d_data); cudaFree(d_work);
    }
  }

  // ===========================================================================
  //  GRAPH 2: Neighborhood Non-Power-of-2 Comparison
  // ===========================================================================
  printf("\n# GRAPH 2: gpu_global vs bluestein around each power of 2\n");
  printf("# graph,kernel,ref_logN,N,is_pow2,mean_us,std_us,gflops\n");
  fflush(stdout);

  for(int logn=1; logn<=22; logn++){
    int N_pow2=1<<logn;
    int offsets[]={-3,-1,0,1,3};

    for(int off : offsets){
      int n=N_pow2+off;
      if(n<2) continue;
      int greps=gpu_reps(logn);
      bool is_pow2=(off==0);

      // Bluestein — runs for all N
      {
        int m=ceilpow2(2*n-1);
        std::vector<fc> h(n);
        for(auto &z:h) z=make_cuFloatComplex(fdist(rng),fdist(rng));
        fc *d_x,*d_y,*d_A,*d_B;
        CK(cudaMalloc(&d_x,n*sizeof(fc)));
        CK(cudaMalloc(&d_y,n*sizeof(fc)));
        CK(cudaMalloc(&d_A,m*sizeof(fc)));
        CK(cudaMalloc(&d_B,m*sizeof(fc)));
        CK(cudaMemcpy(d_x,h.data(),n*sizeof(fc),cudaMemcpyHostToDevice));
        auto fn=[&]{ bluestein_launch(d_x,d_y,d_A,d_B,n,m); };
        Stats s=time_gpu(fn,greps,n);
        printf("graph2,bluestein,%d,%d,%d,%.2f,%.2f,%.4f\n",
               logn,n,(int)is_pow2,s.mean_us,s.std_us,s.gflops);
        fflush(stdout);
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_A); cudaFree(d_B);
      }

      // gpu_global — only at the power-of-2 point
      if(is_pow2){
        std::vector<fc> h(n);
        for(auto &z:h) z=make_cuFloatComplex(fdist(rng),fdist(rng));
        gpu_bitrev_host(h.data(),n);
        fc *d; CK(cudaMalloc(&d,n*sizeof(fc)));
        CK(cudaMemcpy(d,h.data(),n*sizeof(fc),cudaMemcpyHostToDevice));
        auto fn=[&]{
          CK(cudaMemcpyAsync(d,h.data(),n*sizeof(fc),cudaMemcpyHostToDevice));
          fft_global_launch(d,n);
        };
        Stats s=time_gpu(fn,greps,n);
        printf("graph2,gpu_global,%d,%d,%d,%.2f,%.2f,%.4f\n",
               logn,n,(int)is_pow2,s.mean_us,s.std_us,s.gflops);
        fflush(stdout);
        cudaFree(d);
      }
    }
  }

  printf("# done\n");
  return 0;
}
