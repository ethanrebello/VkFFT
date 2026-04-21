# GPU FFT Benchmark — TACC Lonestar6 (A100)

This repository contains a benchmarking suite for comparing various CPU and GPU-based Fast Fourier Transform (FFT) implementations on the TACC Lonestar6 cluster.

## What this benchmarks

The suite produces two sets of graphs comparing power-of-2 performance and arbitrary-N performance (using Bluestein's Algorithm).

| Kernel | Source | Method |
|---|---|---|
| `cpu_radix2` | `fft_cpu.cpp` | Iterative Cooley-Tukey (Double) |
| `cpu_radix4` | `fft_cpu.cpp` | Iterative Radix-4 (Double, powers of 4) |
| `gpu_global` | `benchmark_all.cu` | Baseline: Global memory radix-2 stages |
| `gpu_hybrid` | `benchmark_all.cu` | Shared memory stages (up to 2048) + Global memory |
| `gpu_bailey` | `benchmark_all.cu` | Bailey 4-step transposition-based FFT |
| `bluestein` | `benchmark_all.cu` | Cyclic convolution for arbitrary N |

---

## Step-by-step on Lonestar6

### 1. Prepare Workspace

```bash
# Create directory on scratch
mkdir -p $SCRATCH/fft_bench
cd $SCRATCH/fft_bench
```

### 2. Submit Batch Job

Update the allocation account in `run_benchmark.slurm` if necessary.

```bash
sbatch run_benchmark.slurm
squeue -u $USER
```

### 4. Process and Plot Results

Once the job completes, use the provided Python script to generate graphs.

```bash
python3 plot_results.py results/bench_<JOBID>.csv
```

Output files:
- `bench_<JOBID>_graph1.png`: CPU vs GPU Comparison (Power-of-2)
- `bench_<JOBID>_graph2.png`: Bluestein Performance (Arbitrary-N Neighborhood)

---

## Output Formats

### CSV Format
- **Graph 1**: `graph1,kernel,log2N,N,mean_us,std_us,gflops`
- **Graph 2**: `graph2,kernel,ref_logN,N,is_pow2,mean_us,std_us,gflops`

---

## Technical Implementation Details

- **Precision**: GPU implementations use single-precision (`cuFloatComplex`), while CPU implementations use double-precision (`std::complex<double>`) as baselines.
- **Bluestein's Algorithm**: Pads the input to the next power-of-2 $M \ge 2N-1$ and performs a cyclic convolution via FFTs.
- **Bailey 4-step**: Transposes the $N$ elements into an $N_1 \times N_2$ matrix, performs batched FFTs on columns, applies twiddle factors, transposes, and performs batched FFTs on rows.

---

## System Requirements (Lonestar6)

```bash
module load gcc/11.2.0
module load cuda/12.0
```

Note: Binary compatibility is targeted for NVIDIA A100 GPUs (`sm_80`).
