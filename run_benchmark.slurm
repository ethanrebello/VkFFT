#!/bin/bash
#------------------------------------------------------------
# SLURM job script — TACC Lonestar6 GPU benchmark
# Submit: sbatch run_benchmark.slurm
#------------------------------------------------------------

#SBATCH -J fft_bench
#SBATCH -o results/bench_%j.out
#SBATCH -e results/bench_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:05:00               
#SBATCH -A CCR26022

module purge
module load gcc/11.2.0
module load cuda/12.0

WORKDIR=$SCRATCH/fft_bench
mkdir -p $WORKDIR/results
cd $WORKDIR

echo "=== Compiling ==="
make clean && make benchmark_all
echo "Compile exit: $?"

echo "=== GPU info ==="
nvidia-smi

echo "=== Benchmark starting (Job $SLURM_JOB_ID) ==="
./benchmark_all 2>&1 | tee results/bench_${SLURM_JOB_ID}.csv

echo "=== Done. Results in results/bench_${SLURM_JOB_ID}.csv ==="
