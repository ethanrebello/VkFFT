# Custom CUDA FFT Library & Benchmarking Suite

This repository implements high-performance Fast Fourier Transform (FFT) kernels from scratch using NVIDIA CUDA. It includes a benchmarking suite for performance analysis and a live visualization demo.

## Core Features

- **Hybrid Radix-2 Kernel**: Leverages CUDA shared memory for local FFTs (up to 2048 points) and global memory for larger stages.
- **Bluestein's Algorithm**: Supports arbitrary-N (non-power-of-2) FFTs by transforming the problem into a cyclic convolution.
- **Benchmarking Tools**: Scripts for running large-scale performance comparisons on clusters (tested on TACC Lonestar6).
- **Live Spectrogram**: A real-time audio visualization tool that uses the custom CUDA kernels for signal processing.

## Contents

- `fft_gpu.cu`: Source for the hand-rolled Hybrid Radix-2 kernels.
- `fft_gpu_bluestein.cu`: Source for the hand-rolled Bluestein kernels.
- `benchmark_all.cu`: Unified benchmarking tool for CPU vs GPU comparison.
- `gpu_fft_demo.py`: Real-time audio spectrogram using custom kernels.
- `plot_results.py`: Visualization tool for benchmark data.

## Prerequisites

### Hardware & Drivers
- NVIDIA GPU (RTX 30-series or 40-series recommended; Tesla A100 for benchmarks).
- NVIDIA Driver compatible with your CUDA Toolkit (e.g., Driver 550+ for CUDA 12.x).

### Software
- **CUDA Toolkit**: v11.x or v12.x (ensure `nvcc` is on your PATH).
- **C++ Compiler**: GCC (Linux) or MSVC (Windows).
- **Python 3.8+**: With dependencies installed.

## Quick Start (Demo)

To see the kernels in action with local audio:

```bash
# Install dependencies
pip install -r reqs.txt

# Run the live spectrogram (uses microphone by default)
python gpu_fft_demo.py
```

For more detailed instructions on the benchmarking suite, see [README_benchmark.md](README_benchmark.md).
For more details on the visualization tool, see [README_demo.md](README_demo.md).
