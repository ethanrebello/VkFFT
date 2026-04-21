# Makefile for TACC Lonestar6 (A100, sm_80)
# Usage: make          — builds benchmark_all
#        make verify   — builds comparison binaries (CPU, GPU, Bluestein)
#        make clean

NVCC     = nvcc
NVFLAGS  = -O2 -std=c++17 -arch=sm_80 -Xcompiler -Wall
CXX      = g++
CXXFLAGS = -O2 -std=c++17 -Wall

all: benchmark_all

benchmark_all: benchmark_all.cu
	$(NVCC) $(NVFLAGS) $< -o $@

# Comparison and self-test binaries
verify: fft_gpu fft_gpu_bluestein fft_cpu

fft_gpu: fft_gpu.cu
	$(NVCC) $(NVFLAGS) $< -o $@

fft_gpu_bluestein: fft_gpu_bluestein.cu
	$(NVCC) $(NVFLAGS) $< -o $@

fft_cpu: fft_cpu.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f benchmark_all fft_gpu fft_gpu_bluestein fft_cpu

.PHONY: all verify clean
