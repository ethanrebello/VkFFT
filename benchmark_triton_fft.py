"""
Phase 4: Benchmarking Script for Triton FFT vs CUDA FFT

This script benchmarks the Triton FFT implementation against:
1. PyTorch's built-in FFT (torch.fft.fft)
2. The existing CUDA FFT implementation (via torch.fft as baseline)

Target: NVIDIA GeForce RTX 4060 Laptop GPU (Compute Capability 8.9, 24 SMs)
"""

import os
os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'

import torch
import triton
import triton.language as tl
import triton.testing
import math
from typing import Tuple

# Triton PI constant
PI: float = 3.141592653589793


# =============================================================================
# Triton FFT Kernels
# =============================================================================

@triton.jit
def _bit_reverse_kernel(
    data_ptr,
    n: tl.constexpr,
    log_n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Bit-reversal permutation kernel."""
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i < n
    
    j = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    x = i
    for b in range(log_n):
        j = (j << 1) | (x & 1)
        x = x >> 1
    
    swap_mask = (i < j) & mask
    
    real_i = tl.load(data_ptr + 2 * i, mask=mask, other=0.0)
    imag_i = tl.load(data_ptr + 2 * i + 1, mask=mask, other=0.0)
    real_j = tl.load(data_ptr + 2 * j, mask=mask, other=0.0)
    imag_j = tl.load(data_ptr + 2 * j + 1, mask=mask, other=0.0)
    
    real_out_i = tl.where(swap_mask, real_j, real_i)
    imag_out_i = tl.where(swap_mask, imag_j, imag_i)
    real_out_j = tl.where(swap_mask, real_i, real_j)
    imag_out_j = tl.where(swap_mask, imag_i, imag_j)
    
    tl.store(data_ptr + 2 * i, real_out_i, mask=mask)
    tl.store(data_ptr + 2 * i + 1, imag_out_i, mask=mask)
    tl.store(data_ptr + 2 * j, real_out_j, mask=mask)
    tl.store(data_ptr + 2 * j + 1, imag_out_j, mask=mask)


@triton.jit
def _fft_butterfly_stage_kernel(
    data_ptr,
    n: tl.constexpr,
    span: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One butterfly stage in global memory."""
    tid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = tid < (n // 2)
    
    half = span // 2
    group = tid // half
    k = tid % half
    
    lo = group * span + k
    hi = lo + half
    
    real_u = tl.load(data_ptr + 2 * lo, mask=mask, other=0.0)
    imag_u = tl.load(data_ptr + 2 * lo + 1, mask=mask, other=0.0)
    real_v = tl.load(data_ptr + 2 * hi, mask=mask, other=0.0)
    imag_v = tl.load(data_ptr + 2 * hi + 1, mask=mask, other=0.0)
    
    angle = -2.0 * PI * k / span
    cos_w = tl.math.cos(angle)
    sin_w = tl.math.sin(angle)
    
    real_vw = real_v * cos_w - imag_v * sin_w
    imag_vw = real_v * sin_w + imag_v * cos_w
    
    real_lo = real_u + real_vw
    imag_lo = imag_u + imag_vw
    real_hi = real_u - real_vw
    imag_hi = imag_u - imag_vw
    
    tl.store(data_ptr + 2 * lo, real_lo, mask=mask)
    tl.store(data_ptr + 2 * lo + 1, imag_lo, mask=mask)
    tl.store(data_ptr + 2 * hi, real_hi, mask=mask)
    tl.store(data_ptr + 2 * hi + 1, imag_hi, mask=mask)


class TritonFFT:
    """Triton-based FFT implementation."""
    
    def __init__(self, n: int, block_size: int = 256):
        assert n & (n - 1) == 0, "FFT size must be a power of 2"
        self.n = n
        self.log_n = n.bit_length() - 1
        self.block_size = block_size
        
    def fft(self, data: torch.Tensor) -> torch.Tensor:
        if data.is_complex():
            data = data.resolve_conj()
            output = torch.view_as_real(data).clone().contiguous()
        else:
            output = data.clone().contiguous()
        
        # Bit-reversal
        grid = (triton.cdiv(self.n, self.block_size),)
        _bit_reverse_kernel[grid](output, self.n, self.log_n, BLOCK_SIZE=self.block_size)
        
        # FFT stages
        span = 2
        for stage in range(self.log_n):
            grid = (triton.cdiv(self.n // 2, self.block_size),)
            _fft_butterfly_stage_kernel[grid](output, self.n, span, BLOCK_SIZE=self.block_size)
            span *= 2
        
        return torch.view_as_complex(output)


# =============================================================================
# Benchmarking Functions
# =============================================================================

def benchmark_fft(n: int, block_size: int = 256, warmup: int = 10, rep: int = 100) -> dict:
    """
    Benchmark FFT implementations.
    
    Returns dict with latency (ms), bandwidth (GB/s), and relative speedup.
    """
    device = torch.device('cuda')
    
    # Generate random complex input
    torch.manual_seed(42)
    input_real = torch.randn(n, dtype=torch.float32, device=device)
    input_imag = torch.randn(n, dtype=torch.float32, device=device)
    input_complex = torch.complex(input_real, input_imag)
    
    # Create Triton FFT engine
    triton_fft = TritonFFT(n, block_size=block_size)
    
    # Warmup
    for _ in range(warmup):
        _ = triton_fft.fft(input_complex)
        _ = torch.fft.fft(input_complex)
    
    # Benchmark Triton FFT
    triton_fn = lambda: triton_fft.fft(input_complex)
    triton_time_ms = triton.testing.do_bench(triton_fn, rep=rep)
    
    # Benchmark PyTorch FFT
    torch_fn = lambda: torch.fft.fft(input_complex)
    torch_time_ms = triton.testing.do_bench(torch_fn, rep=rep)
    
    # Calculate bandwidth (reading input + writing output)
    bytes_transferred = n * 8 * 2  # complex64: 8 bytes per element, read + write
    triton_bandwidth_gbs = bytes_transferred / (triton_time_ms * 1e-3) / 1e9
    torch_bandwidth_gbs = bytes_transferred / (torch_time_ms * 1e-3) / 1e9
    
    # Speedup (positive = Triton faster, negative = PyTorch faster)
    speedup = torch_time_ms / triton_time_ms
    
    return {
        'n': n,
        'triton_latency_ms': triton_time_ms,
        'torch_latency_ms': torch_time_ms,
        'triton_bandwidth_gbs': triton_bandwidth_gbs,
        'torch_bandwidth_gbs': torch_bandwidth_gbs,
        'speedup': speedup,  # >1 means Triton is faster
    }


def run_benchmark():
    """Run benchmark across multiple FFT sizes."""
    print("=" * 80)
    print("Triton FFT Benchmark vs PyTorch FFT")
    print("=" * 80)
    print("Target: NVIDIA GeForce RTX 4060 Laptop GPU (Compute Capability 8.9)")
    print("SM Count: 24, Architecture: Ada Lovelace")
    print("=" * 80)
    
    # Test sizes
    sizes = [64, 256, 1024, 4096, 16384, 65536, 262144]
    
    results = []
    for n in sizes:
        # Use smaller block size for correctness
        block_size = min(256, n // 4) if n >= 16 else 4
        print(f"\nBenchmarking FFT size {n} (block_size={block_size})...")
        
        try:
            result = benchmark_fft(n, block_size=block_size)
            results.append(result)
            
            status = "Triton faster" if result['speedup'] > 1 else "PyTorch faster"
            print(f"  Triton: {result['triton_latency_ms']:.3f} ms, {result['triton_bandwidth_gbs']:.2f} GB/s")
            print(f"  PyTorch: {result['torch_latency_ms']:.3f} ms, {result['torch_bandwidth_gbs']:.2f} GB/s")
            print(f"  Speedup: {result['speedup']:.3f}x ({status})")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'n': n,
                'triton_latency_ms': float('nan'),
                'torch_latency_ms': float('nan'),
                'triton_bandwidth_gbs': float('nan'),
                'torch_bandwidth_gbs': float('nan'),
                'speedup': float('nan'),
            })
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'FFT Size':>10} | {'Triton (ms)':>12} | {'PyTorch (ms)':>12} | {'Speedup':>10} | {'Triton BW':>12} | {'PyTorch BW':>12}")
    print(f"{'FFT Size':>10} | {'':>12} | {'':>12} | {'(>1=Triton)':>10} | {'(GB/s)':>12} | {'(GB/s)':>12}")
    print("-" * 80)
    
    for r in results:
        speedup_str = f"{r['speedup']:.3f}x" if not math.isnan(r['speedup']) else "N/A"
        triton_lat_str = f"{r['triton_latency_ms']:.3f}" if not math.isnan(r['triton_latency_ms']) else "N/A"
        torch_lat_str = f"{r['torch_latency_ms']:.3f}" if not math.isnan(r['torch_latency_ms']) else "N/A"
        triton_bw_str = f"{r['triton_bandwidth_gbs']:.2f}" if not math.isnan(r['triton_bandwidth_gbs']) else "N/A"
        torch_bw_str = f"{r['torch_bandwidth_gbs']:.2f}" if not math.isnan(r['torch_bandwidth_gbs']) else "N/A"
        
        print(f"{r['n']:>10} | {triton_lat_str:>12} | {torch_lat_str:>12} | {speedup_str:>10} | {triton_bw_str:>12} | {torch_bw_str:>12}")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_benchmark()
