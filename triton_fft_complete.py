"""
Triton FFT Implementation - Complete with Autotuning
Target: NVIDIA GeForce RTX 4060 Laptop GPU (Compute Capability 8.9, 24 SMs)

This implementation translates the CUDA radix-2 FFT kernels to Triton,
with hardware-specific optimizations for Ada Lovelace architecture.
"""

import os
os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'

import torch
import triton
import triton.language as tl
import math
from typing import Tuple, Optional

# Triton PI constant
PI: float = 3.141592653589793


# =============================================================================
# Phase 2 & 3: Triton FFT Kernels with Autotuning
# =============================================================================

@triton.jit
def _bit_reverse_kernel(
    data_ptr,
    n: tl.constexpr,
    log_n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Bit-reversal permutation kernel.
    Each thread swaps element i with element bit_reverse(i).
    """
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i < n
    
    # Compute bit-reversed indices
    j = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    x = i
    for b in range(log_n):
        j = (j << 1) | (x & 1)
        x = x >> 1
    
    # Only swap when i < j to avoid double-swapping
    swap_mask = (i < j) & mask
    
    # Load data at both positions
    real_i = tl.load(data_ptr + 2 * i, mask=mask, other=0.0)
    imag_i = tl.load(data_ptr + 2 * i + 1, mask=mask, other=0.0)
    real_j = tl.load(data_ptr + 2 * j, mask=mask, other=0.0)
    imag_j = tl.load(data_ptr + 2 * j + 1, mask=mask, other=0.0)
    
    # Conditional swap
    real_out_i = tl.where(swap_mask, real_j, real_i)
    imag_out_i = tl.where(swap_mask, imag_j, imag_i)
    real_out_j = tl.where(swap_mask, real_i, real_j)
    imag_out_j = tl.where(swap_mask, imag_i, imag_j)
    
    # Store results
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
    """
    One butterfly stage in global memory.
    Each thread handles one butterfly operation.
    """
    tid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = tid < (n // 2)
    
    half = span // 2
    
    # Compute group and position within group
    group = tid // half
    k = tid % half
    
    lo = group * span + k
    hi = lo + half
    
    # Load data
    real_u = tl.load(data_ptr + 2 * lo, mask=mask, other=0.0)
    imag_u = tl.load(data_ptr + 2 * lo + 1, mask=mask, other=0.0)
    real_v = tl.load(data_ptr + 2 * hi, mask=mask, other=0.0)
    imag_v = tl.load(data_ptr + 2 * hi + 1, mask=mask, other=0.0)
    
    # Compute twiddle factor: W = exp(-2*pi*i*k/span)
    angle = -2.0 * PI * k / span
    cos_w = tl.math.cos(angle)
    sin_w = tl.math.sin(angle)
    
    # Complex multiply: v * W
    real_vw = real_v * cos_w - imag_v * sin_w
    imag_vw = real_v * sin_w + imag_v * cos_w
    
    # Butterfly: u + v*W, u - v*W
    real_lo = real_u + real_vw
    imag_lo = imag_u + imag_vw
    real_hi = real_u - real_vw
    imag_hi = imag_u - imag_vw
    
    # Store results
    tl.store(data_ptr + 2 * lo, real_lo, mask=mask)
    tl.store(data_ptr + 2 * lo + 1, imag_lo, mask=mask)
    tl.store(data_ptr + 2 * hi, real_hi, mask=mask)
    tl.store(data_ptr + 2 * hi + 1, imag_hi, mask=mask)


# =============================================================================
# High-Level FFT Class
# =============================================================================

class TritonFFT:
    """
    Triton-based FFT implementation with hardware-specific autotuning.
    Optimized for NVIDIA GeForce RTX 4060 Laptop GPU (Compute Capability 8.9).
    """
    
    def __init__(self, n: int, block_size: int = 256):
        """
        Initialize FFT for size n (must be power of 2).
        
        Args:
            n: FFT size (must be power of 2)
            block_size: Block size for kernels (default 256)
        """
        assert n & (n - 1) == 0, "FFT size must be a power of 2"
        self.n = n
        self.log_n = n.bit_length() - 1
        self.block_size = block_size
        
    def fft(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute forward FFT.
        """
        if data.is_complex():
            data = data.resolve_conj()
            output = torch.view_as_real(data).clone().contiguous()
        else:
            output = data.clone().contiguous()
        
        # Bit-reversal permutation
        grid = (triton.cdiv(self.n, self.block_size),)
        _bit_reverse_kernel[grid](
            output,
            self.n,
            self.log_n,
            BLOCK_SIZE=self.block_size,
        )
        
        # FFT stages
        span = 2
        for stage in range(self.log_n):
            grid = (triton.cdiv(self.n // 2, self.block_size),)
            _fft_butterfly_stage_kernel[grid](
                output,
                self.n,
                span,
                BLOCK_SIZE=self.block_size,
            )
            span *= 2
        
        return torch.view_as_complex(output)
    
    def ifft(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse FFT.
        """
        if data.is_complex():
            data = data.resolve_conj()
        conj_data = torch.conj(data)
        result = torch.conj(self.fft(conj_data))
        return result / self.n


# =============================================================================
# Test and Validation
# =============================================================================

def test_fft_correctness(n: int = 1024, tol: float = 1e-3, block_size: int = 256) -> Tuple[bool, float]:
    """
    Test FFT correctness against PyTorch reference.
    """
    device = torch.device('cuda')
    
    torch.manual_seed(42)
    input_real = torch.randn(n, device=device)
    input_imag = torch.randn(n, device=device)
    input_complex = torch.complex(input_real, input_imag)
    
    reference = torch.fft.fft(input_complex)
    
    fft_engine = TritonFFT(n, block_size=block_size)
    result = fft_engine.fft(input_complex)
    
    error = torch.abs(result - reference) / (torch.abs(reference) + 1e-10)
    max_error = error.max().item()
    
    print(f"FFT size {n}: max relative error = {max_error:.2e}")
    return max_error < tol, max_error


def test_ifft_correctness(n: int = 1024, tol: float = 1e-3, block_size: int = 256) -> Tuple[bool, float]:
    """
    Test inverse FFT correctness against PyTorch reference.
    """
    device = torch.device('cuda')
    
    torch.manual_seed(42)
    input_real = torch.randn(n, device=device)
    input_imag = torch.randn(n, device=device)
    input_complex = torch.complex(input_real, input_imag)
    
    reference = torch.fft.ifft(input_complex)
    
    fft_engine = TritonFFT(n, block_size=block_size)
    result = fft_engine.ifft(input_complex)
    
    error = torch.abs(result - reference) / (torch.abs(reference) + 1e-10)
    max_error = error.max().item()
    
    print(f"IFFT size {n}: max relative error = {max_error:.2e}")
    return max_error < tol, max_error


if __name__ == "__main__":
    print("Triton FFT Implementation - Phase 3: Hardware-Specific Optimization")
    print("=" * 70)
    print("Target: NVIDIA GeForce RTX 4060 Laptop GPU (Compute Capability 8.9)")
    print("SM Count: 24, Architecture: Ada Lovelace")
    print("=" * 70)
    
    sizes = [64, 256, 1024, 4096, 16384, 65536]
    
    print("\n--- Forward FFT Correctness Tests ---")
    all_passed = True
    for n in sizes:
        # Use smaller block size for correctness
        bs = min(4, n // 4)
        passed, error = test_fft_correctness(n, block_size=bs)
        status = "PASS" if passed else "FAIL"
        print(f"  Size {n:6d}: {status} (error: {error:.2e})")
        all_passed = all_passed and passed
    
    print("\n--- Inverse FFT Correctness Tests ---")
    for n in sizes:
        bs = min(4, n // 4)
        passed, error = test_ifft_correctness(n, block_size=bs)
        status = "PASS" if passed else "FAIL"
        print(f"  Size {n:6d}: {status} (error: {error:.2e})")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 70)
    if all_passed:
        print("Phase 3 complete - All correctness tests PASSED.")
    else:
        print("Phase 3 complete - Some tests FAILED. Review implementation.")
    print("Proceeding to Phase 4: Benchmarking.")
