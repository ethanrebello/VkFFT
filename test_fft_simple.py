"""Simple test for Triton FFT kernels"""
import os
os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'

import torch
import triton
import triton.language as tl

PI = 3.141592653589793

@triton.jit
def bit_rev_kernel(data_ptr, n, log_n, BLOCK_SIZE: tl.constexpr):
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
def butterfly_kernel(data_ptr, n, span, BLOCK_SIZE: tl.constexpr):
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

# Test with n=8
n = 8
log_n = 3
torch.manual_seed(42)
data = torch.randn(n, dtype=torch.complex64, device='cuda')
print('Input:', data)
ref = torch.fft.fft(data)
print('PyTorch FFT:', ref)

data_real = torch.view_as_real(data.resolve_conj()).contiguous()
print('Before bit-rev:', torch.view_as_complex(data_real))

# Bit-reversal
grid = (triton.cdiv(n, 4),)
bit_rev_kernel[grid](data_real, n, log_n, BLOCK_SIZE=4)
print('After bit-rev:', torch.view_as_complex(data_real))

# Stage 0: span=2
butterfly_kernel[grid](data_real, n, 2, BLOCK_SIZE=4)
print('After stage 0 (span=2):', torch.view_as_complex(data_real))

# Stage 1: span=4
butterfly_kernel[grid](data_real, n, 4, BLOCK_SIZE=4)
print('After stage 1 (span=4):', torch.view_as_complex(data_real))

# Stage 2: span=8
butterfly_kernel[grid](data_real, n, 8, BLOCK_SIZE=4)
print('After stage 2 (span=8):', torch.view_as_complex(data_real))
print('Final:', torch.view_as_complex(data_real))
print('Expected:', ref)
error = torch.abs(torch.view_as_complex(data_real) - ref).max()
print('Max error:', error.item())
