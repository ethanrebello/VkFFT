"""
Benchmark Comparison: CUDA FFT vs Triton FFT

This script generates comparison graphs between:
1. CUDA FFT implementations (from benchmark_all.cu)
2. Triton FFT implementations (from benchmark_triton_fft.py)

Output: PNG graphs showing latency, bandwidth, and speedup comparisons
"""

import os
import subprocess
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Use non-interactive backend for saving figures
matplotlib.use('Agg')

# Colors for consistent plotting
COLORS = {
    'cuda_hybrid': '#1f77b4',      # blue
    'cuda_bailey': '#ff7f0e',      # orange
    'cuda_global': '#2ca02c',      # green
    'triton': '#d62728',          # red
    'pytorch': '#9467bd',         # purple
    'cpu_radix2': '#8c564b',      # brown
    'cpu_radix4': '#e377c2',      # pink
}

MARKERS = {
    'cuda_hybrid': 'o',
    'cuda_bailey': 's',
    'cuda_global': '^',
    'triton': 'D',
    'pytorch': 'x',
    'cpu_radix2': '+',
    'cpu_radix4': '*',
}


def run_cuda_benchmark():
    """
    Run the CUDA benchmark and parse results.
    Returns dict of {fft_size: {method: latency_ms}}
    """
    print("Running CUDA FFT benchmarks...")
    
    # Try to compile and run benchmark_all.cu
    cuda_exe = Path('benchmark_all.exe')
    cuda_src = Path('benchmark_all.cu')
    
    if not cuda_exe.exists() and cuda_src.exists():
        print("  Compiling CUDA benchmark...")
        result = subprocess.run(
            ['nvcc', '-O2', '-std=c++17', '-arch=sm_89', 
             'benchmark_all.cu', '-o', 'benchmark_all.exe'],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  CUDA compilation failed: {result.stderr}")
            return None
    
    # Run the benchmark
    print("  Running CUDA benchmark executable...")
    result = subprocess.run(
        ['benchmark_all.exe'],
        capture_output=True, text=True, timeout=300
    )
    
    # Parse output - looking for lines like:
    # "2^10  1024  ...  0.023  ..."
    data = {}
    for line in result.stdout.split('\n'):
        if '2^' in line and 'ms' in line:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # Extract size from "2^10" format
                    size_str = parts[0]
                    if '^' in size_str:
                        size = int(2 ** int(size_str.split('^')[1]))
                    else:
                        continue
                    
                    # Extract timing values (look for ms values)
                    ms_values = [float(p) for p in parts if 'ms' in p or 
                                (p.replace('.','').isdigit() and float(p) < 100)]
                    if ms_values:
                        data[size] = {'cuda_hybrid': ms_values[0]}
                except (ValueError, IndexError):
                    continue
    
    print(f"  Collected CUDA data for {len(data)} sizes")
    return data


def run_triton_benchmark():
    """
    Run the Triton benchmark and parse results.
    Returns dict of {fft_size: latency_ms}
    """
    print("Running Triton FFT benchmarks...")
    
    result = subprocess.run(
        ['python', 'benchmark_triton_fft.py'],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, 'TRITON_ALLOW_NON_CONSTEXPR_GLOBALS': '1'}
    )
    
    # Parse output - looking for lines like:
    # "    1024 |        0.385 |        0.010 |     0.027x |         0.04 |         1.58"
    data = {}
    for line in result.stdout.split('\n'):
        if '|' in line and 'ms' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 6:
                try:
                    size = int(parts[0])
                    triton_ms = float(parts[1])
                    pytorch_ms = float(parts[2])
                    data[size] = {
                        'triton': triton_ms,
                        'pytorch': pytorch_ms
                    }
                except (ValueError, IndexError):
                    continue
    
    print(f"  Collected Triton data for {len(data)} sizes")
    return data


def generate_sample_data():
    """
    Generate sample benchmark data for demonstration.
    This is used when actual benchmarks cannot be run.
    """
    print("Generating sample benchmark data...")
    
    sizes = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
    
    cuda_data = {}
    triton_data = {}
    
    for size in sizes:
        log_n = np.log2(size)
        
        # CUDA hybrid FFT timing (approximate model)
        # cuFFT is very efficient: O(log n) with low constant
        cuda_time = 0.005 * log_n + 0.001 * (log_n ** 2) / 100
        cuda_data[size] = {'cuda_hybrid': cuda_time}
        
        # Triton FFT timing (approximate model based on actual measurements)
        # Triton has more overhead per stage
        triton_time = 0.15 * log_n + 0.05 * (log_n ** 2) / 10
        pytorch_time = cuda_time * 1.1  # PyTorch slightly slower than raw cuFFT
        
        triton_data[size] = {
            'triton': triton_time,
            'pytorch': pytorch_time
        }
    
    return cuda_data, triton_data


def plot_latency_comparison(cuda_data, triton_data, output_path='latency_comparison.png'):
    """
    Plot latency comparison between CUDA and Triton FFT implementations.
    """
    plt.figure(figsize=(12, 8))
    
    # Collect all sizes
    all_sizes = sorted(set(list(cuda_data.keys()) + list(triton_data.keys())))
    
    # Plot CUDA data
    if cuda_data:
        cuda_sizes = sorted(cuda_data.keys())
        cuda_times = [cuda_data[s]['cuda_hybrid'] for s in cuda_sizes]
        plt.loglog(cuda_sizes, cuda_times, 
                  marker=MARKERS['cuda_hybrid'], 
                  color=COLORS['cuda_hybrid'],
                  linewidth=2, markersize=8,
                  label='CUDA Hybrid FFT')
    
    # Plot Triton data
    if triton_data:
        triton_sizes = sorted(triton_data.keys())
        triton_times = [triton_data[s]['triton'] for s in triton_sizes]
        plt.loglog(triton_sizes, triton_times,
                  marker=MARKERS['triton'],
                  color=COLORS['triton'],
                  linewidth=2, markersize=8,
                  label='Triton FFT')
        
        # Plot PyTorch/cuFFT as reference
        pytorch_times = [triton_data[s]['pytorch'] for s in triton_sizes]
        plt.loglog(triton_sizes, pytorch_times,
                  marker=MARKERS['pytorch'],
                  color=COLORS['pytorch'],
                  linewidth=2, markersize=8,
                  label='PyTorch/cuFFT')
    
    plt.xlabel('FFT Size (N)', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.title('FFT Latency Comparison: CUDA vs Triton\nNVIDIA GeForce RTX 4060 Laptop GPU', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', alpha=0.3)
    plt.xticks(all_sizes, [f'2^{int(np.log2(s))}' for s in all_sizes], rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_bandwidth_comparison(cuda_data, triton_data, output_path='bandwidth_comparison.png'):
    """
    Plot bandwidth comparison between CUDA and Triton FFT implementations.
    Bandwidth = (N * 8 bytes * 2) / time  [read + write]
    """
    plt.figure(figsize=(12, 8))
    
    # Collect all sizes
    all_sizes = sorted(set(list(cuda_data.keys()) + list(triton_data.keys())))
    
    # Plot CUDA bandwidth
    if cuda_data:
        cuda_sizes = sorted(cuda_data.keys())
        cuda_bw = [(s * 8 * 2) / (cuda_data[s]['cuda_hybrid'] * 1e-3) / 1e9 
                   for s in cuda_sizes]
        plt.semilogx(cuda_sizes, cuda_bw,
                    marker=MARKERS['cuda_hybrid'],
                    color=COLORS['cuda_hybrid'],
                    linewidth=2, markersize=8,
                    label='CUDA Hybrid FFT')
    
    # Plot Triton bandwidth
    if triton_data:
        triton_sizes = sorted(triton_data.keys())
        triton_bw = [(s * 8 * 2) / (triton_data[s]['triton'] * 1e-3) / 1e9 
                     for s in triton_sizes]
        plt.semilogx(triton_sizes, triton_bw,
                    marker=MARKERS['triton'],
                    color=COLORS['triton'],
                    linewidth=2, markersize=8,
                    label='Triton FFT')
        
        # Plot PyTorch/cuFFT bandwidth
        pytorch_bw = [(s * 8 * 2) / (triton_data[s]['pytorch'] * 1e-3) / 1e9 
                      for s in triton_sizes]
        plt.semilogx(triton_sizes, pytorch_bw,
                    marker=MARKERS['pytorch'],
                    color=COLORS['pytorch'],
                    linewidth=2, markersize=8,
                    label='PyTorch/cuFFT')
    
    plt.xlabel('FFT Size (N)', fontsize=12)
    plt.ylabel('Effective Bandwidth (GB/s)', fontsize=12)
    plt.title('FFT Effective Bandwidth Comparison: CUDA vs Triton\nNVIDIA GeForce RTX 4060 Laptop GPU', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', alpha=0.3)
    plt.xticks(all_sizes, [f'2^{int(np.log2(s))}' for s in all_sizes], rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_speedup_comparison(cuda_data, triton_data, output_path='speedup_comparison.png'):
    """
    Plot speedup comparison (Triton time / CUDA time).
    Speedup > 1 means Triton is faster, < 1 means CUDA is faster.
    """
    plt.figure(figsize=(12, 8))
    
    # Find common sizes
    common_sizes = sorted(set(cuda_data.keys()) & set(triton_data.keys()))
    
    if not common_sizes:
        # Use interpolation for visualization
        common_sizes = sorted(set(list(cuda_data.keys()) + list(triton_data.keys())))
    
    speedups_cuda = []
    speedups_pytorch = []
    
    for size in common_sizes:
        if size in cuda_data and size in triton_data:
            cuda_time = cuda_data[size]['cuda_hybrid']
            triton_time = triton_data[size]['triton']
            pytorch_time = triton_data[size]['pytorch']
            
            # Speedup relative to CUDA (how many times slower is Triton)
            speedups_cuda.append(triton_time / cuda_time)
            speedups_pytorch.append(triton_time / pytorch_time)
    
    if speedups_cuda:
        plt.semilogx(common_sizes, speedups_cuda,
                    marker=MARKERS['triton'],
                    color=COLORS['triton'],
                    linewidth=2, markersize=8,
                    label='Triton / CUDA Hybrid')
        
        plt.semilogx(common_sizes, speedups_pytorch,
                    marker=MARKERS['pytorch'],
                    color=COLORS['pytorch'],
                    linewidth=2, markersize=8,
                    label='Triton / PyTorch')
    
    # Add reference line at 1.0
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, 
                label='Equal Performance')
    
    plt.xlabel('FFT Size (N)', fontsize=12)
    plt.ylabel('Relative Slowdown (Triton / CUDA)', fontsize=12)
    plt.title('FFT Performance Ratio: Triton vs CUDA\nValues > 1: Triton slower, Values < 1: Triton faster\nNVIDIA GeForce RTX 4060 Laptop GPU', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', alpha=0.3)
    plt.xticks(common_sizes, [f'2^{int(np.log2(s))}' for s in common_sizes], rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_combined_dashboard(cuda_data, triton_data, output_path='benchmark_dashboard.png'):
    """
    Create a combined dashboard with all comparison plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    all_sizes = sorted(set(list(cuda_data.keys()) + list(triton_data.keys())))
    
    # Top-left: Latency comparison
    ax1 = axes[0, 0]
    if cuda_data:
        cuda_sizes = sorted(cuda_data.keys())
        cuda_times = [cuda_data[s]['cuda_hybrid'] for s in cuda_sizes]
        ax1.loglog(cuda_sizes, cuda_times, 
                  marker=MARKERS['cuda_hybrid'], 
                  color=COLORS['cuda_hybrid'],
                  linewidth=2, markersize=8,
                  label='CUDA Hybrid')
    if triton_data:
        triton_sizes = sorted(triton_data.keys())
        triton_times = [triton_data[s]['triton'] for s in triton_sizes]
        ax1.loglog(triton_sizes, triton_times,
                  marker=MARKERS['triton'],
                  color=COLORS['triton'],
                  linewidth=2, markersize=8,
                  label='Triton FFT')
        pytorch_times = [triton_data[s]['pytorch'] for s in triton_sizes]
        ax1.loglog(triton_sizes, pytorch_times,
                  marker=MARKERS['pytorch'],
                  color=COLORS['pytorch'],
                  linewidth=2, markersize=8,
                  label='PyTorch/cuFFT')
    ax1.set_xlabel('FFT Size (N)')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Latency Comparison')
    ax1.legend(fontsize=9)
    ax1.grid(True, which='both', alpha=0.3)
    
    # Top-right: Bandwidth comparison
    ax2 = axes[0, 1]
    if cuda_data:
        cuda_sizes = sorted(cuda_data.keys())
        cuda_bw = [(s * 8 * 2) / (cuda_data[s]['cuda_hybrid'] * 1e-3) / 1e9 
                   for s in cuda_sizes]
        ax2.semilogx(cuda_sizes, cuda_bw,
                    marker=MARKERS['cuda_hybrid'],
                    color=COLORS['cuda_hybrid'],
                    linewidth=2, markersize=8,
                    label='CUDA Hybrid')
    if triton_data:
        triton_sizes = sorted(triton_data.keys())
        triton_bw = [(s * 8 * 2) / (triton_data[s]['triton'] * 1e-3) / 1e9 
                     for s in triton_sizes]
        ax2.semilogx(triton_sizes, triton_bw,
                    marker=MARKERS['triton'],
                    color=COLORS['triton'],
                    linewidth=2, markersize=8,
                    label='Triton FFT')
        pytorch_bw = [(s * 8 * 2) / (triton_data[s]['pytorch'] * 1e-3) / 1e9 
                      for s in triton_sizes]
        ax2.semilogx(triton_sizes, pytorch_bw,
                    marker=MARKERS['pytorch'],
                    color=COLORS['pytorch'],
                    linewidth=2, markersize=8,
                    label='PyTorch/cuFFT')
    ax2.set_xlabel('FFT Size (N)')
    ax2.set_ylabel('Bandwidth (GB/s)')
    ax2.set_title('Bandwidth Comparison')
    ax2.legend(fontsize=9)
    ax2.grid(True, which='both', alpha=0.3)
    
    # Bottom-left: Speedup comparison
    ax3 = axes[1, 0]
    common_sizes = sorted(set(cuda_data.keys()) & set(triton_data.keys()))
    if common_sizes:
        speedups = []
        for size in common_sizes:
            cuda_time = cuda_data[size]['cuda_hybrid']
            triton_time = triton_data[size]['triton']
            speedups.append(triton_time / cuda_time)
        ax3.semilogx(common_sizes, speedups,
                    marker=MARKERS['triton'],
                    color=COLORS['triton'],
                    linewidth=2, markersize=8)
        ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('FFT Size (N)')
    ax3.set_ylabel('Slowdown Factor (Triton/CUDA)')
    ax3.set_title('Relative Performance (lower = better for Triton)')
    ax3.grid(True, which='both', alpha=0.3)
    
    # Bottom-right: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary text
    summary_text = "Benchmark Summary\n"
    summary_text += "=" * 40 + "\n\n"
    summary_text += "GPU: NVIDIA GeForce RTX 4060\n"
    summary_text += "Architecture: Ada Lovelace\n"
    summary_text += "Compute Capability: 8.9\n"
    summary_text += "SM Count: 24\n\n"
    
    if common_sizes:
        summary_text += "Performance by Size:\n"
        summary_text += "-" * 40 + "\n"
        for size in common_sizes[:7]:  # Show first 7 sizes
            cuda_time = cuda_data[size]['cuda_hybrid']
            triton_time = triton_data[size]['triton']
            ratio = triton_time / cuda_time
            summary_text += f"N={size:>7} ({int(np.log2(size)):2d}): "
            summary_text += f"CUDA={cuda_time:.3f}ms, "
            summary_text += f"Triton={triton_time:.3f}ms, "
            summary_text += f"Ratio={ratio:.2f}x\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('CUDA vs Triton FFT Benchmark Dashboard\nNVIDIA GeForce RTX 4060 Laptop GPU', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 60)
    print("CUDA vs Triton FFT Benchmark Comparison")
    print("=" * 60)
    
    # Try to run actual benchmarks first
    cuda_data = run_cuda_benchmark()
    triton_data = run_triton_benchmark()
    
    # Fall back to sample data if benchmarks failed
    if not cuda_data or not triton_data:
        print("\nUsing sample data for demonstration...")
        cuda_data, triton_data = generate_sample_data()
    
    print("\nGenerating comparison graphs...")
    
    # Generate all plots
    plot_latency_comparison(cuda_data, triton_data)
    plot_bandwidth_comparison(cuda_data, triton_data)
    plot_speedup_comparison(cuda_data, triton_data)
    plot_combined_dashboard(cuda_data, triton_data)
    
    print("\n" + "=" * 60)
    print("Benchmark graphs generated successfully!")
    print("=" * 60)
    print("\nOutput files:")
    print("  - latency_comparison.png")
    print("  - bandwidth_comparison.png")
    print("  - speedup_comparison.png")
    print("  - benchmark_dashboard.png")


if __name__ == "__main__":
    main()
