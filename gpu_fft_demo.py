#!/usr/bin/env python3
"""
gpu_fft_demo.py  --  Live GPU FFT spectrogram demo
Runs your fft_gpu / fft_gpu_bluestein CUDA kernels via ctypes,
displays a live scrolling spectrogram, and accepts:
  - Microphone input (default)
  - WAV / MP3 / FLAC file drag-and-drop or --file flag

Requirements:
  pip install numpy scipy sounddevice soundfile matplotlib tkinter
  (tkinter usually ships with Python)

Your .cu files must be compiled first (the script does this automatically
if nvcc is on PATH, otherwise compile manually -- see COMPILE section below).

Usage:
  python gpu_fft_demo.py               # microphone
  python gpu_fft_demo.py --file song.wav
  python gpu_fft_demo.py --n 1021      # force Bluestein (non-pow2 window)
  python gpu_fft_demo.py --n 1024      # radix-2 window
"""

import argparse
import ctypes
import os
import platform
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# ── optional heavy deps -- imported lazily so missing ones give clear errors ──

def require(pkg, pip_name=None):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        name = pip_name or pkg
        print(f"[ERROR] Missing package '{name}'.  Run:  pip install {name}")
        sys.exit(1)

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GPU FFT live spectrogram demo")
parser.add_argument("--file",  default=None, help="Audio file to analyse (wav/mp3/flac)")
parser.add_argument("--n",     type=int, default=None,
                    help="FFT window size (default: 1024). Non-pow2 -> Bluestein.")
parser.add_argument("--sr",    type=int, default=22050, help="Sample rate (default 22050)")
parser.add_argument("--no-gpu", action="store_true",
                    help="Skip GPU and use numpy FFT (for machines without CUDA)")
args = parser.parse_args()

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()

# Try to find the .cu source files next to this script
SRC_HYBRID    = SCRIPT_DIR / "fft_gpu.cu"
SRC_BLUESTEIN = SCRIPT_DIR / "fft_gpu_bluestein.cu"

# Compiled shared libraries we will build
IS_WIN = (platform.system() == "Windows")
EXT = ".dll" if IS_WIN else ".so"
LIB_DIR   = SCRIPT_DIR / "_fft_libs"
LIB_HYBRID    = LIB_DIR / ("fft_hybrid" + EXT)
LIB_BLUESTEIN = LIB_DIR / ("fft_bluestein" + EXT)

# ── GPU compilation ────────────────────────────────────────────────────────────
WRAPPER_HYBRID = """
// Hand-rolled radix-2 FFT using shared + global memory.
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

using fc = cuFloatComplex;
static constexpr float PI = 3.14159265358979f;
static constexpr int   TPB = 256;

// ---- Global-memory butterfly stage (len > chunk phase) ----
__global__ void stage_g(fc* data, int n, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;
    int half  = len >> 1;
    int group = tid / half;
    int k     = tid - group * half;
    int lo    = group * len + k;
    int hi    = lo + half;
    float ang = -2.f * PI * k / (float)len;
    fc w = make_cuFloatComplex(cosf(ang), sinf(ang));
    fc u = data[lo];
    fc v = cuCmulf(data[hi], w);
    data[lo] = cuCaddf(u, v);
    data[hi] = cuCsubf(u, v);
}

// ---- Shared-memory local FFT stage (first log2(chunk) levels) ----
// Each block handles `chunk` elements; each thread handles one butterfly pair.
__global__ void local_fft_k(fc* data, int chunk) {
    extern __shared__ fc sh[];
    int base = blockIdx.x * chunk;
    int tid  = threadIdx.x;   // tid in [0, chunk/2)

    // Load the two elements this thread owns into shared memory
    sh[tid]            = data[base + tid];
    sh[tid + chunk/2]  = data[base + tid + chunk/2];
    __syncthreads();

    for (int len = 2; len <= chunk; len <<= 1) {
        int half  = len >> 1;
        int group = tid / half;
        int k     = tid - group * half;
        int lo    = group * len + k;
        int hi    = lo + half;
        float ang = -2.f * PI * (float)k / (float)len;
        fc w = make_cuFloatComplex(cosf(ang), sinf(ang));
        fc u = sh[lo];
        fc v = cuCmulf(sh[hi], w);
        sh[lo] = cuCaddf(u, v);
        sh[hi] = cuCsubf(u, v);
        __syncthreads();
    }

    data[base + tid]           = sh[tid];
    data[base + tid + chunk/2] = sh[tid + chunk/2];
}

// ---- GPU bit-reversal permutation ----
__device__ int bit_rev_d(int x, int bits) {
    int r = 0;
    for (int i = 0; i < bits; i++) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

__global__ void bitrev_k(fc* data, int n, int lg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = bit_rev_d(i, lg);
    if (i < j) { fc t = data[i]; data[i] = data[j]; data[j] = t; }
}

static int ilog2(int n) { int r = 0; while ((1 << r) < n) r++; return r; }

static void fft_hybrid_gpu(fc* d, int n) {
    int chunk   = (n < 2048) ? n : 2048;
    int threads = chunk / 2;       // one thread per butterfly pair
    int blocks  = n / chunk;
    int gb      = (n / 2 + TPB - 1) / TPB;

    bitrev_k<<<(n + TPB - 1) / TPB, TPB>>>(d, n, ilog2(n));
    cudaDeviceSynchronize();       // must finish before shared-mem FFT reads

    local_fft_k<<<blocks, threads, chunk * sizeof(fc)>>>(d, chunk);
    cudaDeviceSynchronize();       // must finish before global-mem stages

    for (int len = chunk * 2; len <= n; len <<= 1) {
        stage_g<<<gb, TPB>>>(d, n, len);
        cudaDeviceSynchronize();   // each stage reads results of the previous one
    }
}

extern "C" {EXPORT_MACRO} float fft_hybrid_magnitude(
    const float* re_in, const float* im_in, float* mag_out, int n)
{
    fc* d_data;
    cudaMalloc(&d_data, n * sizeof(fc));

    fc* h_buf = (fc*)malloc(n * sizeof(fc));
    for (int i = 0; i < n; i++)
        h_buf[i] = make_cuFloatComplex(re_in[i], im_in ? im_in[i] : 0.f);
    cudaMemcpy(d_data, h_buf, n * sizeof(fc), cudaMemcpyHostToDevice);
    free(h_buf);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    fft_hybrid_gpu(d_data, n);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    fc* h_out = (fc*)malloc(n * sizeof(fc));
    cudaMemcpy(h_out, d_data, n * sizeof(fc), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n / 2; i++)
        mag_out[i] = sqrtf(h_out[i].x * h_out[i].x + h_out[i].y * h_out[i].y);

    free(h_out);
    cudaFree(d_data);
    return ms;
}
"""

WRAPPER_BLUESTEIN = """
// Hand-rolled Bluestein (chirp-z) FFT for arbitrary N.
// Algorithm: X[k] = chirp_k * IFFT( FFT(chirp*x) * conj_chirp_FFT )[k]
// where  chirp[n] = exp(-j*pi*n^2/N),  conj_chirp[n] = exp(+j*pi*n^2/N)
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

using fc = cuFloatComplex;
static constexpr float PI = 3.14159265358979f;
static constexpr int   TPB = 256;

static int ilog2b(int n) { int r = 0; while ((1 << r) < n) r++; return r; }
static int ceilpow2(int v) { int m = 1; while (m < v) m <<= 1; return m; }

__global__ void bitrev_b(fc* data, int n, int lg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = (unsigned)i;
    x = (x >> 16) | (x << 16);
    x = ((x & 0xff00ff00u) >> 8)  | ((x & 0x00ff00ffu) << 8);
    x = ((x & 0xf0f0f0f0u) >> 4)  | ((x & 0x0f0f0f0fu) << 4);
    x = ((x & 0xccccccccu) >> 2)  | ((x & 0x33333333u) << 2);
    x = ((x & 0xaaaaaaaau) >> 1)  | ((x & 0x55555555u) << 1);
    int j = (int)(x >> (32 - lg));
    if (i < j) { fc t = data[i]; data[i] = data[j]; data[j] = t; }
}

__global__ void butterfly_b(fc* data, int n, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;
    int half  = len >> 1;
    int group = tid / half;
    int k     = tid - group * half;
    int lo    = group * len + k;
    int hi    = lo + half;
    float ang = -2.f * PI * (float)k / (float)len;
    fc w = make_cuFloatComplex(cosf(ang), sinf(ang));
    fc u = data[lo], v = cuCmulf(data[hi], w);
    data[lo] = cuCaddf(u, v);
    data[hi] = cuCsubf(u, v);
}

// Negate imaginary part (complex conjugate)
__global__ void conj_b(fc* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i].y = -data[i].y;
}

__global__ void scale_b(fc* data, int n, float f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { data[i].x *= f; data[i].y *= f; }
}

__global__ void zerofill_b(fc* data, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) data[i] = make_cuFloatComplex(0.f, 0.f);
}

// A[k] = x[k] * exp(-j*pi*k^2/N)
__global__ void pre_b(const fc* x, fc* A, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    long long ksq = (long long)k * k % (2LL * n);
    float ang = -PI * (float)ksq / (float)n;
    A[k] = cuCmulf(x[k], make_cuFloatComplex(cosf(ang), sinf(ang)));
}

// B[k] = exp(+j*pi*k^2/N) zero-padded symmetrically for circular convolution
__global__ void build_b_k(fc* B, int n, int m) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    long long ksq = (long long)k * k % (2LL * n);
    float ang = PI * (float)ksq / (float)n;
    fc chirp = make_cuFloatComplex(cosf(ang), sinf(ang));
    B[k] = chirp;
    if (k > 0) B[m - k] = chirp;
}

__global__ void hadamard_b(fc* A, const fc* B, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) A[i] = cuCmulf(A[i], B[i]);
}

// y[k] = A[k] * exp(-j*pi*k^2/N)
__global__ void post_b(const fc* A, fc* y, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    long long ksq = (long long)k * k % (2LL * n);
    float ang = -PI * (float)ksq / (float)n;
    y[k] = cuCmulf(A[k], make_cuFloatComplex(cosf(ang), sinf(ang)));
}

static void fft2_fwd(fc* d, int m) {
    int bf = (m + TPB - 1) / TPB;
    int bh = (m / 2 + TPB - 1) / TPB;
    bitrev_b<<<bf, TPB>>>(d, m, ilog2b(m));  cudaDeviceSynchronize();
    for (int len = 2; len <= m; len <<= 1) {
        butterfly_b<<<bh, TPB>>>(d, m, len); cudaDeviceSynchronize();
    }
}

// IFFT via identity: IFFT(x) = conj(FFT(conj(x))) / m
static void fft2_inv(fc* d, int m) {
    int bf = (m + TPB - 1) / TPB;
    conj_b<<<bf, TPB>>>(d, m);  cudaDeviceSynchronize();
    fft2_fwd(d, m);
    conj_b<<<bf, TPB>>>(d, m);  cudaDeviceSynchronize();
    scale_b<<<bf, TPB>>>(d, m, 1.f / (float)m);  cudaDeviceSynchronize();
}

extern "C" {EXPORT_MACRO} float fft_bluestein_magnitude(
    const float* re_in, const float* im_in, float* mag_out, int n)
{
    int M = ceilpow2(2 * n - 1);
    fc *d_x, *d_y, *d_A, *d_B;
    cudaMalloc(&d_x, n * sizeof(fc));
    cudaMalloc(&d_y, n * sizeof(fc));
    cudaMalloc(&d_A, M * sizeof(fc));
    cudaMalloc(&d_B, M * sizeof(fc));

    fc* h_in = (fc*)malloc(n * sizeof(fc));
    for (int i = 0; i < n; i++)
        h_in[i] = make_cuFloatComplex(re_in[i], im_in ? im_in[i] : 0.f);
    cudaMemcpy(d_x, h_in, n * sizeof(fc), cudaMemcpyHostToDevice);
    free(h_in);

    int bm = (M + TPB - 1) / TPB;
    int bn = (n + TPB - 1) / TPB;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    zerofill_b<<<bm, TPB>>>(d_A, M);       cudaDeviceSynchronize();
    zerofill_b<<<bm, TPB>>>(d_B, M);       cudaDeviceSynchronize();
    pre_b<<<bn, TPB>>>(d_x, d_A, n);       cudaDeviceSynchronize();
    build_b_k<<<bn, TPB>>>(d_B, n, M);     cudaDeviceSynchronize();
    fft2_fwd(d_A, M);
    fft2_fwd(d_B, M);
    hadamard_b<<<bm, TPB>>>(d_A, d_B, M);  cudaDeviceSynchronize();
    fft2_inv(d_A, M);
    post_b<<<bn, TPB>>>(d_A, d_y, n);      cudaDeviceSynchronize();

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    fc* h_out = (fc*)malloc(n * sizeof(fc));
    cudaMemcpy(h_out, d_y, n * sizeof(fc), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n / 2; i++)
        mag_out[i] = sqrtf(h_out[i].x * h_out[i].x + h_out[i].y * h_out[i].y);

    free(h_out);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_A); cudaFree(d_B);
    return ms;
}
"""

def compile_libs():
    """Compile the two CUDA shared libraries if not already present."""
    if args.no_gpu:
        return False

    # Prefer CUDA 12.x nvcc -- CUDA 13.x requires a driver newer than 551.88
    import glob as _glob
    _candidates = (
        _glob.glob(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12*\bin\nvcc.exe') +
        _glob.glob(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11*\bin\nvcc.exe') +
        ["nvcc"]
    )
    nvcc = _candidates[0]
    try:
        subprocess.run([nvcc, "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[WARN] nvcc not found -- falling back to numpy FFT.")
        return False

    LIB_DIR.mkdir(exist_ok=True)

    import hashlib
    def needs_rebuild(lib, wrapper_src, code_str):
        if not lib.exists():
            return True
        hash_file = lib.with_suffix(".hash")
        current_hash = hashlib.md5(code_str.encode("utf-8")).hexdigest()
        if not hash_file.exists():
            return True
        return hash_file.read_text(encoding="utf-8").strip() != current_hash

    # ── Detect Local GPU Architecture ──
    local_arch = "sm_80"
    detect_src = LIB_DIR / "detect_arch.cu"
    detect_exe = LIB_DIR / ("detect" + (".exe" if IS_WIN else ""))
    if not detect_exe.exists():
        detect_src.write_text('''
        #include <cuda_runtime.h>
        #include <stdio.h>
        int main() {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
                printf("sm_%d%d", prop.major, prop.minor);
                return 0;
            }
            return 1;
        }
        ''', encoding="utf-8")
        try:
            res = subprocess.run([nvcc, str(detect_src), "-o", str(detect_exe)], capture_output=True)
            if res.returncode == 0:
                res2 = subprocess.run([str(detect_exe)], capture_output=True, text=True)
                if res2.returncode == 0:
                    local_arch = res2.stdout.strip()
        except: pass
    else:
        try:
            local_arch = subprocess.run([str(detect_exe)], capture_output=True, text=True).stdout.strip()
        except: pass

    archs = [local_arch, "sm_80", "sm_75", "sm_61", "sm_52", "sm_50", "sm_35"]
    
    ok = True
    for wrapper_code, lib_path, name in [
        (WRAPPER_HYBRID,    LIB_HYBRID,    "hybrid"),
        (WRAPPER_BLUESTEIN, LIB_BLUESTEIN, "bluestein"),
    ]:
        # Inject the dllexport macro on Windows directly, as nvcc _WIN32 might not trigger
        export_macro = "__declspec(dllexport)" if IS_WIN else ""
        code_to_write = wrapper_code.replace("{EXPORT_MACRO}", export_macro)
        
        if not needs_rebuild(lib_path, f"_wrap_{name}.cu", code_to_write):
            print(f"[GPU] {lib_path.name} already compiled.")
            continue
        print(f"[GPU] Compiling {name} library...")
        
        src = LIB_DIR / f"_wrap_{name}.cu"
        src.write_text(code_to_write, encoding="utf-8")

        success = False
        for arch in archs:
            cmd = [
                nvcc, "-O2", "-std=c++17",
                f"-arch={arch}",
                "--shared",
            ]
            if not IS_WIN:
                cmd += ["-Xcompiler", "-fPIC"]
            
            cmd += [str(src), "-o", str(lib_path), "-lcufft"]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[GPU] {lib_path.name} compiled OK (arch={arch}).")
                success = True
                lib_path.with_suffix(".hash").write_text(hashlib.md5(code_to_write.encode("utf-8")).hexdigest(), encoding="utf-8")
                break
            
            # If the error is NOT about the architecture, stop falling back so we see it
            combined_out = (result.stdout + result.stderr).lower()
            if "unsupported gpu architecture" not in combined_out:
                break
        
        if not success:
            print(f"[ERROR] Compile failed for {name}.")
            if result.stdout: print(f"stdout:\n{result.stdout}")
            if result.stderr: print(f"stderr:\n{result.stderr}")
            if IS_WIN:
                print("[TIP] On Windows, ensure Visual Studio (cl.exe) is in your PATH.")
            ok = False
            
    return ok


def load_libs():
    """Load the compiled shared libraries and return (hybrid_fn, bluestein_fn) or None."""
    # On Windows, add common CUDA bin dirs so ctypes can find cudart/cufft DLLs
    # even when running from regular PowerShell (not x64 Native Tools)
    if IS_WIN:
        cuda_homes = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            r"C:\Program Files\NVIDIA Corporation\CUDA",
        ]
        import glob
        for cuda_home in cuda_homes:
            for bin_dir in glob.glob(cuda_home + r"\v*\bin"):
                try:
                    os.add_dll_directory(bin_dir)
                except Exception:
                    pass

    try:
        lib_h = ctypes.CDLL(str(LIB_HYBRID))
        lib_b = ctypes.CDLL(str(LIB_BLUESTEIN))
    except OSError as e:
        print(f"[WARN] Could not load GPU libs: {e}")
        return None

    f32p = ctypes.POINTER(ctypes.c_float)

    def setup(lib, name):
        fn = getattr(lib, name)
        fn.restype  = ctypes.c_float
        fn.argtypes = [f32p, f32p, f32p, ctypes.c_int]
        return fn

    return (
        setup(lib_h, "fft_hybrid_magnitude"),
        setup(lib_b, "fft_bluestein_magnitude"),
    )


# ── FFT dispatch ──────────────────────────────────────────────────────────────
def is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def next_pow2(n):
    m = 1
    while m < n:
        m <<= 1
    return m


def run_fft(samples: np.ndarray, n: int, gpu_fns):
    """
    Compute magnitude spectrum of `samples` (length n) using GPU if available.
    Returns (mag[n//2], elapsed_ms, method_str).
    """
    win = np.hanning(n)
    chunk = (samples[:n] * win).astype(np.float32)
    if len(chunk) < n:
        chunk = np.pad(chunk, (0, n - len(chunk)))

    if gpu_fns is not None:
        hybrid_fn, blue_fn = gpu_fns
        re_c  = chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # Pass a properly typed NULL pointer for the imaginary part
        im_in = ctypes.cast(None, ctypes.POINTER(ctypes.c_float))
        mag   = np.zeros(n // 2, dtype=np.float32)
        mag_c = mag.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if is_pow2(n):
            ms = hybrid_fn(re_c, im_in, mag_c, n)
            method = f"GPU hand-rolled hybrid  (N={n})"
        else:
            ms = blue_fn(re_c, im_in, mag_c, n)
            method = f"GPU hand-rolled Bluestein  (N={n})"
        return mag, ms, method
    else:
        # numpy fallback
        t0  = time.perf_counter()
        out = np.fft.rfft(chunk, n=n)
        ms  = (time.perf_counter() - t0) * 1000
        mag = np.abs(out[: n // 2]).astype(np.float32)
        method = f"numpy FFT (N={n})"
        return mag, ms, method


# ── Audio source ──────────────────────────────────────────────────────────────
class AudioSource:
    """Yields chunks of float32 mono audio at `sr` samples/s."""

    def __init__(self, sr: int, n: int, file_path=None):
        self.sr = sr
        self.n  = n
        self.q  : queue.Queue = queue.Queue(maxsize=20)
        self._stop = threading.Event()
        self._file = file_path
        self.paused = False
        self._data_orig = None
        self._file_sr = None
        self._pos = 0

    def set_pause(self, paused):
        self.paused = paused
        if not self._file:
            return
        try:
            sd = require("sounddevice")
            if paused:
                sd.stop()
            else:
                if self._data_orig is not None:
                    # Map resampled position back to original audio sample rate
                    current_sample = int(self._pos * (self._file_sr / self.sr))
                    sd.play(self._data_orig[current_sample:], samplerate=self._file_sr)
        except Exception:
            pass

    def start(self):
        if self._file:
            t = threading.Thread(target=self._file_thread, daemon=True)
        else:
            t = threading.Thread(target=self._mic_thread,  daemon=True)
        t.start()

    def stop(self):
        self._stop.set()

    def _mic_thread(self):
        sd = require("sounddevice")
        hop = self.n // 2

        def cb(indata, frames, t, status):
            mono = indata[:, 0].copy()
            try:
                self.q.put_nowait(mono)
            except queue.Full:
                pass

        with sd.InputStream(samplerate=self.sr, channels=1,
                            blocksize=hop, dtype="float32", callback=cb):
            self._stop.wait()

    def _file_thread(self):
        sf = require("soundfile")
        data, file_sr = sf.read(self._file, dtype="float32", always_2d=True)
        mono = data[:, 0]

        self._data_orig = data
        self._file_sr = file_sr

        # Play audio through the default output device in a background thread
        try:
            sd = require("sounddevice")
            if not self.paused:
                sd.play(data, samplerate=file_sr)
            print(f"[Audio] Playing '{Path(self._file).name}' through default output device.")
        except Exception as e:
            print(f"[WARN] Could not play audio: {e}")

        # resample if needed for FFT (doesn't affect playback, which uses original sr)
        if file_sr != self.sr:
            try:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(self.sr, file_sr)
                mono = resample_poly(mono, self.sr // g, file_sr // g).astype(np.float32)
            except ImportError:
                print("[WARN] scipy not installed -- skipping resample, pitch may be off")

        hop = self.n // 2
        self._pos = 0
        while not self._stop.is_set() and self._pos + hop <= len(mono):
            if self.paused:
                time.sleep(0.1)
                continue
                
            try:
                self.q.put(mono[self._pos: self._pos + hop], timeout=0.5)
            except queue.Full:
                pass
            self._pos += hop
            time.sleep(hop / self.sr * 0.8)  # pace playback roughly in real time

        # Stop audio playback when file ends
        try:
            sd.stop()
        except Exception:
            pass

        # signal end
        if not self._stop.is_set():
            self._stop.set()

    def get(self, timeout=0.5):
        return self.q.get(timeout=timeout)


# ── Matplotlib UI ─────────────────────────────────────────────────────────────
def main():
    plt = require("matplotlib.pyplot", "matplotlib")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Button, Slider, RadioButtons
    import matplotlib.gridspec as gridspec

    np_ = require("numpy", "numpy")

    # ── detect N ──────────────────────────────────────────────────────────────
    N = args.n if args.n else 1024
    SR = args.sr

    # ── compile / load GPU ────────────────────────────────────────────────────
    gpu_ok  = compile_libs() if not args.no_gpu else False
    gpu_fns = load_libs()    if gpu_ok else None
    if gpu_fns:
        print("[GPU] CUDA libraries loaded successfully.")
    else:
        print("[INFO] Running with numpy FFT (CPU fallback).")

    # ── state ──────────────────────────────────────────────────────────────────
    GRAM_COLS = 200
    state = {
        "n":         N,
        "buffer":    np.zeros(N, dtype=np.float32),
        "gram_data": np.zeros((N // 2, GRAM_COLS), dtype=np.float32),
        "paused":    False,
        "src":       AudioSource(sr=SR, n=N, file_path=args.file),
        "gram_vmax": 2.0 if not args.file else 10.0
    }
    state["src"].start()

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 7), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(3, 2, height_ratios=[0.08, 0.42, 0.5],
                            hspace=0.35, wspace=0.3,
                            left=0.07, right=0.97, top=0.93, bottom=0.05)

    ax_ctrl  = fig.add_subplot(gs[0, :])   # top: controls row
    ax_spec  = fig.add_subplot(gs[1, :])   # middle: live spectrum
    ax_gram  = fig.add_subplot(gs[2, :])   # bottom: spectrogram

    for ax in (ax_ctrl, ax_spec, ax_gram):
        ax.set_facecolor("#0d0d1a")
    ax_ctrl.axis("off")

    # ── spectrum axes ──────────────────────────────────────────────────────────
    freqs    = np.linspace(0, SR / 2, N // 2)
    spec_ln, = ax_spec.plot(freqs, np.zeros(N // 2), color="#00d4ff", lw=1.2)
    ax_spec.set_xlim(0, SR / 2)
    ax_spec.set_ylim(0, 50)
    ax_spec.set_xlabel("Frequency (Hz)", color="#aaa", fontsize=9)
    ax_spec.set_ylabel("Magnitude",       color="#aaa", fontsize=9)
    ax_spec.tick_params(colors="#888", labelsize=8)
    for spine in ax_spec.spines.values():
        spine.set_edgecolor("#333")
    ax_spec.grid(True, color="#222", lw=0.5)

    title_txt = ax_spec.set_title("", color="#00d4ff", fontsize=10, pad=4)

    # we moved state instantiation to the top to include src

    # ── spectrogram ────────────────────────────────────────────────────────────
    gram_img  = ax_gram.imshow(
        state["gram_data"], aspect="auto", origin="lower",
        extent=[0, GRAM_COLS, 0, SR / 2],
        cmap="inferno", vmin=0, vmax=state["gram_vmax"],
    )
    ax_gram.set_xlabel("Time ->",           color="#aaa", fontsize=9)
    ax_gram.set_ylabel("Frequency (Hz)",   color="#aaa", fontsize=9)
    ax_gram.set_title("Spectrogram",       color="#aaa", fontsize=9, pad=3)
    ax_gram.tick_params(colors="#888", labelsize=8)
    for spine in ax_gram.spines.values():
        spine.set_edgecolor("#333")

    # colorbar
    cbar = fig.colorbar(gram_img, ax=ax_gram, fraction=0.02, pad=0.01)
    cbar.ax.tick_params(colors="#888", labelsize=7)


    N_OPTIONS = [256, 512, 1021, 1024, 1025, 2000, 2048, 4000, 4096]

    # ── N selector (radio below the spectrum) ──────────────────────────────────
    ax_radio = fig.add_axes([0.01, 0.02, 0.12, 0.38], facecolor="#0d0d1a")
    ax_radio.set_title("Window N", color="#aaa", fontsize=8, pad=2)
    labels = [str(n) + (" ★" if is_pow2(n) else "") for n in N_OPTIONS]
    radio  = RadioButtons(ax_radio, labels,
                          active=N_OPTIONS.index(N) if N in N_OPTIONS else 1)
    for lbl in radio.labels:
        lbl.set_fontsize(8)
        lbl.set_color("#ccc")

    def on_radio(label):
        idx = labels.index(label)
        new_n = N_OPTIONS[idx]
        state["n"] = new_n
        state["buffer"] = np.zeros(new_n, dtype=np.float32)
        state["gram_data"] = np.zeros((new_n // 2, GRAM_COLS), dtype=np.float32)
        # update image data and extent for the new resolution
        gram_img.set_data(state["gram_data"])
        gram_img.set_extent([0, GRAM_COLS, 0, SR / 2])
        fig.canvas.draw_idle()

    radio.on_clicked(on_radio)

    # ── switch source button ───────────────────────────────────────────────────
    ax_switch = fig.add_axes([0.77, 0.01, 0.08, 0.04])
    btn_switch = Button(ax_switch, "Switch Src", color="#222", hovercolor="#444")
    btn_switch.label.set_color("#eee")

    def on_switch(_):
        old_src = state["src"]
        was_paused = state["paused"]
        
        # Toggle file
        new_file = None if old_src._file else "song.wav"
        
        old_src.stop()
        old_src.set_pause(True)  # force audio stop
        
        new_src = AudioSource(sr=SR, n=state["n"], file_path=new_file)
        new_src.start()
        if was_paused:
            new_src.set_pause(True)
            
        state["src"] = new_src
        # Adjust sensitivity for source
        val = 2.0 if not new_file else 10.0
        slider.set_val(val)
        fig.canvas.draw_idle()

    btn_switch.on_clicked(on_switch)

    # ── pause button ───────────────────────────────────────────────────────────
    ax_pause = fig.add_axes([0.86, 0.01, 0.06, 0.04])
    btn_pause = Button(ax_pause, "Pause", color="#222", hovercolor="#444")
    btn_pause.label.set_color("#eee")

    def on_pause(_):
        state["paused"] = not state["paused"]
        state["src"].set_pause(state["paused"])
        btn_pause.label.set_text("Resume" if state["paused"] else "Pause")
        fig.canvas.draw_idle()

    btn_pause.on_clicked(on_pause)

    
    # ── Gram Intensity Slider ─────────────────────────────────────────────────────────
    ax_slider = fig.add_axes([0.35, 0.015, 0.35, 0.03], facecolor="#0d0d1a")
    slider = Slider(ax_slider, "Sensitivity", 0.1, 50.0, valinit=state["gram_vmax"], color="#00d4ff")
    slider.label.set_color("#aaa")
    slider.label.set_fontsize(8)

    def on_slider(val):
        state["gram_vmax"] = val
        gram_img.set_clim(vmax=val)
        fig.canvas.draw_idle()
    slider.on_changed(on_slider)

    # ── animation update ───────────────────────────────────────────────────────
    def update(_frame):
        if state["paused"]:
            return spec_ln, gram_img, title_txt

        n = state["n"]

        try:
            chunk = state["src"].get(timeout=0.05)
        except queue.Empty:
            return spec_ln, gram_img, title_txt

        # update rolling buffer
        buf = state["buffer"]
        if len(buf) != n:
            buf = np.zeros(n, dtype=np.float32)
        hop = min(len(chunk), n)
        buf = np.roll(buf, -hop)
        buf[-hop:] = chunk[:hop]
        state["buffer"] = buf

        # compute FFT
        mag, ms, method = run_fft(buf, n, gpu_fns)

        # update spectrum
        f = np.linspace(0, SR / 2, n // 2)
        spec_ln.set_data(f, mag)
        ax_spec.set_xlim(0, SR / 2)

        # update spectrogram
        gd = state["gram_data"]
        gd[:, :-1] = gd[:, 1:]
        gd[:, -1]  = mag
        gram_img.set_data(gd)
        gram_img.set_extent([0, GRAM_COLS, 0, SR / 2])

        # update title
        alg = "Bluestein" if not is_pow2(n) else "Hybrid radix-2"
        m_str = f"  M={next_pow2(2*n-1)}" if not is_pow2(n) else ""
        src_str = f"file: {Path(state['src']._file).name}" if state["src"]._file else "microphone"
        title_txt.set_text(
            f"{alg}{m_str}  |  N={n}  |  {ms*1000:.1f} µs  |  source: {src_str}"
        )
        col = "#ff9f43" if not is_pow2(n) else "#00d4ff"
        title_txt.set_color(col)

        return spec_ln, gram_img, title_txt

    ani = animation.FuncAnimation(
        fig, update, interval=40, blit=False, cache_frame_data=False
    )

    fig.suptitle(
        "GPU FFT Live Demo  --  ★ = power-of-2 (radix-2)   other = Bluestein",
        color="#888", fontsize=9, y=0.98,
    )

    plt.show()
    state["src"].stop()


if __name__ == "__main__":
    main()
