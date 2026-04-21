# GPU FFT Live Spectrogram Demo

`gpu_fft_demo.py` is a real-time visualization tool that demonstrates the performance and accuracy of custom-built CUDA FFT kernels. It captures audio input and displays it as a scrolling "waterfall" spectrogram.

## Features

- **Custom Kernel Execution**: Unlike standard demos using `cuFFT`, this tool runs hand-rolled **Hybrid Radix-2** and **Bluestein** kernels.
- **Dynamic Windowing**: Choose between power-of-2 (★) sizes for maximum speed or arbitrary sizes for precise frequency bins.
- **Dual Source Support**: Hot-swap between your system **microphone** and high-fidelity **WAV files**.
- **Interactive Controls**:
    - **Sensitivity Slider**: Adjust the color intensity on the fly to see faint frequency lines in quiet environments.
    - **N Selector**: Change the FFT window size in real-time.
    - **Pause/Resume**: Freeze the analysis to inspect specific spectral features.

## Requirements

### Python Dependencies
Install the required packages using pip:
```bash
pip install numpy matplotlib sounddevice soundfile scipy
```

### Hardware
- **NVIDIA GPU**: Required for kernel acceleration.
- **Microphone**: For live analysis mode.

## Usage

Run the script from the root of the repository (using x64 Native Tools Command Prompt for VS 2022 if on Windows)

```bash
# Standard Microphone Mode (Log2 N=1024)
python gpu_fft_demo.py

# File Analysis Mode
python gpu_fft_demo.py --file song.wav

# Arbitrary Window Size (Triggers Bluestein Kernel)
python gpu_fft_demo.py --n 1021
```

## How it Works
1. **Compilation**: On startup, the script detects your GPU architecture (`SM_xx`) and automatically compiles the CUDA source into local DLLs using `nvcc`.
2. **Buffering**: Audio is captured via `sounddevice` and buffered into overlapping windows.
3. **GPU Processing**: The windowed audio is moved to VRAM, processed by the custom FFT kernels, and the magnitude spectrum is returned.
4. **Visualization**: Matplotlib handles the real-time plotting of the 1D spectrum and 2D rolling spectrogram.

## Troubleshooting
- **No nvcc on PATH**: If the script fails to compile, ensure the CUDA bin directory (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`) is in your system PATH.
- **Driver Mismatch**: Ensure your NVIDIA driver version supports the CUDA toolkit version installed.
