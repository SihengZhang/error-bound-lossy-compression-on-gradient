# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project implementing error-bounded lossy compression algorithms for gradient data, targeting scientific computing and HPC applications.

## Technology Stack

- **MATLAB**: Spectral analysis and gradient computation
- **Python**: Primary development language
- **C++/CUDA**: High-performance compute kernels
- **CMake**: Build system for C++/CUDA components

## Data

- `data/` - Test datasets (gitignored), includes Nyx cosmological simulation baryon density fields in raw float32 format
- `result/` - Output files (gitignored)

## MATLAB Scripts

Located in `MATLAB_scripts/`. Run from repository root:

```bash
matlab -batch "addpath('MATLAB_scripts'); spectral_gradient('input.raw', m, n, 'output_prefix')"
```

**spectral_gradient.m** - Computes x/y gradients using spectral differentiation
- Input: 2D scalar field (float32, row-major)
- Output: `_fft_grad_x.raw`, `_fft_grad_y.raw`

**central_diff_gradient.m** - Computes x/y gradients using central difference
- Input: 2D scalar field (float32, row-major)
- Output: `_cd_grad_x.raw`, `_cd_grad_y.raw`
- Optional params: `order` (2/4/6/8, default 2), `boundary` (circular/symmetric/replicate/zero, default circular)
- Example: `central_diff_gradient('input.raw', 512, 512, 'output', 8, 'circular')`

**spectral_fft.m** - Computes 2D FFT with compact storage
- Input: 2D scalar field (float32, row-major)
- Output: Single `_fft.raw` file using conjugate symmetry:
  - Diagonal (i == j): real part
  - Bottom-left (i > j): real part
  - Top-right (i < j): imaginary part

## Python Scripts

Located in `python_scripts/`. Run from repository root:

```bash
python python_scripts/generate_cone_function_field.py
```

**generate_cone_function_field.py** - Generates synthetic test field with sharp gradients
- Output: 512x512 2D field with 6 cone functions along diagonal (bottom-left to top-right)
- Peak amplitudes: 1, 10, 100, 1000, 10000, 100000
- Cone function provides linear falloff (sharp gradient changes)
- 10% padding from corners
- Saves to `result/cone_field_512x512.f32.raw` (float32, row-major)
- Displays matplotlib visualization (linear and log10 scale)

**generate_step_field.py** - Generates synthetic test field with step discontinuities
- Output: 512x512 2D field with step functions
- Upper half (y): all zeros
- Lower half (y): 7 equal-width bands along x-axis with values 0, 1, 10, 100, 1000, 10000, 100000
- Saves to `result/step_field_512x512.f32.raw` (float32, row-major)
- Displays matplotlib visualization (linear and log10 scale)

## FFT Wavenumber Layout

MATLAB's fft2 output (without fftshift) for an n×n grid:
- Index 1: k=0 (DC)
- Index 2 to n/2: k=1 to n/2-1 (positive frequencies)
- Index n/2+1: k=-n/2 (Nyquist, highest frequency)
- Index n/2+2 to n: k=-n/2+1 to -1 (negative frequencies, conjugate symmetric with positive)
