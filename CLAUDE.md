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

**compare_fields.py** - Compares two raw binary fields and analyzes error bounds
- Input: Two float32 raw files (ground truth and test), dimensions m×n
- Output: `_diff.f32.raw` (difference field), `_oob_mask.f32.raw` (out-of-bound mask, 1.0=exceeded)
- REL error bound: `|error| > rel_bound × (max - min)` (same as SZ3)
- Example: `python python_scripts/compare_fields.py ground.raw test.raw 512 512 output --rel 1e-3`

**gradient_matrix.py** - Generates central difference gradient as sparse matrices
- Output: `_Dx.npz`, `_Dy.npz` (scipy sparse CSR format)
- Matrix size: (m×n) × (m×n) for 2D field of size m×n
- Supports order 2/4/6/8 and boundary conditions: circular/symmetric/replicate/zero
- Uses Kronecker structure: `Dx = I_m ⊗ Dx_1d`, `Dy = Dy_1d ⊗ I_n`
- Example: `python python_scripts/gradient_matrix.py 512 512 result/grad --order 8 --boundary circular`
- Apply to field: `python python_scripts/gradient_matrix.py 512 512 result/grad --order 8 --apply data/field.raw`

**svd_analysis.py** - SVD analysis of gradient matrices for invertibility
- Analyzes rank and null space via Kronecker product structure (fast)
- For circular BC with even n: null space dim = 2 per 1D (DC + Nyquist modes)
- 512×512 field with order 8: Dx/Dy rank = 261,120, null dim = 1,024 (99.61% invertible)
- Combined gradient [Dx; Dy]: rank = 262,140, null dim = 4 (99.998% invertible)
- Example: `python python_scripts/svd_analysis.py --size 512 --order 8 --boundary circular`

## External Dependencies

### SZ3 Lossy Compressor

Located in `external/SZ3/` as a git submodule. SZ3 is an error-bounded lossy compressor for scientific floating-point and integer data.

**Repository**: https://github.com/szcompressor/SZ3

**Build** (already built, but to rebuild):
```bash
cd external/SZ3/build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_SHARED_LIBS=ON
make -j$(sysctl -n hw.ncpu)
make install
```

**Usage** (CLI):
```bash
# Compress with absolute error bound 1e-3
./external/SZ3/install/bin/sz3 -f -i input.raw -z output.sz -2 512 512 -M ABS 1e-3

# Decompress
./external/SZ3/install/bin/sz3 -f -z output.sz -o decompressed.raw -2 512 512

# Compress with relative error bound
./external/SZ3/install/bin/sz3 -f -i input.raw -z output.sz -2 512 512 -M REL 1e-4
```

**Error modes**: ABS (absolute), REL (relative to value range), PSNR, NORM, ABS_AND_REL, ABS_OR_REL

**REL error bound definition**: `|x - x'| ≤ rel_bound × (max_value - min_value)`

**C++ API**: Include `<SZ3/api/sz.hpp>`, link against SZ3 headers (header-only for core API)

## FFT Wavenumber Layout

MATLAB's fft2 output (without fftshift) for an n×n grid:
- Index 1: k=0 (DC)
- Index 2 to n/2: k=1 to n/2-1 (positive frequencies)
- Index n/2+1: k=-n/2 (Nyquist, highest frequency)
- Index n/2+2 to n: k=-n/2+1 to -1 (negative frequencies, conjugate symmetric with positive)
