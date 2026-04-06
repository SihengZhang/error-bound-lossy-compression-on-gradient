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

**gradient_matrix.py** - Generates differential operators as sparse matrices
- Operations: `dx`, `dy` (first derivatives), `dxx`, `dyy`, `dxy` (second derivatives), `laplacian`
- Output naming: `matrix_<m>x<n>_<boundary>_<operation>_order<order>.npz` (scipy sparse CSR)
- Matrix size: (m×n) × (m×n) for 2D field of size m×n
- Supports order 2/4/6/8 and boundary conditions: circular/symmetric/replicate/zero
- Uses Kronecker structure: `Dx = I_m ⊗ Dx_1d`, `Dy = Dy_1d ⊗ I_n`
- Generate single op: `python python_scripts/gradient_matrix.py 512 512 result/matrices --op dy --order 8 --boundary circular`
- Generate all ops: `python python_scripts/gradient_matrix.py 512 512 result/matrices --op all --order 4`
- Apply to field: `python python_scripts/gradient_matrix.py 512 512 result/matrices --op laplacian --order 8 --apply data/field.raw`
- Run tests: `python python_scripts/gradient_matrix.py --test`

**svd_analysis.py** - SVD analysis of gradient matrices for invertibility
- Analyzes rank and null space via Kronecker product structure (fast)
- For circular BC with even n: null space dim = 2 per 1D (DC + Nyquist modes)
- 512×512 field with order 8: Dx/Dy rank = 261,120, null dim = 1,024 (99.61% invertible)
- Combined gradient [Dx; Dy]: rank = 262,140, null dim = 4 (99.998% invertible)
- Example: `python python_scripts/svd_analysis.py --size 512 --order 8 --boundary circular`

**direct_projection.py** - Projects error field onto feasible region using LP (L1) or QP (L2)
- Solves: minimize ||x^ - x*||_p s.t. |x^ - x| ≤ b (space), |Ax^ - Ax| ≤ c (gradient)
- Norm selection: `--norm L1` (linear programming) or `--norm L2` (quadratic programming, default)
- L1 uses scipy.linprog (HiGHS), L2 uses OSQP or CVXPY
- Requires: osqp or cvxpy for L2 (`pip install osqp`)
- Input: ground truth (x), error field (x*), gradient matrix A, error bounds (REL or ABS)
- Output (in result/L1/ or result/L2/):
  - `ground_truth_gradient.f32.raw` - Ax
  - `error_gradient.f32.raw` - Ax*
  - `gradient_diff.f32.raw` - Ax* - Ax
  - `gradient_oob_mask.f32.raw` - Out-of-bound mask (1.0=exceeded)
  - `projected_space.f32.raw` - x^ (projected)
  - `projected_gradient.f32.raw` - Ax^
  - `space_change_mask.f32.raw` - Changed pixels mask
  - `projected_space_error.f32.raw` - x^ - x (projected error in space)
  - `projected_gradient_error.f32.raw` - Ax^ - Ax (projected error in gradient)
  - `projection_stats.txt` - Statistics summary
- L2 example: `python python_scripts/direct_projection.py data/field.raw data/compressed.raw 512 512 result/matrices/matrix_512x512_circular_dy_order8.npz --space-rel 1e-3 --grad-rel 1e-3`
- L1 example: `python python_scripts/direct_projection.py data/field.raw data/compressed.raw 512 512 result/matrices/matrix_512x512_circular_dy_order8.npz --space-rel 1e-3 --grad-rel 1e-3 --norm L1`

**multi_operator_projection.py** - Projects error field with multiple operator constraints (Lemma 4.6)
- Solves: minimize ||e - e*||_p s.t. -b ≤ e ≤ b (space), -c^(k) ≤ A_k e ≤ c^(k) for k=1..m (operators)
- Supports multiple gradient operators simultaneously (dx, dy, laplacian, etc.)
- Configuration via JSON file (see `configs/` directory for examples)
- Norm selection: L1 (linear programming) or L2 (quadratic programming)
- Requires: osqp or cvxpy for L2 (`pip install osqp`)
- Config format:
  ```json
  {
    "ground_truth": "data/field.raw",
    "error_field": "data/compressed.raw",
    "dimensions": {"m": 512, "n": 512},
    "space_bounds": {"rel": 1e-3},
    "operators": [
      {"name": "dx", "matrix": "result/matrices/matrix_dx.npz", "bounds": {"rel": 1e-3}},
      {"name": "dy", "matrix": "result/matrices/matrix_dy.npz", "bounds": {"rel": 1e-3}}
    ],
    "norm": "L2",
    "output_dir": "result/L2_dx_dy",
    "solver": "auto"
  }
  ```
- Output (per-operator + combined):
  - Space: `ground_truth.f32.raw`, `projected_space.f32.raw`, `space_change_mask.f32.raw`
  - Per-operator: `ground_truth_<op>.f32.raw`, `projected_<op>.f32.raw`, `<op>_oob_mask.f32.raw`
  - Stats: `projection_stats.txt` with violation counts and distances
- Examples:
  - Two operators: `python python_scripts/multi_operator_projection.py configs/projection_dx_dy.json`
  - Single operator: `python python_scripts/multi_operator_projection.py configs/projection_dy_only.json`
  - L1 norm: `python python_scripts/multi_operator_projection.py configs/projection_L1_dx_dy.json`

## GPU Solvers

Located in `GPU_solver/`. GPU-accelerated implementations using ADMM algorithm.

**multi_operator_projection_gpu.py** - PyTorch-based GPU solver
- Algorithm: ADMM (Alternating Direction Method of Multipliers)
- Supports both L1 and L2 norm minimization
- Uses Conjugate Gradient for linear system solves
- Adaptive rho parameter tuning
- Falls back to CPU if CUDA unavailable
- Requires: `pip install torch` (with CUDA support)
- Config format: Same JSON as multi_operator_projection.py, with optional `admm` section:
  ```json
  {
    "admm": {
      "rho": 1.0,
      "max_iter": 2000,
      "tol_abs": 1e-6,
      "tol_rel": 1e-4,
      "adaptive_rho": true
    }
  }
  ```
- Examples:
  - Auto device: `python GPU_solver/multi_operator_projection_gpu.py configs/projection_dx_dy.json`
  - Specific GPU: `python GPU_solver/multi_operator_projection_gpu.py configs/projection_dx_dy.json --device cuda:0`
  - CPU mode: `python GPU_solver/multi_operator_projection_gpu.py configs/projection_dx_dy.json --device cpu`

**cupy_solver.py** - CuPy-based GPU solver (faster sparse operations)
- Algorithm: ADMM with CuPy sparse matrix operations
- Better performance for large sparse constraint matrices
- Requires: `pip install cupy-cuda11x` (or `cupy-cuda12x` for CUDA 12)
- Examples:
  - Default GPU: `python GPU_solver/cupy_solver.py configs/projection_dx_dy.json`
  - Specific GPU: `python GPU_solver/cupy_solver.py configs/projection_dx_dy.json --gpu 1`

**chambolle_pock_solver.py** - Chambolle-Pock (PDHG) solver for L1 optimization (PyTorch)
- Algorithm: Primal-Dual Hybrid Gradient (state-of-the-art for L1 + linear constraints)
- Specifically designed for L1 norm minimization
- Automatic step size selection based on operator norm estimation
- More stable convergence for L1 than ADMM
- Config format: Same JSON with optional `chambolle_pock` section:
  ```json
  {
    "chambolle_pock": {
      "max_iter": 5000,
      "tol": 1e-6,
      "theta": 1.0,
      "print_interval": 50
    }
  }
  ```
- Examples:
  - Run L1 projection: `python GPU_solver/chambolle_pock_solver.py result/configs/projection_dy_only.json`
  - Specific GPU: `python GPU_solver/chambolle_pock_solver.py config.json --device cuda:1`

**cupy_pdhg_solver.py** - CuPy-based PDHG solver (recommended for L1)
- Algorithm: PDHG (Chambolle-Pock) as described in `LaTeX/feasible_region_lemmas.tex`
- Uses CuPy for faster sparse matrix operations than PyTorch
- Features: adaptive step sizes, adaptive restart, fine-tuning, post-processing projection
- Requires: `pip install cupy-cuda11x` (or `cupy-cuda12x` for CUDA 12)
- Config format: Same JSON with optional `pdhg` section:
  ```json
  {
    "pdhg": {
      "tau": null,
      "sigma": null,
      "theta": 1.0,
      "max_iter": 20000,
      "tol": 1e-6,
      "primal_tol": null,
      "dual_tol": null,
      "bound_tol": 1e-5,
      "adaptive": true,
      "adaptive_gamma": 0.7,
      "adaptive_eta": 0.95,
      "fine_tune": true,
      "fine_tune_threshold": 1e-3,
      "fine_tune_factor": 0.7,
      "min_step": 1e-4,
      "restart": true,
      "restart_interval": 100,
      "post_project": true,
      "post_project_iter": 1000,
      "print_interval": 100
    }
  }
  ```
- Step size parameters:
  - `tau`: Primal step size (auto if null, requires tau*sigma*||K||^2 < 1)
  - `sigma`: Dual step size (auto if null)
  - `theta`: Extrapolation parameter [0,1], default 1.0
- Adaptive parameters:
  - `adaptive`: Enable adaptive step size balancing (recommended: true)
  - `adaptive_gamma`: Backtracking factor (recommended: 0.7)
  - `adaptive_eta`: Target primal/dual ratio
- Fine-tuning parameters (for tight convergence):
  - `fine_tune`: Enable automatic step reduction near solution
  - `fine_tune_factor`: Step reduction factor (recommended: 0.7, not too aggressive)
  - `min_step`: Minimum step size (recommended: 1e-4, prevents steps from getting too small)
- Post-processing (ensures feasibility):
  - `post_project`: Enable post-processing projection to eliminate remaining violations
  - `post_project_iter`: Max iterations for Dykstra-like projection
- Acceleration:
  - `restart`: Enable adaptive restart for faster convergence
  - `restart_interval`: Check restart condition every N iterations
- Examples:
  - Default: `python GPU_solver/cupy_pdhg_solver.py configs/pdhg_example.json`
  - Specific GPU: `python GPU_solver/cupy_pdhg_solver.py configs/pdhg_dx_dy.json --gpu 1`
  - Override step sizes: `python GPU_solver/cupy_pdhg_solver.py config.json --tau 0.01 --sigma 0.01`

## Configuration Files

Located in `configs/`. JSON configuration files for multi-operator projection.

**projection_dx_dy.json** - L2 projection with both x and y gradient constraints
**projection_dy_only.json** - L2 projection with y gradient constraint only
**projection_laplacian.json** - L2 projection with Laplacian constraint
**projection_L1_dx_dy.json** - L1 projection with both x and y gradient constraints

See `configs/README.md` for detailed configuration format and examples.

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
