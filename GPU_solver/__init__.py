"""
GPU-accelerated solvers for error-bounded lossy compression projection.

This module provides GPU implementations of the projection algorithms
defined in the feasible_region_lemmas.tex mathematical framework.

Available Solvers:
    - ADMMSolver: ADMM-based solver using PyTorch (multi_operator_projection_gpu.py)
    - CuPyADMMSolver: ADMM-based solver using CuPy (cupy_solver.py)

Requirements:
    - PyTorch with CUDA support (for multi_operator_projection_gpu.py)
    - CuPy with CUDA support (for cupy_solver.py)

Usage:
    python GPU_solver/multi_operator_projection_gpu.py config.json
    python GPU_solver/cupy_solver.py config.json
"""

__version__ = '0.1.0'
