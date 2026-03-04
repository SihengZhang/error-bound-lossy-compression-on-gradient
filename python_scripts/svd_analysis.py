#!/usr/bin/env python3
"""
SVD analysis of gradient matrices to determine invertible directions.

For large sparse matrices, full SVD is impractical. This script:
1. Analyzes small examples with full SVD
2. Uses truncated SVD for large matrices
3. Exploits Kronecker structure for efficient analysis

Usage:
    python python_scripts/svd_analysis.py [--size SIZE] [--order ORDER] [--boundary BOUNDARY]

Examples:
    python python_scripts/svd_analysis.py --size 8        # Small example with full SVD
    python python_scripts/svd_analysis.py --size 512      # Large example with analysis
"""

import argparse
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds, eigsh
from typing import Tuple, Dict


# Central difference coefficients
CENTRAL_DIFF_COEFFS: Dict[int, np.ndarray] = {
    2: np.array([-1/2, 0, 1/2]),
    4: np.array([1/12, -2/3, 0, 2/3, -1/12]),
    6: np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]),
    8: np.array([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]),
}


def build_1d_diff_matrix_dense(n: int, order: int, boundary: str = 'circular') -> np.ndarray:
    """Build 1D central difference matrix (dense, for small sizes)."""
    coeffs = CENTRAL_DIFF_COEFFS[order]
    half_width = order // 2

    D = np.zeros((n, n))
    for i in range(n):
        for k, coeff in enumerate(coeffs):
            offset = k - half_width
            if coeff == 0:
                continue
            if boundary == 'circular':
                j = (i + offset) % n
                D[i, j] = coeff
    return D


def analyze_1d_svd(n: int, order: int, boundary: str = 'circular'):
    """Analyze SVD of 1D gradient matrix."""
    print(f"\n{'='*60}")
    print(f"1D Gradient Matrix Analysis (n={n}, order={order}, {boundary})")
    print(f"{'='*60}")

    D = build_1d_diff_matrix_dense(n, order, boundary)

    # Full SVD
    U, S, Vh = np.linalg.svd(D)

    # Count non-zero singular values (using threshold)
    threshold = 1e-10
    rank = np.sum(S > threshold)
    null_dim = n - rank

    print(f"\nMatrix shape: ({n}, {n})")
    print(f"Rank: {rank}")
    print(f"Null space dimension: {null_dim}")

    print(f"\nSingular values:")
    print(f"  Max: {S.max():.6e}")
    print(f"  Min (non-zero): {S[S > threshold].min():.6e}" if rank > 0 else "  Min: N/A")
    print(f"  Near-zero (< {threshold}): {S[S <= threshold]}")

    # Condition number (for invertible part)
    if rank > 0:
        cond = S.max() / S[S > threshold].min()
        print(f"  Condition number (non-zero part): {cond:.2e}")

    # Show null space vectors
    if null_dim > 0:
        print(f"\nNull space basis vectors (right singular vectors with σ ≈ 0):")
        for i in range(null_dim):
            v = Vh[-(i+1), :]
            print(f"  v_{i+1}: {v[:min(8, n)]}{'...' if n > 8 else ''}")
            # Check if it's a constant vector
            if np.allclose(v, v[0]):
                print(f"       → Constant vector (all elements ≈ {v[0]:.4f})")

    return S, rank, null_dim


def analyze_2d_via_kronecker(m: int, n: int, order: int, boundary: str = 'circular'):
    """
    Analyze 2D gradient matrix using Kronecker product structure.

    For Dx = Im ⊗ Dx_1d:
    - Singular values of Dx are singular values of Dx_1d, each with multiplicity m
    - Null space dimension = m × (null space dimension of Dx_1d)

    For Dy = Dy_1d ⊗ In:
    - Singular values of Dy are singular values of Dy_1d, each with multiplicity n
    - Null space dimension = n × (null space dimension of Dy_1d)
    """
    print(f"\n{'='*60}")
    print(f"2D Gradient Matrix Analysis via Kronecker Structure")
    print(f"Field size: {m}×{n}, order={order}, {boundary}")
    print(f"{'='*60}")

    # Analyze 1D components
    Dx_1d = build_1d_diff_matrix_dense(n, order, boundary)
    Dy_1d = build_1d_diff_matrix_dense(m, order, boundary)

    # SVD of 1D matrices
    _, Sx, _ = np.linalg.svd(Dx_1d)
    _, Sy, _ = np.linalg.svd(Dy_1d)

    threshold = 1e-10

    # Dx analysis
    rank_x_1d = np.sum(Sx > threshold)
    null_x_1d = n - rank_x_1d

    print(f"\n--- Dx (x-gradient) = I_m ⊗ Dx_1d ---")
    print(f"Dx_1d singular values: {Sx}")
    print(f"Dx_1d rank: {rank_x_1d}, null dim: {null_x_1d}")
    print(f"\nDx (2D) properties:")
    print(f"  Shape: ({m*n}, {m*n})")
    print(f"  Rank: {m * rank_x_1d}")
    print(f"  Null space dimension: {m * null_x_1d}")
    print(f"  Invertible directions: {m * rank_x_1d} / {m*n} ({100*m*rank_x_1d/(m*n):.2f}%)")

    # Physical interpretation
    print(f"\n  Physical meaning of null space:")
    print(f"    Any field constant along x (within each row) is in null space")
    print(f"    = {m} independent 'row-constant' modes")

    # Dy analysis
    rank_y_1d = np.sum(Sy > threshold)
    null_y_1d = m - rank_y_1d

    print(f"\n--- Dy (y-gradient) = Dy_1d ⊗ I_n ---")
    print(f"Dy_1d singular values: {Sy}")
    print(f"Dy_1d rank: {rank_y_1d}, null dim: {null_y_1d}")
    print(f"\nDy (2D) properties:")
    print(f"  Shape: ({m*n}, {m*n})")
    print(f"  Rank: {n * rank_y_1d}")
    print(f"  Null space dimension: {n * null_y_1d}")
    print(f"  Invertible directions: {n * rank_y_1d} / {m*n} ({100*n*rank_y_1d/(m*n):.2f}%)")

    print(f"\n  Physical meaning of null space:")
    print(f"    Any field constant along y (within each column) is in null space")
    print(f"    = {n} independent 'column-constant' modes")

    # Combined gradient analysis
    print(f"\n--- Combined Gradient [Dx; Dy] ---")
    print(f"  If we stack Dx and Dy vertically:")
    print(f"  The combined null space = intersection of both null spaces")
    print(f"  = fields constant in BOTH x and y = globally constant fields")
    print(f"  Combined null space dimension: 1 (just the constant field)")
    print(f"  Combined invertible directions: {m*n - 1} / {m*n}")

    return {
        'Dx': {'rank': m * rank_x_1d, 'null_dim': m * null_x_1d},
        'Dy': {'rank': n * rank_y_1d, 'null_dim': n * null_y_1d},
        'combined_null_dim': 1
    }


def analyze_large_matrix_truncated(npz_path: str, k: int = 20):
    """Analyze large sparse matrix using truncated SVD."""
    print(f"\n{'='*60}")
    print(f"Truncated SVD Analysis: {npz_path}")
    print(f"{'='*60}")

    from scipy.sparse import load_npz

    D = load_npz(npz_path)
    print(f"Matrix shape: {D.shape}")
    print(f"Non-zeros: {D.nnz}")

    # Find largest singular values
    print(f"\nComputing {k} largest singular values...")
    U, S_large, Vh = svds(D, k=k, which='LM')
    S_large = np.sort(S_large)[::-1]
    print(f"Largest singular values: {S_large}")

    # Find smallest singular values via D^T D
    print(f"\nComputing {k} smallest singular values...")
    DtD = D.T @ D
    eigenvalues, _ = eigsh(DtD, k=k, which='SM', tol=1e-6)
    S_small = np.sqrt(np.maximum(eigenvalues, 0))
    S_small = np.sort(S_small)
    print(f"Smallest singular values: {S_small}")

    # Estimate rank
    threshold = 1e-10
    near_zero = np.sum(S_small < threshold)
    print(f"\nSingular values < {threshold}: {near_zero} (out of {k} computed)")
    print(f"Estimated null space dimension: ≥ {near_zero}")

    return S_large, S_small


def main():
    parser = argparse.ArgumentParser(description='SVD analysis of gradient matrices')
    parser.add_argument('--size', type=int, default=8, help='Field size (default: 8)')
    parser.add_argument('--order', type=int, default=2, choices=[2, 4, 6, 8],
                        help='Central difference order (default: 2)')
    parser.add_argument('--boundary', type=str, default='circular',
                        choices=['circular'], help='Boundary condition')
    parser.add_argument('--load', type=str, default=None,
                        help='Load and analyze existing .npz matrix file')

    args = parser.parse_args()

    if args.load:
        analyze_large_matrix_truncated(args.load)
    else:
        # 1D analysis
        if args.size <= 16:
            analyze_1d_svd(args.size, args.order, args.boundary)

        # 2D analysis via Kronecker structure
        analyze_2d_via_kronecker(args.size, args.size, args.order, args.boundary)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
For central difference gradient with CIRCULAR boundary:

1D case (size n):
  - Rank = n - 1
  - Null space = 1 (constant functions)
  - Invertible directions: n - 1

2D case (size m × n), separate Dx and Dy:
  - Dx rank = m(n-1), null dim = m
  - Dy rank = n(m-1), null dim = n

2D case, combined gradient [Dx; Dy]:
  - Rank = mn - 1
  - Null space = 1 (globally constant)
  - Invertible directions: mn - 1

For a 512×512 field:
  - Dx alone: 261,632 invertible / 262,144 total (99.80%)
  - Dy alone: 261,632 invertible / 262,144 total (99.80%)
  - Combined: 262,143 invertible / 262,144 total (99.9996%)
""")


if __name__ == '__main__':
    main()
