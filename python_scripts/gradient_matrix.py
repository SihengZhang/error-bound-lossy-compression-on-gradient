#!/usr/bin/env python3
"""
Generate central difference gradient operators as sparse matrices.

For a 2D field of size m×n, the gradient operator is a sparse matrix of size (m*n)×(m*n).
When applied to the flattened field (row-major), it produces the gradient.

Usage:
    python python_scripts/gradient_matrix.py <m> <n> <output_prefix> [--order ORDER] [--boundary BOUNDARY] [--direction DIRECTION]

Examples:
    # Generate both Dx and Dy matrices (512x512 field, order 8, circular boundary)
    python python_scripts/gradient_matrix.py 512 512 result/grad_matrix --order 8 --boundary circular

    # Generate only Dx matrix
    python python_scripts/gradient_matrix.py 512 512 result/grad_matrix_x --order 8 --boundary circular --direction x

    # Apply matrix to a field
    python python_scripts/gradient_matrix.py 512 512 result/grad --order 8 --boundary circular --apply data/field.f32.raw

Output:
    <output_prefix>_Dx.npz  - Sparse matrix for x-gradient (scipy sparse format)
    <output_prefix>_Dy.npz  - Sparse matrix for y-gradient (scipy sparse format)

    If --apply is used:
    <output_prefix>_grad_x.f32.raw  - Gradient in x direction
    <output_prefix>_grad_y.f32.raw  - Gradient in y direction
"""

import argparse
import numpy as np
from scipy import sparse
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from typing import Tuple, Dict


# Central difference coefficients for different orders
# Format: {order: coefficients} where coefficients are for [... -2, -1, 0, +1, +2, ...]
CENTRAL_DIFF_COEFFS: Dict[int, np.ndarray] = {
    2: np.array([-1/2, 0, 1/2]),
    4: np.array([1/12, -2/3, 0, 2/3, -1/12]),
    6: np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]),
    8: np.array([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]),
}


def get_stencil_offsets(order: int) -> np.ndarray:
    """Get the stencil offsets for a given order."""
    half_width = order // 2
    return np.arange(-half_width, half_width + 1)


def apply_boundary_condition(idx: int, size: int, boundary: str) -> Tuple[int, float]:
    """
    Apply boundary condition to an index.

    Args:
        idx: The index (may be out of bounds)
        size: The size of the dimension
        boundary: Boundary condition type

    Returns:
        (mapped_index, coefficient_multiplier)
        Returns (-1, 0) if the index should be ignored (zero boundary)
    """
    if 0 <= idx < size:
        return idx, 1.0

    if boundary == 'circular':
        return idx % size, 1.0
    elif boundary == 'symmetric':
        # Reflect at boundaries: [..., 2, 1, 0, 1, 2, ...] and [..., n-3, n-2, n-1, n-2, n-3, ...]
        if idx < 0:
            return -idx - 1, 1.0
        else:
            return 2 * size - idx - 1, 1.0
    elif boundary == 'replicate':
        # Clamp to boundary values
        return max(0, min(size - 1, idx)), 1.0
    elif boundary == 'zero':
        # Zero padding - ignore this contribution
        return -1, 0.0
    else:
        raise ValueError(f"Unknown boundary condition: {boundary}")


def build_1d_diff_matrix(size: int, order: int, boundary: str) -> csr_matrix:
    """
    Build a 1D central difference matrix.

    Args:
        size: Size of the 1D array
        order: Order of the central difference (2, 4, 6, or 8)
        boundary: Boundary condition ('circular', 'symmetric', 'replicate', 'zero')

    Returns:
        Sparse matrix of shape (size, size)
    """
    if order not in CENTRAL_DIFF_COEFFS:
        raise ValueError(f"Unsupported order: {order}. Supported: {list(CENTRAL_DIFF_COEFFS.keys())}")

    coeffs = CENTRAL_DIFF_COEFFS[order]
    offsets = get_stencil_offsets(order)

    # Use lil_matrix for efficient construction
    D = lil_matrix((size, size), dtype=np.float64)

    for i in range(size):
        for offset, coeff in zip(offsets, coeffs):
            if coeff == 0:
                continue
            j = i + offset
            mapped_j, mult = apply_boundary_condition(j, size, boundary)
            if mapped_j >= 0:
                D[i, mapped_j] += coeff * mult

    return D.tocsr()


def build_gradient_matrices(m: int, n: int, order: int, boundary: str) -> Tuple[csr_matrix, csr_matrix]:
    """
    Build 2D gradient matrices Dx and Dy for a field of size m×n.

    The field is assumed to be stored in row-major order (C order).
    For a field F[i, j] where i is row (y) and j is column (x):
    - Dx computes dF/dx (derivative along columns, j direction)
    - Dy computes dF/dy (derivative along rows, i direction)

    Args:
        m: Number of rows
        n: Number of columns
        order: Order of central difference
        boundary: Boundary condition

    Returns:
        (Dx, Dy) sparse matrices of shape (m*n, m*n)
    """
    # Build 1D difference matrices
    Dx_1d = build_1d_diff_matrix(n, order, boundary)  # For x direction (along columns)
    Dy_1d = build_1d_diff_matrix(m, order, boundary)  # For y direction (along rows)

    # Identity matrices
    Im = sparse.eye(m, format='csr')
    In = sparse.eye(n, format='csr')

    # Kronecker products to extend to 2D
    # For row-major ordering:
    # Dx = Im ⊗ Dx_1d (apply Dx_1d independently to each row)
    # Dy = Dy_1d ⊗ In (apply Dy_1d across rows)
    Dx = sparse.kron(Im, Dx_1d, format='csr')
    Dy = sparse.kron(Dy_1d, In, format='csr')

    return Dx, Dy


def print_matrix_info(name: str, M: csr_matrix):
    """Print information about a sparse matrix."""
    print(f"\n{name}:")
    print(f"  Shape: {M.shape}")
    print(f"  Non-zeros: {M.nnz} ({100*M.nnz/(M.shape[0]*M.shape[1]):.6f}%)")
    print(f"  Memory: {(M.data.nbytes + M.indices.nbytes + M.indptr.nbytes) / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Generate central difference gradient operators as sparse matrices.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('m', type=int, help='Number of rows')
    parser.add_argument('n', type=int, help='Number of columns')
    parser.add_argument('output_prefix', help='Output file prefix')
    parser.add_argument('--order', type=int, default=2, choices=[2, 4, 6, 8],
                        help='Order of central difference (default: 2)')
    parser.add_argument('--boundary', type=str, default='circular',
                        choices=['circular', 'symmetric', 'replicate', 'zero'],
                        help='Boundary condition (default: circular)')
    parser.add_argument('--direction', type=str, default='both', choices=['x', 'y', 'both'],
                        help='Which gradient direction(s) to compute (default: both)')
    parser.add_argument('--apply', type=str, default=None,
                        help='Apply the matrix to this input field (float32 raw)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the matrices (useful with --apply)')

    args = parser.parse_args()

    print(f"Building gradient matrices for {args.m}x{args.n} field")
    print(f"Order: {args.order}, Boundary: {args.boundary}")
    print(f"Matrix size: ({args.m*args.n}, {args.m*args.n})")

    # Build matrices
    Dx, Dy = None, None

    if args.direction in ['x', 'both']:
        print("\nBuilding Dx (x-gradient)...")
        Dx, _ = build_gradient_matrices(args.m, args.n, args.order, args.boundary)
        print_matrix_info("Dx", Dx)

        if not args.no_save:
            dx_path = f"{args.output_prefix}_Dx.npz"
            save_npz(dx_path, Dx)
            print(f"  Saved: {dx_path}")

    if args.direction in ['y', 'both']:
        print("\nBuilding Dy (y-gradient)...")
        _, Dy = build_gradient_matrices(args.m, args.n, args.order, args.boundary)
        print_matrix_info("Dy", Dy)

        if not args.no_save:
            dy_path = f"{args.output_prefix}_Dy.npz"
            save_npz(dy_path, Dy)
            print(f"  Saved: {dy_path}")

    # Apply to input field if specified
    if args.apply:
        print(f"\nApplying matrices to: {args.apply}")
        field = np.fromfile(args.apply, dtype=np.float32)
        if field.size != args.m * args.n:
            raise ValueError(f"Field size mismatch: expected {args.m*args.n}, got {field.size}")

        field = field.astype(np.float64)  # Use double precision for matrix multiply

        if Dx is not None:
            grad_x = Dx @ field
            grad_x_path = f"{args.output_prefix}_grad_x.f32.raw"
            grad_x.astype(np.float32).tofile(grad_x_path)
            print(f"  Wrote: {grad_x_path}")

        if Dy is not None:
            grad_y = Dy @ field
            grad_y_path = f"{args.output_prefix}_grad_y.f32.raw"
            grad_y.astype(np.float32).tofile(grad_y_path)
            print(f"  Wrote: {grad_y_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
