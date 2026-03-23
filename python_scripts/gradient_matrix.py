#!/usr/bin/env python3
"""
Generate differential operators as sparse matrices for 2D fields.

For a 2D field of size m×n, each operator is a sparse matrix of size (m*n)×(m*n).
When applied to the flattened field (row-major), it produces the corresponding derivative.

Supported Operations:
    dx      - First derivative in x direction (dF/dx)
    dy      - First derivative in y direction (dF/dy)
    dxx     - Second derivative in x direction (d²F/dx²)
    dyy     - Second derivative in y direction (d²F/dy²)
    dxy     - Mixed second derivative (d²F/dxdy)
    laplacian - Laplacian (d²F/dx² + d²F/dy²)

Supported Orders: 2, 4, 6, 8 (central difference accuracy)

Supported Boundary Conditions:
    circular  - Periodic boundary (wraps around)
    symmetric - Reflects at boundaries (mirror)
    replicate - Repeats edge values
    zero      - Zero padding outside domain

Usage:
    python python_scripts/gradient_matrix.py <m> <n> <output_dir> --op <operation> [options]

Examples:
    # Generate x-gradient matrix (512x512 field, order 8, circular boundary)
    python python_scripts/gradient_matrix.py 512 512 result/ --op dx --order 8 --boundary circular

    # Generate Laplacian matrix
    python python_scripts/gradient_matrix.py 256 256 result/ --op laplacian --order 4 --boundary symmetric

    # Generate all derivative matrices at once
    python python_scripts/gradient_matrix.py 128 128 result/ --op all --order 2 --boundary circular

    # Apply matrix to a field
    python python_scripts/gradient_matrix.py 512 512 result/ --op dx --order 8 --apply data/field.f32.raw

    # Run tests
    python python_scripts/gradient_matrix.py --test

Output Naming:
    matrix_<m>x<n>_<boundary>_<operation>_order<order>.npz

    Examples:
        matrix_512x512_circular_dx_order8.npz
        matrix_256x256_symmetric_laplacian_order4.npz
"""

import argparse
import numpy as np
from scipy import sparse
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from typing import Tuple, Dict, Optional, List
import os
import sys


# Central difference coefficients for first derivative
# Format: {order: coefficients} where coefficients are for stencil centered at 0
FIRST_DERIV_COEFFS: Dict[int, np.ndarray] = {
    2: np.array([-1/2, 0, 1/2]),
    4: np.array([1/12, -2/3, 0, 2/3, -1/12]),
    6: np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]),
    8: np.array([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]),
}

# Central difference coefficients for second derivative
# Format: {order: coefficients} where coefficients are for stencil centered at 0
SECOND_DERIV_COEFFS: Dict[int, np.ndarray] = {
    2: np.array([1, -2, 1]),
    4: np.array([-1/12, 4/3, -5/2, 4/3, -1/12]),
    6: np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]),
    8: np.array([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]),
}


def get_stencil_offsets(order: int, deriv_order: int = 1) -> np.ndarray:
    """Get the stencil offsets for a given accuracy order and derivative order."""
    if deriv_order == 1:
        half_width = order // 2
    else:  # second derivative
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


def build_1d_diff_matrix(size: int, order: int, boundary: str, deriv_order: int = 1) -> csr_matrix:
    """
    Build a 1D central difference matrix for first or second derivative.

    Args:
        size: Size of the 1D array
        order: Order of the central difference (2, 4, 6, or 8)
        boundary: Boundary condition ('circular', 'symmetric', 'replicate', 'zero')
        deriv_order: 1 for first derivative, 2 for second derivative

    Returns:
        Sparse matrix of shape (size, size)
    """
    if deriv_order == 1:
        coeffs_dict = FIRST_DERIV_COEFFS
    elif deriv_order == 2:
        coeffs_dict = SECOND_DERIV_COEFFS
    else:
        raise ValueError(f"Unsupported derivative order: {deriv_order}")

    if order not in coeffs_dict:
        raise ValueError(f"Unsupported order: {order}. Supported: {list(coeffs_dict.keys())}")

    coeffs = coeffs_dict[order]
    offsets = get_stencil_offsets(order, deriv_order)

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


def build_dx_matrix(m: int, n: int, order: int, boundary: str) -> csr_matrix:
    """
    Build x-gradient matrix Dx for a field of size m×n.

    Dx computes dF/dx (derivative along columns, j direction).
    """
    Dx_1d = build_1d_diff_matrix(n, order, boundary, deriv_order=1)
    Im = sparse.eye(m, format='csr')
    # Dx = Im ⊗ Dx_1d (apply Dx_1d independently to each row)
    return sparse.kron(Im, Dx_1d, format='csr')


def build_dy_matrix(m: int, n: int, order: int, boundary: str) -> csr_matrix:
    """
    Build y-gradient matrix Dy for a field of size m×n.

    Dy computes dF/dy (derivative along rows, i direction).
    """
    Dy_1d = build_1d_diff_matrix(m, order, boundary, deriv_order=1)
    In = sparse.eye(n, format='csr')
    # Dy = Dy_1d ⊗ In (apply Dy_1d across rows)
    return sparse.kron(Dy_1d, In, format='csr')


def build_dxx_matrix(m: int, n: int, order: int, boundary: str) -> csr_matrix:
    """
    Build second x-derivative matrix Dxx for a field of size m×n.

    Dxx computes d²F/dx².
    """
    Dxx_1d = build_1d_diff_matrix(n, order, boundary, deriv_order=2)
    Im = sparse.eye(m, format='csr')
    return sparse.kron(Im, Dxx_1d, format='csr')


def build_dyy_matrix(m: int, n: int, order: int, boundary: str) -> csr_matrix:
    """
    Build second y-derivative matrix Dyy for a field of size m×n.

    Dyy computes d²F/dy².
    """
    Dyy_1d = build_1d_diff_matrix(m, order, boundary, deriv_order=2)
    In = sparse.eye(n, format='csr')
    return sparse.kron(Dyy_1d, In, format='csr')


def build_dxy_matrix(m: int, n: int, order: int, boundary: str) -> csr_matrix:
    """
    Build mixed second derivative matrix Dxy for a field of size m×n.

    Dxy computes d²F/dxdy = Dx @ Dy = Dy @ Dx (commutes for smooth functions).
    """
    Dx = build_dx_matrix(m, n, order, boundary)
    Dy = build_dy_matrix(m, n, order, boundary)
    return Dx @ Dy


def build_laplacian_matrix(m: int, n: int, order: int, boundary: str) -> csr_matrix:
    """
    Build Laplacian matrix for a field of size m×n.

    Laplacian = d²F/dx² + d²F/dy² = Dxx + Dyy.
    """
    Dxx = build_dxx_matrix(m, n, order, boundary)
    Dyy = build_dyy_matrix(m, n, order, boundary)
    return Dxx + Dyy


# Map operation names to builder functions
OPERATION_BUILDERS = {
    'dx': build_dx_matrix,
    'dy': build_dy_matrix,
    'dxx': build_dxx_matrix,
    'dyy': build_dyy_matrix,
    'dxy': build_dxy_matrix,
    'laplacian': build_laplacian_matrix,
}

ALL_OPERATIONS = list(OPERATION_BUILDERS.keys())


def get_output_filename(output_dir: str, m: int, n: int, boundary: str, operation: str, order: int) -> str:
    """Generate the output filename according to naming convention."""
    filename = f"matrix_{m}x{n}_{boundary}_{operation}_order{order}.npz"
    return os.path.join(output_dir, filename)


def print_matrix_info(name: str, M: csr_matrix):
    """Print information about a sparse matrix."""
    print(f"\n{name}:")
    print(f"  Shape: {M.shape}")
    print(f"  Non-zeros: {M.nnz} ({100*M.nnz/(M.shape[0]*M.shape[1]):.6f}%)")
    print(f"  Memory: {(M.data.nbytes + M.indices.nbytes + M.indptr.nbytes) / 1024 / 1024:.2f} MB")


def run_tests():
    """Run comprehensive tests on the gradient matrix implementations."""
    print("=" * 60)
    print("Running Gradient Matrix Tests")
    print("=" * 60)

    all_passed = True

    # Test 1: Verify first derivative coefficients sum to zero
    print("\nTest 1: First derivative coefficients sum to zero")
    for order, coeffs in FIRST_DERIV_COEFFS.items():
        total = np.sum(coeffs)
        passed = np.abs(total) < 1e-14
        status = "PASS" if passed else "FAIL"
        print(f"  Order {order}: sum = {total:.2e} [{status}]")
        all_passed = all_passed and passed

    # Test 2: Verify second derivative coefficients sum to zero
    print("\nTest 2: Second derivative coefficients sum to zero")
    for order, coeffs in SECOND_DERIV_COEFFS.items():
        total = np.sum(coeffs)
        passed = np.abs(total) < 1e-14
        status = "PASS" if passed else "FAIL"
        print(f"  Order {order}: sum = {total:.2e} [{status}]")
        all_passed = all_passed and passed

    # Test 3: Apply dx to linear function f(x,y) = x should give constant 1 (interior only)
    # Note: Linear functions are not periodic, so we only check interior points
    print("\nTest 3: dx of f(x,y) = x should be 1 (interior points)")
    for order in [2, 4, 6, 8]:
        m, n = 32, 32
        Dx = build_dx_matrix(m, n, order, 'circular')
        # Create field f[i,j] = j (x-coordinate)
        x = np.arange(n, dtype=np.float64)
        field = np.tile(x, (m, 1)).flatten()
        result = Dx @ field
        expected = np.ones(m * n)
        # Use interior points only (linear function is not periodic)
        interior_mask = np.zeros((m, n), dtype=bool)
        interior_mask[order:m-order, order:n-order] = True
        interior_mask = interior_mask.flatten()
        max_error = np.max(np.abs(result[interior_mask] - expected[interior_mask]))
        passed = max_error < 1e-10
        status = "PASS" if passed else "FAIL"
        print(f"  Order {order}: max_error (interior) = {max_error:.2e} [{status}]")
        all_passed = all_passed and passed

    # Test 4: Apply dy to linear function f(x,y) = y should give constant 1 (interior only)
    print("\nTest 4: dy of f(x,y) = y should be 1 (interior points)")
    for order in [2, 4, 6, 8]:
        m, n = 32, 32
        Dy = build_dy_matrix(m, n, order, 'circular')
        # Create field f[i,j] = i (y-coordinate)
        y = np.arange(m, dtype=np.float64)
        field = np.repeat(y, n)
        result = Dy @ field
        expected = np.ones(m * n)
        # Use interior points only (linear function is not periodic)
        interior_mask = np.zeros((m, n), dtype=bool)
        interior_mask[order:m-order, order:n-order] = True
        interior_mask = interior_mask.flatten()
        max_error = np.max(np.abs(result[interior_mask] - expected[interior_mask]))
        passed = max_error < 1e-10
        status = "PASS" if passed else "FAIL"
        print(f"  Order {order}: max_error (interior) = {max_error:.2e} [{status}]")
        all_passed = all_passed and passed

    # Test 5: Apply dxx to quadratic function f(x,y) = x^2 should give constant 2
    print("\nTest 5: dxx of f(x,y) = x² should be 2 (circular boundary)")
    for order in [2, 4, 6, 8]:
        m, n = 32, 32
        Dxx = build_dxx_matrix(m, n, order, 'circular')
        # Create field f[i,j] = j^2
        x = np.arange(n, dtype=np.float64)
        field = np.tile(x**2, (m, 1)).flatten()
        result = Dxx @ field
        expected = 2.0 * np.ones(m * n)
        # Use interior points only (avoid boundary effects from x^2 not being periodic)
        interior_mask = np.zeros((m, n), dtype=bool)
        interior_mask[order:m-order, order:n-order] = True
        interior_mask = interior_mask.flatten()
        max_error = np.max(np.abs(result[interior_mask] - expected[interior_mask]))
        passed = max_error < 1e-8
        status = "PASS" if passed else "FAIL"
        print(f"  Order {order}: max_error (interior) = {max_error:.2e} [{status}]")
        all_passed = all_passed and passed

    # Test 6: Apply dyy to quadratic function f(x,y) = y^2 should give constant 2
    print("\nTest 6: dyy of f(x,y) = y² should be 2 (circular boundary)")
    for order in [2, 4, 6, 8]:
        m, n = 32, 32
        Dyy = build_dyy_matrix(m, n, order, 'circular')
        # Create field f[i,j] = i^2
        y = np.arange(m, dtype=np.float64)
        field = np.repeat(y**2, n)
        result = Dyy @ field
        expected = 2.0 * np.ones(m * n)
        # Use interior points only
        interior_mask = np.zeros((m, n), dtype=bool)
        interior_mask[order:m-order, order:n-order] = True
        interior_mask = interior_mask.flatten()
        max_error = np.max(np.abs(result[interior_mask] - expected[interior_mask]))
        passed = max_error < 1e-8
        status = "PASS" if passed else "FAIL"
        print(f"  Order {order}: max_error (interior) = {max_error:.2e} [{status}]")
        all_passed = all_passed and passed

    # Test 7: Test sinusoidal function with circular boundary (exact for spectral)
    print("\nTest 7: dx of sin(2πx/n) should be (2π/n)cos(2πx/n) (circular)")
    for order in [2, 4, 6, 8]:
        m, n = 64, 64
        Dx = build_dx_matrix(m, n, order, 'circular')
        # Create field f[i,j] = sin(2*pi*j/n)
        x = np.arange(n, dtype=np.float64)
        k = 2 * np.pi / n
        field = np.tile(np.sin(k * x), (m, 1)).flatten()
        result = Dx @ field
        expected = np.tile(k * np.cos(k * x), (m, 1)).flatten()
        max_error = np.max(np.abs(result - expected))
        # Error decreases with order
        tolerance = {2: 0.1, 4: 0.01, 6: 0.001, 8: 0.0001}[order]
        passed = max_error < tolerance
        status = "PASS" if passed else "FAIL"
        print(f"  Order {order}: max_error = {max_error:.2e} (tol={tolerance:.0e}) [{status}]")
        all_passed = all_passed and passed

    # Test 8: Laplacian of x^2 + y^2 should be 4
    print("\nTest 8: Laplacian of f(x,y) = x² + y² should be 4 (interior)")
    for order in [2, 4, 6, 8]:
        m, n = 32, 32
        L = build_laplacian_matrix(m, n, order, 'circular')
        # Create field f[i,j] = j^2 + i^2
        x = np.arange(n, dtype=np.float64)
        y = np.arange(m, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        field = (xx**2 + yy**2).flatten()
        result = L @ field
        expected = 4.0 * np.ones(m * n)
        # Use interior points only
        interior_mask = np.zeros((m, n), dtype=bool)
        interior_mask[order:m-order, order:n-order] = True
        interior_mask = interior_mask.flatten()
        max_error = np.max(np.abs(result[interior_mask] - expected[interior_mask]))
        passed = max_error < 1e-8
        status = "PASS" if passed else "FAIL"
        print(f"  Order {order}: max_error (interior) = {max_error:.2e} [{status}]")
        all_passed = all_passed and passed

    # Test 9: dxy of f(x,y) = xy should be 1
    print("\nTest 9: dxy of f(x,y) = xy should be 1 (interior)")
    for order in [2, 4, 6, 8]:
        m, n = 32, 32
        Dxy = build_dxy_matrix(m, n, order, 'circular')
        # Create field f[i,j] = i * j
        x = np.arange(n, dtype=np.float64)
        y = np.arange(m, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        field = (xx * yy).flatten()
        result = Dxy @ field
        expected = np.ones(m * n)
        # Use interior points only
        interior_mask = np.zeros((m, n), dtype=bool)
        interior_mask[order:m-order, order:n-order] = True
        interior_mask = interior_mask.flatten()
        max_error = np.max(np.abs(result[interior_mask] - expected[interior_mask]))
        passed = max_error < 1e-10
        status = "PASS" if passed else "FAIL"
        print(f"  Order {order}: max_error (interior) = {max_error:.2e} [{status}]")
        all_passed = all_passed and passed

    # Test 10: Test all boundary conditions don't crash
    print("\nTest 10: All boundary conditions work without error")
    for boundary in ['circular', 'symmetric', 'replicate', 'zero']:
        try:
            m, n = 16, 16
            for op_name, builder in OPERATION_BUILDERS.items():
                M = builder(m, n, 2, boundary)
                assert M.shape == (m*n, m*n)
            print(f"  {boundary}: PASS")
        except Exception as e:
            print(f"  {boundary}: FAIL ({e})")
            all_passed = False

    # Test 11: Non-square matrices
    print("\nTest 11: Non-square domains (m ≠ n)")
    test_sizes = [(32, 64), (64, 32), (17, 23)]
    for m, n in test_sizes:
        try:
            for op_name, builder in OPERATION_BUILDERS.items():
                M = builder(m, n, 4, 'circular')
                assert M.shape == (m*n, m*n), f"Wrong shape for {op_name}"
            print(f"  {m}x{n}: PASS")
        except Exception as e:
            print(f"  {m}x{n}: FAIL ({e})")
            all_passed = False

    # Test 12: Matrix symmetry properties
    print("\nTest 12: Matrix symmetry properties")
    m, n = 16, 16
    # Dxx and Dyy should be symmetric for circular boundary
    Dxx = build_dxx_matrix(m, n, 4, 'circular')
    Dyy = build_dyy_matrix(m, n, 4, 'circular')
    L = build_laplacian_matrix(m, n, 4, 'circular')

    dxx_sym_error = sparse.linalg.norm(Dxx - Dxx.T)
    dyy_sym_error = sparse.linalg.norm(Dyy - Dyy.T)
    lap_sym_error = sparse.linalg.norm(L - L.T)

    passed = dxx_sym_error < 1e-14 and dyy_sym_error < 1e-14 and lap_sym_error < 1e-14
    status = "PASS" if passed else "FAIL"
    print(f"  Dxx symmetry error: {dxx_sym_error:.2e}")
    print(f"  Dyy symmetry error: {dyy_sym_error:.2e}")
    print(f"  Laplacian symmetry error: {lap_sym_error:.2e} [{status}]")
    all_passed = all_passed and passed

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Generate differential operators as sparse matrices for 2D fields.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('m', type=int, nargs='?', help='Number of rows')
    parser.add_argument('n', type=int, nargs='?', help='Number of columns')
    parser.add_argument('output_dir', nargs='?', help='Output directory')
    parser.add_argument('--op', type=str, default='all',
                        choices=ALL_OPERATIONS + ['all'],
                        help='Operation type (default: all)')
    parser.add_argument('--order', type=int, default=2, choices=[2, 4, 6, 8],
                        help='Order of central difference (default: 2)')
    parser.add_argument('--boundary', type=str, default='circular',
                        choices=['circular', 'symmetric', 'replicate', 'zero'],
                        help='Boundary condition (default: circular)')
    parser.add_argument('--apply', type=str, default=None,
                        help='Apply the matrix to this input field (float32 raw)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the matrices (useful with --apply)')
    parser.add_argument('--test', action='store_true',
                        help='Run tests instead of generating matrices')

    args = parser.parse_args()

    # Run tests if requested
    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    # Validate required arguments
    if args.m is None or args.n is None or args.output_dir is None:
        parser.error("m, n, and output_dir are required (unless --test is specified)")

    # Create output directory if needed
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"Building differential operator matrices for {args.m}x{args.n} field")
    print(f"Order: {args.order}, Boundary: {args.boundary}")
    print(f"Matrix size: ({args.m*args.n}, {args.m*args.n})")

    # Determine which operations to build
    operations = ALL_OPERATIONS if args.op == 'all' else [args.op]

    matrices = {}
    for op_name in operations:
        print(f"\nBuilding {op_name}...")
        builder = OPERATION_BUILDERS[op_name]
        M = builder(args.m, args.n, args.order, args.boundary)
        matrices[op_name] = M
        print_matrix_info(op_name, M)

        if not args.no_save:
            output_path = get_output_filename(args.output_dir, args.m, args.n,
                                              args.boundary, op_name, args.order)
            save_npz(output_path, M)
            print(f"  Saved: {output_path}")

    # Apply to input field if specified
    if args.apply:
        print(f"\nApplying matrices to: {args.apply}")
        field = np.fromfile(args.apply, dtype=np.float32)
        if field.size != args.m * args.n:
            raise ValueError(f"Field size mismatch: expected {args.m*args.n}, got {field.size}")

        field = field.astype(np.float64)  # Use double precision for matrix multiply

        for op_name, M in matrices.items():
            result = M @ field
            result_path = os.path.join(args.output_dir,
                                       f"result_{args.m}x{args.n}_{args.boundary}_{op_name}_order{args.order}.f32.raw")
            result.astype(np.float32).tofile(result_path)
            print(f"  Wrote: {result_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
