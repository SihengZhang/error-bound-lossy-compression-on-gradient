#!/usr/bin/env python3
"""
Project error field onto feasible region satisfying both space and gradient error bounds.

Supports both L1 norm (Linear Programming) and L2 norm (Quadratic Programming) minimization.

Problem Formulation:
    minimize    ||x^ - x*||_p    (p = 1 for L1, p = 2 for L2)
    subject to  |x^ - x| ≤ b    (space error bound, per-pixel)
                |Ax^ - Ax| ≤ c  (gradient error bound, per-pixel)

    Equivalently, with e = x^ - x:
    minimize    ||e - e*||_p    where e* = x* - x
    subject to  -b ≤ e ≤ b
                -c ≤ Ae ≤ c

Usage:
    python python_scripts/direct_projection.py <ground_truth> <error_field> <m> <n> <matrix.npz> \\
        --space-rel <rel> --grad-rel <rel> [--norm L1|L2] [options]

Examples:
    # L2 projection (default) - minimizes sum of squared differences
    python python_scripts/direct_projection.py data/field.raw data/field_compressed.raw 512 512 \\
        result/matrices/matrix_512x512_circular_dy_order8.npz \\
        --space-rel 1e-3 --grad-rel 1e-3

    # L1 projection - minimizes sum of absolute differences (more sparse changes)
    python python_scripts/direct_projection.py data/field.raw data/field_compressed.raw 512 512 \\
        result/matrices/matrix_512x512_circular_dy_order8.npz \\
        --space-rel 1e-3 --grad-rel 1e-3 --norm L1

Output (in output directory):
    ground_truth_gradient.f32.raw     - Ax (ground truth gradient)
    error_gradient.f32.raw            - Ax* (gradient of error field)
    gradient_diff.f32.raw             - Ax* - Ax (gradient difference)
    gradient_oob_mask.f32.raw         - Out-of-bound mask for gradient (1.0=exceeded)
    projected_space.f32.raw           - x^ (projected space field)
    projected_gradient.f32.raw        - Ax^ (projected gradient)
    space_change_mask.f32.raw         - Mask of changed pixels |x^ - x*| > 0
    projected_space_error.f32.raw     - x^ - x (projected error in space)
    projected_gradient_error.f32.raw  - Ax^ - Ax (projected error in gradient)
    projection_stats.txt              - Statistics summary
"""

import argparse
import numpy as np
from scipy.sparse import load_npz, csr_matrix, eye, vstack, hstack, diags
from scipy.optimize import linprog
import os
import time

# Try to import OSQP for QP solving
try:
    import osqp
    from scipy import sparse
    HAS_OSQP = True
except ImportError:
    HAS_OSQP = False

# Try to import CVXPY as alternative QP solver
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


def read_raw_float32(filepath: str, m: int, n: int) -> np.ndarray:
    """Read raw binary file as float32 array."""
    data = np.fromfile(filepath, dtype=np.float32)
    if data.size != m * n:
        raise ValueError(f"File size mismatch: expected {m*n} floats, got {data.size}")
    return data.astype(np.float64)  # Use double precision for computation


def write_raw_float32(filepath: str, data: np.ndarray):
    """Write array to raw binary file as float32."""
    data.astype(np.float32).tofile(filepath)


def solve_lp_scipy(e_star: np.ndarray, A: csr_matrix,
                   b: np.ndarray, c: np.ndarray,
                   verbose: bool = True) -> np.ndarray:
    """
    Solve the L1 projection using Linear Programming (scipy.linprog with HiGHS).

    minimize    ||e - e*||₁ = Σ|eᵢ - e*ᵢ|
    subject to  -b ≤ e ≤ b
                -c ≤ Ae ≤ c

    Reformulate with slack variables t:
    minimize    Σtᵢ
    subject to  e - e* ≤ t       (upper bound on |e - e*|)
                -(e - e*) ≤ t    (lower bound on |e - e*|)
                -b ≤ e ≤ b       (space constraints)
                -c ≤ Ae ≤ c      (gradient constraints)
                t ≥ 0

    Variables: [e; t] where e ∈ ℝⁿ, t ∈ ℝⁿ (total 2n variables)
    """
    n = len(e_star)
    m_grad = A.shape[0]

    if verbose:
        print("Solving LP with scipy.linprog (HiGHS)...")
        print(f"  Variables: {2*n} (e: {n}, t: {n})")
        print(f"  Constraints: {2*n + 2*n + 2*m_grad} inequality")

    # Objective: minimize Σtᵢ
    # c = [0...0, 1...1] for [e; t]
    obj = np.concatenate([np.zeros(n), np.ones(n)])

    # Build constraint matrix A_ub @ x <= b_ub
    # Constraints:
    # 1. e - e* <= t  =>  [I, -I][e; t] <= e*
    # 2. -(e - e*) <= t  =>  [-I, -I][e; t] <= -e*
    # 3. e <= b  =>  [I, 0][e; t] <= b
    # 4. -e <= b  =>  [-I, 0][e; t] <= b
    # 5. Ae <= c  =>  [A, 0][e; t] <= c
    # 6. -Ae <= c  =>  [-A, 0][e; t] <= c

    I_n = eye(n, format='csr')
    Z_n = csr_matrix((n, n))
    Z_grad = csr_matrix((m_grad, n))

    # Stack constraint matrices
    A_ub = vstack([
        hstack([I_n, -I_n]),      # e - t <= e*
        hstack([-I_n, -I_n]),     # -e - t <= -e*
        hstack([I_n, Z_n]),       # e <= b
        hstack([-I_n, Z_n]),      # -e <= b
        hstack([A, Z_grad]),      # Ae <= c
        hstack([-A, Z_grad]),     # -Ae <= c
    ], format='csr')

    b_ub = np.concatenate([
        e_star,     # e - t <= e*
        -e_star,    # -e - t <= -e*
        b,          # e <= b
        b,          # -e <= b
        c,          # Ae <= c
        c,          # -Ae <= c
    ])

    # Bounds: e unbounded (handled by constraints), t >= 0
    bounds = [(None, None)] * n + [(0, None)] * n

    start_time = time.time()
    result = linprog(
        obj,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method='highs',
        options={'disp': verbose, 'presolve': True}
    )
    solve_time = time.time() - start_time

    if verbose:
        print(f"  Status: {result.message}")
        print(f"  Solve time: {solve_time:.2f}s")
        print(f"  Iterations: {result.nit}")

    if not result.success:
        raise RuntimeError(f"LP failed: {result.message}")

    # Extract e from solution [e; t]
    return result.x[:n]


def solve_qp_osqp(e_star: np.ndarray, A: csr_matrix,
                  b: np.ndarray, c: np.ndarray,
                  verbose: bool = True) -> np.ndarray:
    """
    Solve the L2 projection using Quadratic Programming (OSQP).

    OSQP standard form:
        minimize    (1/2)x'Px + q'x
        subject to  l ≤ Gx ≤ u

    Our problem:
        minimize    ||e - e*||₂² = e'e - 2e*'e + const
        subject to  -b ≤ e ≤ b
                    -c ≤ Ae ≤ c

    So: P = 2I, q = -2e*, G = [I; A], l = [-b; -c], u = [b; c]
    """
    n = len(e_star)
    m_grad = A.shape[0]

    if verbose:
        print("Solving QP with OSQP...")
        print(f"  Variables: {n}")
        print(f"  Constraints: {n + m_grad}")

    # Objective: P = 2I, q = -2e*
    P = sparse.eye(n, format='csc') * 2.0
    q = -2.0 * e_star

    # Constraints: [I; A] with bounds [-b, b] and [-c, c]
    G = sparse.vstack([sparse.eye(n, format='csc'), A], format='csc')
    l = np.concatenate([-b, -c])
    u = np.concatenate([b, c])

    # Create OSQP problem
    prob = osqp.OSQP()
    prob.setup(P, q, G, l, u,
               verbose=verbose,
               eps_abs=1e-6,
               eps_rel=1e-6,
               max_iter=10000,
               polish=True)

    start_time = time.time()
    result = prob.solve()
    solve_time = time.time() - start_time

    if verbose:
        print(f"  Status: {result.info.status}")
        print(f"  Solve time: {solve_time:.2f}s")
        print(f"  Iterations: {result.info.iter}")

    if result.info.status != 'solved' and result.info.status != 'solved_inaccurate':
        raise RuntimeError(f"OSQP failed: {result.info.status}")

    return result.x


def solve_qp_cvxpy(e_star: np.ndarray, A: csr_matrix,
                   b: np.ndarray, c: np.ndarray,
                   verbose: bool = True) -> np.ndarray:
    """
    Solve the L2 projection using CVXPY (fallback QP solver).

    minimize    ||e - e*||₂²
    subject to  -b ≤ e ≤ b
                -c ≤ Ae ≤ c
    """
    n = len(e_star)

    # Define variable
    e = cp.Variable(n)

    # Objective: minimize ||e - e*||₂²
    objective = cp.Minimize(cp.sum_squares(e - e_star))

    # Constraints
    constraints = [
        e >= -b,           # -b ≤ e
        e <= b,            # e ≤ b
        A @ e >= -c,       # -c ≤ Ae
        A @ e <= c,        # Ae ≤ c
    ]

    # Solve
    problem = cp.Problem(objective, constraints)

    if verbose:
        print("Solving QP with CVXPY...")

    # Try different solvers
    solvers_to_try = [cp.OSQP, cp.SCS, cp.ECOS]

    for solver in solvers_to_try:
        try:
            start_time = time.time()
            problem.solve(solver=solver, verbose=verbose)
            solve_time = time.time() - start_time

            if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
                if verbose:
                    print(f"  Solver: {solver}")
                    print(f"  Status: {problem.status}")
                    print(f"  Solve time: {solve_time:.2f}s")
                    print(f"  Optimal value: {problem.value:.6e}")
                return e.value
        except Exception as ex:
            if verbose:
                print(f"  Solver {solver} failed: {ex}")
            continue

    raise RuntimeError("All solvers failed to find a solution")


def compute_error_bounds(ground_truth: np.ndarray,
                         ground_truth_grad: np.ndarray,
                         space_rel: float = None, space_abs: float = None,
                         grad_rel: float = None, grad_abs: float = None):
    """
    Compute per-pixel error bounds from relative or absolute specifications.

    REL bound: |error| ≤ rel × (max - min)
    """
    # Space error bound
    if space_abs is not None:
        b = np.full(len(ground_truth), space_abs)
    elif space_rel is not None:
        space_range = ground_truth.max() - ground_truth.min()
        b = np.full(len(ground_truth), space_rel * space_range)
    else:
        raise ValueError("Must specify either space_rel or space_abs")

    # Gradient error bound
    if grad_abs is not None:
        c = np.full(len(ground_truth_grad), grad_abs)
    elif grad_rel is not None:
        grad_range = ground_truth_grad.max() - ground_truth_grad.min()
        c = np.full(len(ground_truth_grad), grad_rel * grad_range)
    else:
        raise ValueError("Must specify either grad_rel or grad_abs")

    return b, c


def main():
    parser = argparse.ArgumentParser(
        description='Project error field onto feasible region using LP (L1) or QP (L2).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('ground_truth', help='Path to ground truth field (float32 raw)')
    parser.add_argument('error_field', help='Path to error field x* (float32 raw)')
    parser.add_argument('m', type=int, help='Number of rows')
    parser.add_argument('n', type=int, help='Number of columns')
    parser.add_argument('matrix', help='Path to gradient matrix A (.npz)')

    # Error bounds (must specify one of each pair)
    parser.add_argument('--space-rel', type=float, default=None,
                        help='Relative space error bound (REL × value_range)')
    parser.add_argument('--space-abs', type=float, default=None,
                        help='Absolute space error bound')
    parser.add_argument('--grad-rel', type=float, default=None,
                        help='Relative gradient error bound (REL × gradient_range)')
    parser.add_argument('--grad-abs', type=float, default=None,
                        help='Absolute gradient error bound')

    # Norm selection
    parser.add_argument('--norm', type=str, default='L2', choices=['L1', 'L2'],
                        help='Norm to minimize: L1 (linear programming) or L2 (quadratic programming, default)')

    # Options
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: result/L1 or result/L2)')
    parser.add_argument('--solver', type=str, default='auto',
                        choices=['auto', 'cvxpy', 'osqp', 'highs'],
                        help='Solver to use (default: auto). highs only for L1, osqp/cvxpy for L2.')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()
    verbose = not args.quiet

    # Set default output directory based on norm
    if args.output_dir is None:
        args.output_dir = f'result/{args.norm}'

    # Validate error bound arguments
    if args.space_rel is None and args.space_abs is None:
        parser.error("Must specify either --space-rel or --space-abs")
    if args.grad_rel is None and args.grad_abs is None:
        parser.error("Must specify either --grad-rel or --grad-abs")

    # Check solver availability
    if args.norm == 'L2':
        if args.solver == 'osqp' and not HAS_OSQP:
            parser.error("OSQP not installed. Install with: pip install osqp")
        if args.solver == 'cvxpy' and not HAS_CVXPY:
            parser.error("CVXPY not installed. Install with: pip install cvxpy")
        if args.solver == 'auto' and not HAS_OSQP and not HAS_CVXPY:
            parser.error("No QP solver available. Install osqp or cvxpy")
        if args.solver == 'highs':
            parser.error("HiGHS solver only available for L1 norm")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Read inputs
    if verbose:
        print(f"Reading ground truth: {args.ground_truth}")
    x = read_raw_float32(args.ground_truth, args.m, args.n)

    if verbose:
        print(f"Reading error field: {args.error_field}")
    x_star = read_raw_float32(args.error_field, args.m, args.n)

    if verbose:
        print(f"Loading gradient matrix: {args.matrix}")
    A = load_npz(args.matrix)

    if A.shape[0] != args.m * args.n or A.shape[1] != args.m * args.n:
        raise ValueError(f"Matrix shape {A.shape} doesn't match field size {args.m}x{args.n}")

    # Compute ground truth gradient
    if verbose:
        print("Computing ground truth gradient Ax...")
    Ax = A @ x

    # Compute error field gradient
    if verbose:
        print("Computing error gradient Ax*...")
    Ax_star = A @ x_star

    # Compute per-pixel error bounds
    if verbose:
        print("Computing error bounds...")
    b, c = compute_error_bounds(
        x, Ax,
        space_rel=args.space_rel, space_abs=args.space_abs,
        grad_rel=args.grad_rel, grad_abs=args.grad_abs
    )

    if verbose:
        print(f"  Space error bound: {b[0]:.6e} (uniform)")
        print(f"  Gradient error bound: {c[0]:.6e} (uniform)")

    # Compute initial errors
    e_star = x_star - x  # Initial error in space
    Ae_star = Ax_star - Ax  # Initial error in gradient

    space_violations_before = np.sum(np.abs(e_star) > b)
    grad_violations_before = np.sum(np.abs(Ae_star) > c)

    if verbose:
        print(f"\n=== Before Projection ===")
        print(f"  Space violations: {space_violations_before} / {len(e_star)} "
              f"({100*space_violations_before/len(e_star):.4f}%)")
        print(f"  Gradient violations: {grad_violations_before} / {len(Ae_star)} "
              f"({100*grad_violations_before/len(Ae_star):.4f}%)")
        print(f"  Max space error: {np.abs(e_star).max():.6e} (bound: {b[0]:.6e})")
        print(f"  Max gradient error: {np.abs(Ae_star).max():.6e} (bound: {c[0]:.6e})")

    # Check if already feasible
    if space_violations_before == 0 and grad_violations_before == 0:
        if verbose:
            print("\nField already satisfies all constraints. No projection needed.")
        e_proj = e_star
    else:
        # Solve optimization problem
        if verbose:
            print(f"\n=== Solving {args.norm} Projection ===")

        if args.norm == 'L1':
            # Linear Programming for L1 norm
            e_proj = solve_lp_scipy(e_star, A, b, c, verbose=verbose)
        else:
            # Quadratic Programming for L2 norm
            if args.solver == 'osqp' or (args.solver == 'auto' and HAS_OSQP):
                e_proj = solve_qp_osqp(e_star, A, b, c, verbose=verbose)
            else:
                e_proj = solve_qp_cvxpy(e_star, A, b, c, verbose=verbose)

    # Compute projected field
    x_proj = x + e_proj
    Ax_proj = A @ x_proj

    # Compute post-projection stats
    Ae_proj = Ax_proj - Ax
    space_violations_after = np.sum(np.abs(e_proj) > b * (1 + 1e-6))  # Small tolerance
    grad_violations_after = np.sum(np.abs(Ae_proj) > c * (1 + 1e-6))

    # Compute change mask
    change_threshold = 1e-10 * np.abs(x_star).max()
    change_mask = (np.abs(x_proj - x_star) > change_threshold).astype(np.float32)
    pixels_changed = int(change_mask.sum())

    # Compute projection distances
    l1_distance = np.sum(np.abs(x_proj - x_star))
    l2_distance = np.linalg.norm(x_proj - x_star)

    if verbose:
        print(f"\n=== After Projection ===")
        print(f"  Space violations: {space_violations_after}")
        print(f"  Gradient violations: {grad_violations_after}")
        print(f"  Max space error: {np.abs(e_proj).max():.6e} (bound: {b[0]:.6e})")
        print(f"  Max gradient error: {np.abs(Ae_proj).max():.6e} (bound: {c[0]:.6e})")
        print(f"  Pixels changed: {pixels_changed} / {len(x_star)} "
              f"({100*pixels_changed/len(x_star):.4f}%)")
        print(f"  Projection L1 distance: {l1_distance:.6e}")
        print(f"  Projection L2 distance: {l2_distance:.6e}")

    # Write outputs
    if verbose:
        print(f"\n=== Writing Outputs to {args.output_dir} ===")

    # Ground truth gradient
    path = os.path.join(args.output_dir, "ground_truth_gradient.f32.raw")
    write_raw_float32(path, Ax)
    if verbose:
        print(f"  {path}")

    # Error gradient
    path = os.path.join(args.output_dir, "error_gradient.f32.raw")
    write_raw_float32(path, Ax_star)
    if verbose:
        print(f"  {path}")

    # Gradient diff
    path = os.path.join(args.output_dir, "gradient_diff.f32.raw")
    write_raw_float32(path, Ax_star - Ax)
    if verbose:
        print(f"  {path}")

    # Gradient out-of-bound mask
    grad_oob = (np.abs(Ae_star) > c).astype(np.float32)
    path = os.path.join(args.output_dir, "gradient_oob_mask.f32.raw")
    write_raw_float32(path, grad_oob)
    if verbose:
        print(f"  {path}")

    # Projected space
    path = os.path.join(args.output_dir, "projected_space.f32.raw")
    write_raw_float32(path, x_proj)
    if verbose:
        print(f"  {path}")

    # Projected gradient
    path = os.path.join(args.output_dir, "projected_gradient.f32.raw")
    write_raw_float32(path, Ax_proj)
    if verbose:
        print(f"  {path}")

    # Space change mask
    path = os.path.join(args.output_dir, "space_change_mask.f32.raw")
    write_raw_float32(path, change_mask)
    if verbose:
        print(f"  {path}")

    # Projected space error (x^ - x)
    path = os.path.join(args.output_dir, "projected_space_error.f32.raw")
    write_raw_float32(path, e_proj)
    if verbose:
        print(f"  {path}")

    # Projected gradient error (Ax^ - Ax)
    path = os.path.join(args.output_dir, "projected_gradient_error.f32.raw")
    write_raw_float32(path, Ae_proj)
    if verbose:
        print(f"  {path}")

    # Write statistics to file
    stats_path = os.path.join(args.output_dir, "projection_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"=== {args.norm} Projection Statistics ===\n\n")
        f.write(f"Input:\n")
        f.write(f"  Ground truth: {args.ground_truth}\n")
        f.write(f"  Error field: {args.error_field}\n")
        f.write(f"  Dimensions: {args.m} x {args.n}\n")
        f.write(f"  Matrix: {args.matrix}\n")
        f.write(f"  Norm: {args.norm}\n")
        f.write(f"\nError Bounds:\n")
        if args.space_rel is not None:
            f.write(f"  Space: REL {args.space_rel} = {b[0]:.6e} absolute\n")
        else:
            f.write(f"  Space: ABS {args.space_abs}\n")
        if args.grad_rel is not None:
            f.write(f"  Gradient: REL {args.grad_rel} = {c[0]:.6e} absolute\n")
        else:
            f.write(f"  Gradient: ABS {args.grad_abs}\n")
        f.write(f"\nBefore Projection:\n")
        f.write(f"  Space violations: {space_violations_before} ({100*space_violations_before/len(e_star):.4f}%)\n")
        f.write(f"  Gradient violations: {grad_violations_before} ({100*grad_violations_before/len(Ae_star):.4f}%)\n")
        f.write(f"  Max space error: {np.abs(e_star).max():.6e}\n")
        f.write(f"  Max gradient error: {np.abs(Ae_star).max():.6e}\n")
        f.write(f"\nAfter Projection:\n")
        f.write(f"  Space violations: {space_violations_after}\n")
        f.write(f"  Gradient violations: {grad_violations_after}\n")
        f.write(f"  Max space error: {np.abs(e_proj).max():.6e}\n")
        f.write(f"  Max gradient error: {np.abs(Ae_proj).max():.6e}\n")
        f.write(f"  Pixels changed: {pixels_changed} ({100*pixels_changed/len(x_star):.4f}%)\n")
        f.write(f"  Projection L1 distance: {l1_distance:.6e}\n")
        f.write(f"  Projection L2 distance: {l2_distance:.6e}\n")

    if verbose:
        print(f"  {stats_path}")
        print("\nDone!")


if __name__ == '__main__':
    main()
