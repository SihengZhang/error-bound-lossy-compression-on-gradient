#!/usr/bin/env python3
"""
Project error field onto feasible region with multiple operator constraints.

Implements Lemma 4.6 (Combined Spatial and Multiple Output Error Bounds):
For m linear operators A_1, ..., A_m with per-operator error bounds, find the
corrected error vector that minimizes modification while satisfying all constraints.

Supports both L1 norm (Linear Programming) and L2 norm (Quadratic Programming) minimization.

Problem Formulation:
    minimize    ||e - e*||_p    (p = 1 for L1, p = 2 for L2)
    subject to  -b ≤ e ≤ b             (space error bound, per-pixel)
                -c^(k) ≤ A_k e ≤ c^(k)  (gradient error bound for operator k)
                for k ∈ {1, ..., m}

Configuration File Format (JSON):
{
  "ground_truth": "path/to/ground_truth.raw",
  "error_field": "path/to/error_field.raw",
  "dimensions": {"m": 512, "n": 512},
  "space_bounds": {"rel": 1e-3},
  "operators": [
    {
      "name": "dx",
      "matrix": "path/to/matrix_dx.npz",
      "bounds": {"rel": 1e-3}
    },
    {
      "name": "dy",
      "matrix": "path/to/matrix_dy.npz",
      "bounds": {"rel": 1e-3}
    }
  ],
  "norm": "L2",
  "output_dir": "result/L2_multi",
  "solver": "auto",
  "verbose": true
}

Usage:
    python python_scripts/multi_operator_projection.py config.json

Example config files:
    # Two gradient operators (dx and dy)
    python python_scripts/multi_operator_projection.py configs/projection_dx_dy.json

    # Single operator (backward compatible)
    python python_scripts/multi_operator_projection.py configs/projection_dy.json
"""

import argparse
import numpy as np
from scipy.sparse import load_npz, csr_matrix, eye, vstack, hstack, diags
from scipy.optimize import linprog
import os
import json
import time
from typing import Dict, List, Tuple

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


def solve_lp_scipy_multi(e_star: np.ndarray,
                         operators: List[Dict],
                         b: np.ndarray,
                         verbose: bool = True) -> np.ndarray:
    """
    Solve the L1 projection with multiple operator constraints using Linear Programming.

    minimize    Σtᵢ
    subject to  e - e* ≤ t       (upper bound on |e - e*|)
                -(e - e*) ≤ t    (lower bound on |e - e*|)
                -b ≤ e ≤ b       (space constraints)
                -c^(k) ≤ A_k e ≤ c^(k)  (operator k constraints, k=1..m)
                t ≥ 0

    Variables: [e; t] where e ∈ ℝⁿ, t ∈ ℝⁿ (total 2n variables)

    Args:
        e_star: Initial error vector (n,)
        operators: List of dicts with keys 'matrix' (csr_matrix), 'bounds' (ndarray), 'name' (str)
        b: Space error bounds (n,)
        verbose: Print progress
    """
    n = len(e_star)
    m = len(operators)

    # Total number of gradient constraints
    total_grad_constraints = sum(op['matrix'].shape[0] for op in operators)

    if verbose:
        print("Solving LP with scipy.linprog (HiGHS)...")
        print(f"  Variables: {2*n} (e: {n}, t: {n})")
        print(f"  Operators: {m}")
        print(f"  Constraints: {2*n + 2*n + 2*total_grad_constraints} inequality")

    # Objective: minimize Σtᵢ
    # c = [0...0, 1...1] for [e; t]
    obj = np.concatenate([np.zeros(n), np.ones(n)])

    # Build constraint matrix A_ub @ x <= b_ub
    # Constraints:
    # 1. e - e* <= t  =>  [I, -I][e; t] <= e*
    # 2. -(e - e*) <= t  =>  [-I, -I][e; t] <= -e*
    # 3. e <= b  =>  [I, 0][e; t] <= b
    # 4. -e <= b  =>  [-I, 0][e; t] <= b
    # 5. A_k e <= c^(k)  =>  [A_k, 0][e; t] <= c^(k)  for all k
    # 6. -A_k e <= c^(k)  =>  [-A_k, 0][e; t] <= c^(k)  for all k

    I_n = eye(n, format='csr')
    Z_n = csr_matrix((n, n))

    # Start with basic constraints
    constraint_matrices = [
        hstack([I_n, -I_n]),      # e - t <= e*
        hstack([-I_n, -I_n]),     # -e - t <= -e*
        hstack([I_n, Z_n]),       # e <= b
        hstack([-I_n, Z_n]),      # -e <= b
    ]

    constraint_bounds = [
        e_star,     # e - t <= e*
        -e_star,    # -e - t <= -e*
        b,          # e <= b
        b,          # -e <= b
    ]

    # Add operator constraints
    for op in operators:
        A_k = op['matrix']
        c_k = op['bounds']
        m_k = A_k.shape[0]
        Z_grad_k = csr_matrix((m_k, n))

        constraint_matrices.append(hstack([A_k, Z_grad_k]))    # A_k e <= c^(k)
        constraint_matrices.append(hstack([-A_k, Z_grad_k]))   # -A_k e <= c^(k)
        constraint_bounds.append(c_k)
        constraint_bounds.append(c_k)

    # Stack all constraints
    A_ub = vstack(constraint_matrices, format='csr')
    b_ub = np.concatenate(constraint_bounds)

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


def solve_qp_osqp_multi(e_star: np.ndarray,
                        operators: List[Dict],
                        b: np.ndarray,
                        verbose: bool = True) -> np.ndarray:
    """
    Solve the L2 projection with multiple operator constraints using OSQP.

    OSQP standard form:
        minimize    (1/2)x'Px + q'x
        subject to  l ≤ Gx ≤ u

    Our problem:
        minimize    ||e - e*||₂² = e'e - 2e*'e + const
        subject to  -b ≤ e ≤ b
                    -c^(k) ≤ A_k e ≤ c^(k)  for all k

    So: P = 2I, q = -2e*, G = [I; A_1; A_2; ...; A_m],
        l = [-b; -c^(1); -c^(2); ...; -c^(m)],
        u = [b; c^(1); c^(2); ...; c^(m)]
    """
    n = len(e_star)
    m = len(operators)
    total_grad_constraints = sum(op['matrix'].shape[0] for op in operators)

    if verbose:
        print("Solving QP with OSQP...")
        print(f"  Variables: {n}")
        print(f"  Operators: {m}")
        print(f"  Constraints: {n + total_grad_constraints}")

    # Objective: P = 2I, q = -2e*
    P = sparse.eye(n, format='csc') * 2.0
    q = -2.0 * e_star

    # Constraints: [I; A_1; A_2; ...; A_m] with bounds
    constraint_matrices = [sparse.eye(n, format='csc')]
    lower_bounds = [-b]
    upper_bounds = [b]

    for op in operators:
        constraint_matrices.append(op['matrix'])
        lower_bounds.append(-op['bounds'])
        upper_bounds.append(op['bounds'])

    G = sparse.vstack(constraint_matrices, format='csc')
    l = np.concatenate(lower_bounds)
    u = np.concatenate(upper_bounds)

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


def solve_qp_cvxpy_multi(e_star: np.ndarray,
                         operators: List[Dict],
                         b: np.ndarray,
                         verbose: bool = True) -> np.ndarray:
    """
    Solve the L2 projection with multiple operator constraints using CVXPY.

    minimize    ||e - e*||₂²
    subject to  -b ≤ e ≤ b
                -c^(k) ≤ A_k e ≤ c^(k)  for all k
    """
    n = len(e_star)
    m = len(operators)

    # Define variable
    e = cp.Variable(n)

    # Objective: minimize ||e - e*||₂²
    objective = cp.Minimize(cp.sum_squares(e - e_star))

    # Constraints
    constraints = [
        e >= -b,           # -b ≤ e
        e <= b,            # e ≤ b
    ]

    # Add operator constraints
    for op in operators:
        A_k = op['matrix']
        c_k = op['bounds']
        constraints.append(A_k @ e >= -c_k)  # -c^(k) ≤ A_k e
        constraints.append(A_k @ e <= c_k)   # A_k e ≤ c^(k)

    # Solve
    problem = cp.Problem(objective, constraints)

    if verbose:
        print("Solving QP with CVXPY...")
        print(f"  Variables: {n}")
        print(f"  Operators: {m}")

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


def compute_error_bounds(reference_data: np.ndarray,
                         rel: float = None,
                         abs_val: float = None) -> np.ndarray:
    """
    Compute per-pixel error bounds from relative or absolute specifications.

    REL bound: |error| ≤ rel × (max - min)
    """
    if abs_val is not None:
        return np.full(len(reference_data), abs_val)
    elif rel is not None:
        data_range = reference_data.max() - reference_data.min()
        return np.full(len(reference_data), rel * data_range)
    else:
        raise ValueError("Must specify either 'rel' or 'abs' in bounds")


def load_config(config_path: str) -> Dict:
    """Load and validate JSON configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validate required fields
    required = ['ground_truth', 'error_field', 'dimensions', 'space_bounds', 'operators']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    if 'm' not in config['dimensions'] or 'n' not in config['dimensions']:
        raise ValueError("dimensions must contain 'm' and 'n'")

    if len(config['operators']) == 0:
        raise ValueError("At least one operator must be specified")

    # Set defaults
    config.setdefault('norm', 'L2')
    config.setdefault('output_dir', f"result/{config['norm']}_multi")
    config.setdefault('solver', 'auto')
    config.setdefault('verbose', True)

    return config


def main():
    parser = argparse.ArgumentParser(
        description='Project error field onto feasible region with multiple operator constraints.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('config', help='Path to JSON configuration file')

    args = parser.parse_args()

    # Load configuration
    print("=" * 70)
    print(f"Multi-Operator Projection")
    print("=" * 70)
    print(f"\n[1/7] Loading configuration from: {args.config}")
    config = load_config(args.config)

    verbose = config['verbose']
    m = config['dimensions']['m']
    n = config['dimensions']['n']
    norm = config['norm']
    solver = config['solver']

    # Check solver availability
    if norm == 'L2':
        if solver == 'osqp' and not HAS_OSQP:
            raise RuntimeError("OSQP not installed. Install with: pip install osqp")
        if solver == 'cvxpy' and not HAS_CVXPY:
            raise RuntimeError("CVXPY not installed. Install with: pip install cvxpy")
        if solver == 'auto' and not HAS_OSQP and not HAS_CVXPY:
            raise RuntimeError("No QP solver available. Install osqp or cvxpy")
        if solver == 'highs':
            raise RuntimeError("HiGHS solver only available for L1 norm")

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)

    # Read inputs
    print(f"\n[2/7] Reading input fields...")
    start_time = time.time()
    if verbose:
        print(f"  Ground truth: {config['ground_truth']}")
    x = read_raw_float32(config['ground_truth'], m, n)

    if verbose:
        print(f"  Error field: {config['error_field']}")
    x_star = read_raw_float32(config['error_field'], m, n)
    print(f"  ✓ Loaded {m}x{n} fields ({m*n:,} pixels) in {time.time()-start_time:.2f}s")

    # Load operators
    print(f"\n[3/7] Loading {len(config['operators'])} operator(s)...")

    operators = []
    for i, op_config in enumerate(config['operators'], 1):
        op_name = op_config['name']
        op_matrix_path = op_config['matrix']

        if verbose:
            print(f"  [{i}/{len(config['operators'])}] {op_name}: Loading matrix...")

        op_start = time.time()
        A = load_npz(op_matrix_path)

        if A.shape[0] != m * n or A.shape[1] != m * n:
            raise ValueError(f"Matrix {op_name} shape {A.shape} doesn't match field size {m}x{n}")

        if verbose:
            print(f"      Matrix: {A.shape[0]:,} x {A.shape[1]:,}, {A.nnz:,} non-zeros")
            print(f"      Computing ground truth gradient...")

        # Compute ground truth gradient
        Ax = A @ x

        if verbose:
            print(f"      Computing error field gradient...")

        # Compute error field gradient
        Ax_star = A @ x_star

        # Compute error bounds for this operator
        bounds_config = op_config['bounds']
        c = compute_error_bounds(Ax,
                                 rel=bounds_config.get('rel'),
                                 abs_val=bounds_config.get('abs'))

        if verbose:
            print(f"      Error bound: {c[0]:.6e}")
            print(f"      ✓ Completed in {time.time()-op_start:.2f}s")

        operators.append({
            'name': op_name,
            'matrix': A,
            'bounds': c,
            'ground_truth': Ax,
            'error_field': Ax_star,
        })

    # Compute space error bounds
    print(f"\n[4/7] Computing error bounds...")
    space_config = config['space_bounds']
    b = compute_error_bounds(x,
                            rel=space_config.get('rel'),
                            abs_val=space_config.get('abs'))

    if verbose:
        print(f"  Space error bound: {b[0]:.6e}")

    # Compute initial errors
    e_star = x_star - x  # Initial error in space

    # Analyze violations before projection
    print(f"\n[5/7] Analyzing constraint violations...")
    space_violations_before = np.sum(np.abs(e_star) > b)

    print(f"\n  === Before Projection ===")
    print(f"  Space domain:")
    print(f"    Violations: {space_violations_before:,} / {len(e_star):,} pixels "
          f"({100*space_violations_before/len(e_star):.4f}%)")
    print(f"    Max error: {np.abs(e_star).max():.6e} (bound: {b[0]:.6e})")

    total_grad_violations_before = 0
    for op in operators:
        Ae_star = op['error_field'] - op['ground_truth']
        grad_violations = np.sum(np.abs(Ae_star) > op['bounds'])
        total_grad_violations_before += grad_violations

        print(f"  Operator [{op['name']}]:")
        print(f"    Violations: {grad_violations:,} / {len(Ae_star):,} pixels "
              f"({100*grad_violations/len(Ae_star):.4f}%)")
        print(f"    Max error: {np.abs(Ae_star).max():.6e} (bound: {op['bounds'][0]:.6e})")

    # Check if already feasible
    if space_violations_before == 0 and total_grad_violations_before == 0:
        print("\n  ✓ Field already satisfies all constraints. No projection needed.")
        e_proj = e_star
    else:
        # Solve optimization problem
        print(f"\n[6/7] Solving {norm} optimization with {len(operators)} operator(s)...")
        total_constraints = len(e_star) + sum(op['matrix'].shape[0] for op in operators)
        print(f"  Problem size: {len(e_star):,} variables, {total_constraints:,} constraints")
        print(f"  Solver: {solver}")
        print(f"  Starting optimization...")

        solve_start = time.time()

        if norm == 'L1':
            # Linear Programming for L1 norm
            e_proj = solve_lp_scipy_multi(e_star, operators, b, verbose=verbose)
        else:
            # Quadratic Programming for L2 norm
            if solver == 'osqp' or (solver == 'auto' and HAS_OSQP):
                e_proj = solve_qp_osqp_multi(e_star, operators, b, verbose=verbose)
            else:
                e_proj = solve_qp_cvxpy_multi(e_star, operators, b, verbose=verbose)

        print(f"  ✓ Optimization completed in {time.time()-solve_start:.2f}s")

    # Compute projected field
    print(f"\n  Computing projected results...")
    x_proj = x + e_proj

    # Compute post-projection stats
    space_violations_after = np.sum(np.abs(e_proj) > b * (1 + 1e-6))

    # Compute change mask
    change_threshold = 1e-10 * np.abs(x_star).max()
    change_mask = (np.abs(x_proj - x_star) > change_threshold).astype(np.float32)
    pixels_changed = int(change_mask.sum())

    # Compute projection distances
    l1_distance = np.sum(np.abs(x_proj - x_star))
    l2_distance = np.linalg.norm(x_proj - x_star)

    print(f"\n  === After Projection ===")
    print(f"  Space domain:")
    print(f"    Violations: {space_violations_after:,}")
    print(f"    Max error: {np.abs(e_proj).max():.6e} (bound: {b[0]:.6e})")

    total_grad_violations_after = 0
    for i, op in enumerate(operators, 1):
        if verbose:
            print(f"  Computing operator [{i}/{len(operators)}] {op['name']}...")
        Ax_proj = op['matrix'] @ x_proj
        Ae_proj = Ax_proj - op['ground_truth']
        grad_violations = np.sum(np.abs(Ae_proj) > op['bounds'] * (1 + 1e-6))
        total_grad_violations_after += grad_violations

        op['projected_field'] = Ax_proj
        op['projected_error'] = Ae_proj

        print(f"  Operator [{op['name']}]:")
        print(f"    Violations: {grad_violations:,}")
        print(f"    Max error: {np.abs(Ae_proj).max():.6e} (bound: {op['bounds'][0]:.6e})")

    print(f"\n  Summary:")
    print(f"    Pixels changed: {pixels_changed:,} / {len(x_star):,} "
          f"({100*pixels_changed/len(x_star):.4f}%)")
    print(f"    Projection L1 distance: {l1_distance:.6e}")
    print(f"    Projection L2 distance: {l2_distance:.6e}")

    # Write outputs
    print(f"\n[7/7] Writing outputs to {config['output_dir']}...")
    write_start = time.time()

    files_written = 0

    # Ground truth
    path = os.path.join(config['output_dir'], "ground_truth.f32.raw")
    write_raw_float32(path, x)
    files_written += 1
    if verbose:
        print(f"  [{files_written}] {path}")

    # Error field
    path = os.path.join(config['output_dir'], "error_field.f32.raw")
    write_raw_float32(path, x_star)
    files_written += 1
    if verbose:
        print(f"  [{files_written}] {path}")

    # Projected space
    path = os.path.join(config['output_dir'], "projected_space.f32.raw")
    write_raw_float32(path, x_proj)
    files_written += 1
    if verbose:
        print(f"  [{files_written}] {path}")

    # Space change mask
    path = os.path.join(config['output_dir'], "space_change_mask.f32.raw")
    write_raw_float32(path, change_mask)
    files_written += 1
    if verbose:
        print(f"  [{files_written}] {path}")

    # Projected space error (e_proj = x_proj - x)
    path = os.path.join(config['output_dir'], "projected_space_error.f32.raw")
    write_raw_float32(path, e_proj)
    files_written += 1
    if verbose:
        print(f"  [{files_written}] {path}")

    # Write per-operator outputs
    for op in operators:
        op_name = op['name']

        # Ground truth gradient
        path = os.path.join(config['output_dir'], f"ground_truth_{op_name}.f32.raw")
        write_raw_float32(path, op['ground_truth'])
        files_written += 1
        if verbose:
            print(f"  [{files_written}] {path}")

        # Error gradient
        path = os.path.join(config['output_dir'], f"error_{op_name}.f32.raw")
        write_raw_float32(path, op['error_field'])
        files_written += 1
        if verbose:
            print(f"  [{files_written}] {path}")

        # Gradient diff (before projection)
        grad_diff = op['error_field'] - op['ground_truth']
        path = os.path.join(config['output_dir'], f"{op_name}_diff.f32.raw")
        write_raw_float32(path, grad_diff)
        files_written += 1
        if verbose:
            print(f"  [{files_written}] {path}")

        # Gradient out-of-bound mask (before projection)
        grad_oob = (np.abs(grad_diff) > op['bounds']).astype(np.float32)
        path = os.path.join(config['output_dir'], f"{op_name}_oob_mask.f32.raw")
        write_raw_float32(path, grad_oob)
        files_written += 1
        if verbose:
            print(f"  [{files_written}] {path}")

        # Projected gradient
        path = os.path.join(config['output_dir'], f"projected_{op_name}.f32.raw")
        write_raw_float32(path, op['projected_field'])
        files_written += 1
        if verbose:
            print(f"  [{files_written}] {path}")

        # Projected gradient error
        path = os.path.join(config['output_dir'], f"projected_{op_name}_error.f32.raw")
        write_raw_float32(path, op['projected_error'])
        files_written += 1
        if verbose:
            print(f"  [{files_written}] {path}")

    # Write statistics to file
    stats_path = os.path.join(config['output_dir'], "projection_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"=== {norm} Multi-Operator Projection Statistics ===\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Config file: {args.config}\n")
        f.write(f"  Ground truth: {config['ground_truth']}\n")
        f.write(f"  Error field: {config['error_field']}\n")
        f.write(f"  Dimensions: {m} x {n}\n")
        f.write(f"  Norm: {norm}\n")
        f.write(f"  Solver: {solver}\n")
        f.write(f"\nSpace Bounds:\n")
        if 'rel' in space_config:
            f.write(f"  REL {space_config['rel']} = {b[0]:.6e} absolute\n")
        else:
            f.write(f"  ABS {space_config['abs']}\n")
        f.write(f"\nOperators ({len(operators)}):\n")
        for i, op_config in enumerate(config['operators']):
            op = operators[i]
            f.write(f"  [{op['name']}] Matrix: {op_config['matrix']}\n")
            bounds_cfg = op_config['bounds']
            if 'rel' in bounds_cfg:
                f.write(f"  [{op['name']}] Bounds: REL {bounds_cfg['rel']} = {op['bounds'][0]:.6e} absolute\n")
            else:
                f.write(f"  [{op['name']}] Bounds: ABS {bounds_cfg['abs']}\n")
        f.write(f"\nBefore Projection:\n")
        f.write(f"  Space violations: {space_violations_before} ({100*space_violations_before/len(e_star):.4f}%)\n")
        f.write(f"  Max space error: {np.abs(e_star).max():.6e}\n")
        for op in operators:
            Ae_star = op['error_field'] - op['ground_truth']
            grad_violations = np.sum(np.abs(Ae_star) > op['bounds'])
            f.write(f"  [{op['name']}] violations: {grad_violations} ({100*grad_violations/len(Ae_star):.4f}%)\n")
            f.write(f"  [{op['name']}] max error: {np.abs(Ae_star).max():.6e}\n")
        f.write(f"\nAfter Projection:\n")
        f.write(f"  Space violations: {space_violations_after}\n")
        f.write(f"  Max space error: {np.abs(e_proj).max():.6e}\n")
        for op in operators:
            grad_violations = np.sum(np.abs(op['projected_error']) > op['bounds'] * (1 + 1e-6))
            f.write(f"  [{op['name']}] violations: {grad_violations}\n")
            f.write(f"  [{op['name']}] max error: {np.abs(op['projected_error']).max():.6e}\n")
        f.write(f"  Pixels changed: {pixels_changed} ({100*pixels_changed/len(x_star):.4f}%)\n")
        f.write(f"  Projection L1 distance: {l1_distance:.6e}\n")
        f.write(f"  Projection L2 distance: {l2_distance:.6e}\n")

    files_written += 1
    if verbose:
        print(f"  [{files_written}] {stats_path}")

    print(f"  ✓ Wrote {files_written} files in {time.time()-write_start:.2f}s")
    print("\n" + "=" * 70)
    print("✓ Projection complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
