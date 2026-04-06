#!/usr/bin/env python3
"""
CuPy-based GPU solver for multi-operator projection.

Uses CuPy for GPU-accelerated sparse matrix operations, which can be faster
than PyTorch for large sparse systems.

Algorithm: ADMM (Alternating Direction Method of Multipliers)

The problem:
    minimize ||e - e*||_p  (p = 1 or 2)
    subject to -b <= e <= b  (space bounds)
               -c^(k) <= A_k e <= c^(k)  (operator k bounds)

is reformulated as:
    minimize f(e) + g(z)
    subject to Ge = z

where G = [I; A_1; ...; A_m] stacks all constraints.

Requirements:
    pip install cupy-cuda11x  # (or cupy-cuda12x for CUDA 12)

Usage:
    python GPU_solver/cupy_solver.py config.json
"""

import argparse
import numpy as np
from scipy.sparse import load_npz, csr_matrix
import os
import json
import time
from typing import Dict, List, Tuple

# Try to import CuPy
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


def scipy_to_cupy_sparse(scipy_sparse: csr_matrix):
    """Convert scipy sparse CSR to CuPy sparse CSR."""
    return cpsparse.csr_matrix(
        (cp.array(scipy_sparse.data),
         cp.array(scipy_sparse.indices),
         cp.array(scipy_sparse.indptr)),
        shape=scipy_sparse.shape
    )


def read_raw_float32(filepath: str, m: int, n: int) -> np.ndarray:
    """Read raw binary file as float32 array."""
    data = np.fromfile(filepath, dtype=np.float32)
    if data.size != m * n:
        raise ValueError(f"File size mismatch: expected {m*n}, got {data.size}")
    return data.astype(np.float64)


def write_raw_float32(filepath: str, data: np.ndarray):
    """Write array to raw binary file as float32."""
    data.astype(np.float32).tofile(filepath)


class CuPyADMMSolver:
    """
    ADMM solver using CuPy for GPU acceleration.

    Optimized for large sparse constraint matrices.
    """

    def __init__(self,
                 e_star: cp.ndarray,
                 operators: List[Dict],
                 space_bounds: cp.ndarray,
                 norm: str = 'L2',
                 rho: float = 1.0,
                 max_iter: int = 2000,
                 tol_abs: float = 1e-6,
                 tol_rel: float = 1e-4,
                 adaptive_rho: bool = True,
                 verbose: bool = True,
                 print_interval: int = 10):
        """
        Initialize CuPy ADMM solver.

        Args:
            e_star: Initial error vector on GPU
            operators: List with 'matrix' (CuPy sparse) and 'bounds' (CuPy array)
            space_bounds: Per-element bounds on GPU
            norm: 'L1' or 'L2'
            rho: ADMM penalty parameter
            max_iter: Maximum iterations
            tol_abs: Absolute convergence tolerance
            tol_rel: Relative convergence tolerance
            adaptive_rho: Adaptively adjust rho
            verbose: Print progress
            print_interval: Print status every N iterations
        """
        self.n = len(e_star)
        self.norm = norm
        self.rho = rho
        self.max_iter = max_iter
        self.tol_abs = tol_abs
        self.tol_rel = tol_rel
        self.adaptive_rho = adaptive_rho
        self.verbose = verbose
        self.print_interval = print_interval

        self.e_star = e_star
        self.b = space_bounds
        self.operators = operators

        # Total constraint dimension
        self.total_constraints = self.n
        for op in operators:
            self.total_constraints += op['matrix'].shape[0]

        # Build bound vectors
        lower = [(-self.b)]
        upper = [self.b]
        for op in operators:
            lower.append(-op['bounds'])
            upper.append(op['bounds'])

        self.z_lower = cp.concatenate(lower)
        self.z_upper = cp.concatenate(upper)

        # Initialize variables
        self.e = self.e_star.copy()
        self.z = cp.zeros(self.total_constraints, dtype=cp.float64)
        self.u = cp.zeros(self.total_constraints, dtype=cp.float64)

        # CG parameters
        self.cg_max_iter = 100
        self.cg_tol = 1e-8

    def _apply_G(self, e: cp.ndarray) -> cp.ndarray:
        """Apply G = [I; A_1; ...; A_m] to e."""
        results = [e]
        for op in self.operators:
            results.append(op['matrix'] @ e)
        return cp.concatenate(results)

    def _apply_Gt(self, z: cp.ndarray) -> cp.ndarray:
        """Apply G^T = [I, A_1^T, ..., A_m^T] to z."""
        idx = 0
        result = z[idx:idx + self.n].copy()
        idx += self.n

        for op in self.operators:
            size = op['matrix'].shape[0]
            z_k = z[idx:idx + size]
            result += op['matrix'].T @ z_k
            idx += size

        return result

    def _apply_GtG(self, e: cp.ndarray) -> cp.ndarray:
        """Apply G^T G = I + sum_k A_k^T A_k to e."""
        result = e.copy()
        for op in self.operators:
            Ae = op['matrix'] @ e
            result += op['matrix'].T @ Ae
        return result

    def _cg_solve(self, b: cp.ndarray, x0: cp.ndarray) -> cp.ndarray:
        """Solve (coeff*I + rho*G^T G) x = b using CG."""
        coeff = 2.0 if self.norm == 'L2' else 0.0

        def matvec(x):
            return coeff * x + self.rho * self._apply_GtG(x)

        x = x0.copy()
        r = b - matvec(x)
        p = r.copy()
        rsold = cp.dot(r, r)

        for _ in range(self.cg_max_iter):
            Ap = matvec(p)
            pAp = cp.dot(p, Ap)
            if pAp < 1e-15:
                break

            alpha = rsold / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = cp.dot(r, r)

            if cp.sqrt(rsnew) < self.cg_tol:
                break

            p = r + (rsnew / (rsold + 1e-15)) * p
            rsold = rsnew

        return x

    def _e_update_L2(self):
        """E-update for L2 norm."""
        rhs = 2.0 * self.e_star + self.rho * self._apply_Gt(self.z - self.u)
        self.e = self._cg_solve(rhs, self.e)

    def _e_update_L1(self):
        """
        E-update for L1 norm using proximal gradient method.

        Solve: minimize ||e - e*||_1 + (rho/2)||Ge - z + u||_2^2
        """
        def soft_threshold(x, threshold):
            return cp.sign(x) * cp.maximum(cp.abs(x) - threshold, 0)

        # Estimate Lipschitz constant
        if not hasattr(self, '_L_estimate'):
            v = cp.random.randn(self.n)
            v = v / cp.linalg.norm(v)
            for _ in range(20):
                Gv = self._apply_GtG(v)
                norm_Gv = cp.linalg.norm(Gv)
                if norm_Gv > 1e-10:
                    v = Gv / norm_Gv
            self._L_estimate = float(self.rho * cp.dot(v, self._apply_GtG(v)))
            self._L_estimate = max(self._L_estimate, self.rho)

        L = self._L_estimate
        tau = 1.0 / L

        e = self.e.copy()
        target = self.z - self.u

        for _ in range(self.cg_max_iter):
            Ge = self._apply_G(e)
            grad = self.rho * self._apply_Gt(Ge - target)
            v = e - (1.0 / L) * grad
            e_new = self.e_star + soft_threshold(v - self.e_star, tau)

            diff = cp.linalg.norm(e_new - e)
            e = e_new

            if diff < self.cg_tol * cp.linalg.norm(e):
                break

        self.e = e

    def _z_update(self):
        """Z-update: project onto box constraints."""
        v = self._apply_G(self.e) + self.u
        self.z = cp.clip(v, self.z_lower, self.z_upper)

    def _u_update(self):
        """U-update: dual variable."""
        self.u = self.u + self._apply_G(self.e) - self.z

    def _compute_residuals(self) -> Tuple[float, float]:
        """Compute primal and dual residuals."""
        Ge = self._apply_G(self.e)
        primal = float(cp.linalg.norm(Ge - self.z))
        dual = self.rho * float(cp.linalg.norm(self._apply_Gt(self.z)))
        return primal, dual

    def _update_rho(self, primal_res: float, dual_res: float):
        """Adaptive rho update."""
        mu = 10.0
        tau = 2.0

        old_rho = self.rho

        if primal_res > mu * dual_res:
            self.rho *= tau
            self.u /= tau
        elif dual_res > mu * primal_res:
            self.rho /= tau
            self.u *= tau

        # Reset Lipschitz estimate if rho changed
        if self.rho != old_rho and hasattr(self, '_L_estimate'):
            del self._L_estimate

    def _compute_objective(self) -> float:
        """Compute objective value."""
        diff = self.e - self.e_star
        if self.norm == 'L2':
            return float(cp.sum(diff ** 2))
        else:
            return float(cp.sum(cp.abs(diff)))

    def _count_violations(self) -> Tuple[int, int]:
        """Count constraint violations."""
        space_viol = int(cp.sum(cp.abs(self.e) > self.b))
        grad_viol = 0
        idx = self.n
        Ge = self._apply_G(self.e)
        for op in self.operators:
            size = op['matrix'].shape[0]
            Ae = Ge[idx:idx + size]
            grad_viol += int(cp.sum(cp.abs(Ae) > op['bounds']))
            idx += size
        return space_viol, grad_viol

    def solve(self) -> cp.ndarray:
        """Run ADMM solver."""
        if self.verbose:
            print(f"  CuPy ADMM solver")
            print(f"  Variables: {self.n:,}")
            print(f"  Constraints: {self.total_constraints:,}")
            print(f"  Initial rho: {self.rho}")
            print(f"  Print interval: every {self.print_interval} iterations")
            print()
            print(f"  {'Iter':>6} | {'Primal':>10} | {'Dual':>10} | {'Objective':>12} | "
                  f"{'Space Viol':>10} | {'Grad Viol':>10} | {'Rho':>8} | {'Time':>6}")
            print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}")

        start_time = time.time()

        for iteration in range(self.max_iter):
            # ADMM steps
            if self.norm == 'L2':
                self._e_update_L2()
            else:
                self._e_update_L1()

            self._z_update()
            self._u_update()

            # Check convergence
            primal_res, dual_res = self._compute_residuals()

            Ge = self._apply_G(self.e)
            eps_pri = np.sqrt(self.total_constraints) * self.tol_abs + \
                      self.tol_rel * max(float(cp.linalg.norm(Ge)), float(cp.linalg.norm(self.z)))
            eps_dual = np.sqrt(self.n) * self.tol_abs + \
                       self.tol_rel * float(cp.linalg.norm(self.u))

            if self.verbose and (iteration + 1) % self.print_interval == 0:
                obj_val = self._compute_objective()
                space_viol, grad_viol = self._count_violations()
                elapsed = time.time() - start_time
                print(f"  {iteration+1:>6} | {primal_res:>10.2e} | {dual_res:>10.2e} | "
                      f"{obj_val:>12.4e} | {space_viol:>10,} | {grad_viol:>10,} | "
                      f"{self.rho:>8.2e} | {elapsed:>5.1f}s")

            if primal_res < eps_pri and dual_res < eps_dual:
                if self.verbose:
                    obj_val = self._compute_objective()
                    space_viol, grad_viol = self._count_violations()
                    elapsed = time.time() - start_time
                    print(f"  {iteration+1:>6} | {primal_res:>10.2e} | {dual_res:>10.2e} | "
                          f"{obj_val:>12.4e} | {space_viol:>10,} | {grad_viol:>10,} | "
                          f"{self.rho:>8.2e} | {elapsed:>5.1f}s")
                    print()
                    print(f"  Converged at iteration {iteration + 1}")
                break

            if self.adaptive_rho and (iteration + 1) % 10 == 0:
                self._update_rho(primal_res, dual_res)

        solve_time = time.time() - start_time

        if self.verbose:
            print(f"  Solve time: {solve_time:.2f}s")
            print(f"  Final objective: {self._compute_objective():.6e}")

        return self.e


def compute_error_bounds(reference_data: np.ndarray,
                         rel: float = None,
                         abs_val: float = None) -> np.ndarray:
    """Compute per-pixel error bounds."""
    if abs_val is not None:
        return np.full(len(reference_data), abs_val)
    elif rel is not None:
        data_range = reference_data.max() - reference_data.min()
        return np.full(len(reference_data), rel * data_range)
    else:
        raise ValueError("Must specify 'rel' or 'abs'")


def load_config(config_path: str) -> Dict:
    """Load JSON configuration."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    required = ['ground_truth', 'error_field', 'dimensions', 'space_bounds', 'operators']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing: {field}")

    config.setdefault('norm', 'L2')
    config.setdefault('output_dir', f"result/{config['norm']}_multi_cupy")
    config.setdefault('verbose', True)
    config.setdefault('admm', {})
    config['admm'].setdefault('rho', 1.0)
    config['admm'].setdefault('max_iter', 2000)
    config['admm'].setdefault('tol_abs', 1e-6)
    config['admm'].setdefault('tol_rel', 1e-4)
    config['admm'].setdefault('adaptive_rho', True)
    config['admm'].setdefault('print_interval', 10)

    return config


def main():
    if not HAS_CUPY:
        print("ERROR: CuPy not installed. Install with: pip install cupy-cuda11x")
        print("       (or cupy-cuda12x for CUDA 12)")
        return

    parser = argparse.ArgumentParser(
        description='CuPy GPU solver for multi-operator projection.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('config', help='Path to JSON config file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')

    args = parser.parse_args()

    # Set GPU device
    cp.cuda.Device(args.gpu).use()

    print("=" * 70)
    print("CuPy Multi-Operator Projection (ADMM)")
    print("=" * 70)
    print(f"\nGPU {args.gpu}: {cp.cuda.runtime.getDeviceProperties(args.gpu)['name'].decode()}")
    mem_info = cp.cuda.runtime.memGetInfo()
    print(f"Memory: {mem_info[1] / 1e9:.1f} GB total, {mem_info[0] / 1e9:.1f} GB free")

    print(f"\n[1/7] Loading configuration: {args.config}")
    config = load_config(args.config)

    verbose = config['verbose']
    m = config['dimensions']['m']
    n = config['dimensions']['n']
    norm = config['norm']
    admm_config = config['admm']

    os.makedirs(config['output_dir'], exist_ok=True)

    # Read inputs
    print(f"\n[2/7] Reading input fields...")
    x = read_raw_float32(config['ground_truth'], m, n)
    x_star = read_raw_float32(config['error_field'], m, n)
    print(f"  Loaded {m}x{n} fields ({m*n:,} pixels)")

    # Load operators
    print(f"\n[3/7] Loading {len(config['operators'])} operator(s)...")

    operators_gpu = []
    operators_np = []

    for i, op_config in enumerate(config['operators'], 1):
        op_name = op_config['name']
        A_scipy = load_npz(op_config['matrix'])

        if verbose:
            print(f"  [{i}] {op_name}: {A_scipy.shape}, {A_scipy.nnz:,} nnz")

        # Convert to CuPy sparse
        A_cupy = scipy_to_cupy_sparse(A_scipy)

        Ax = A_scipy @ x
        Ax_star = A_scipy @ x_star

        bounds_config = op_config['bounds']
        c = compute_error_bounds(Ax, rel=bounds_config.get('rel'), abs_val=bounds_config.get('abs'))

        operators_gpu.append({
            'name': op_name,
            'matrix': A_cupy,
            'bounds': cp.array(c),
        })

        operators_np.append({
            'name': op_name,
            'matrix': A_scipy,
            'bounds': c,
            'ground_truth': Ax,
            'error_field': Ax_star,
        })

    # Space bounds
    print(f"\n[4/7] Computing error bounds...")
    space_config = config['space_bounds']
    b = compute_error_bounds(x, rel=space_config.get('rel'), abs_val=space_config.get('abs'))
    if verbose:
        print(f"  Space bound: {b[0]:.6e}")

    e_star = x_star - x

    # Analyze violations
    print(f"\n[5/7] Analyzing violations...")
    space_violations_before = np.sum(np.abs(e_star) > b)
    print(f"\n  === Before Projection ===")
    print(f"  Space violations: {space_violations_before:,} ({100*space_violations_before/len(e_star):.4f}%)")

    total_grad_violations = 0
    for op in operators_np:
        Ae = op['error_field'] - op['ground_truth']
        viol = np.sum(np.abs(Ae) > op['bounds'])
        total_grad_violations += viol
        print(f"  [{op['name']}] violations: {viol:,}")

    if space_violations_before == 0 and total_grad_violations == 0:
        print("\n  Already feasible!")
        e_proj = e_star
    else:
        # Solve
        print(f"\n[6/7] Solving {norm} optimization...")

        e_star_gpu = cp.array(e_star)
        b_gpu = cp.array(b)

        solver = CuPyADMMSolver(
            e_star=e_star_gpu,
            operators=operators_gpu,
            space_bounds=b_gpu,
            norm=norm,
            rho=admm_config['rho'],
            max_iter=admm_config['max_iter'],
            tol_abs=admm_config['tol_abs'],
            tol_rel=admm_config['tol_rel'],
            adaptive_rho=admm_config['adaptive_rho'],
            verbose=verbose,
            print_interval=admm_config['print_interval']
        )

        solve_start = time.time()
        e_proj_gpu = solver.solve()
        print(f"  Completed in {time.time()-solve_start:.2f}s")

        e_proj = cp.asnumpy(e_proj_gpu)

    # Results
    x_proj = x + e_proj

    space_violations_after = np.sum(np.abs(e_proj) > b * (1 + 1e-6))
    change_mask = (np.abs(x_proj - x_star) > 1e-10 * np.abs(x_star).max()).astype(np.float32)
    pixels_changed = int(change_mask.sum())

    l1_dist = np.sum(np.abs(x_proj - x_star))
    l2_dist = np.linalg.norm(x_proj - x_star)

    print(f"\n  === After Projection ===")
    print(f"  Space violations: {space_violations_after}")

    for op in operators_np:
        Ax_proj = op['matrix'] @ x_proj
        Ae_proj = Ax_proj - op['ground_truth']
        viol = np.sum(np.abs(Ae_proj) > op['bounds'] * (1 + 1e-6))
        op['projected_field'] = Ax_proj
        op['projected_error'] = Ae_proj
        print(f"  [{op['name']}] violations: {viol}")

    print(f"\n  Pixels changed: {pixels_changed:,} ({100*pixels_changed/len(x_star):.4f}%)")
    print(f"  L1 distance: {l1_dist:.6e}")
    print(f"  L2 distance: {l2_dist:.6e}")

    # Write outputs
    print(f"\n[7/7] Writing outputs...")

    write_raw_float32(os.path.join(config['output_dir'], "ground_truth.f32.raw"), x)
    write_raw_float32(os.path.join(config['output_dir'], "error_field.f32.raw"), x_star)
    write_raw_float32(os.path.join(config['output_dir'], "projected_space.f32.raw"), x_proj)
    write_raw_float32(os.path.join(config['output_dir'], "space_change_mask.f32.raw"), change_mask)
    write_raw_float32(os.path.join(config['output_dir'], "projected_space_error.f32.raw"), e_proj)

    for op in operators_np:
        name = op['name']
        write_raw_float32(os.path.join(config['output_dir'], f"ground_truth_{name}.f32.raw"),
                          op['ground_truth'])
        write_raw_float32(os.path.join(config['output_dir'], f"projected_{name}.f32.raw"),
                          op['projected_field'])
        write_raw_float32(os.path.join(config['output_dir'], f"projected_{name}_error.f32.raw"),
                          op['projected_error'])

    # Stats file
    stats_path = os.path.join(config['output_dir'], "projection_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"=== {norm} Multi-Operator Projection (CuPy ADMM) ===\n")
        f.write(f"GPU: {args.gpu}\n")
        f.write(f"Dimensions: {m} x {n}\n")
        f.write(f"Space violations after: {space_violations_after}\n")
        f.write(f"Pixels changed: {pixels_changed}\n")
        f.write(f"L1 distance: {l1_dist:.6e}\n")
        f.write(f"L2 distance: {l2_dist:.6e}\n")

    print("\n" + "=" * 70)
    print("Projection complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
