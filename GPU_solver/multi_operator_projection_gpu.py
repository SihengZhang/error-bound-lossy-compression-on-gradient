#!/usr/bin/env python3
"""
GPU-accelerated projection of error field onto feasible region with multiple operator constraints.

Implements Lemma 4.6 (Combined Spatial and Multiple Output Error Bounds) using ADMM on GPU.

Uses PyTorch with CUDA for GPU acceleration. Falls back to CPU if CUDA is not available.

Algorithm: ADMM (Alternating Direction Method of Multipliers)
    For L2: minimize ||e - e*||_2^2 subject to linear constraints
    For L1: minimize ||e - e*||_1 subject to linear constraints

The constrained problem is reformulated as:
    minimize f(e) + g(z)
    subject to Ge = z

where:
    - f(e) = ||e - e*||_p^p (p=1 or 2)
    - g(z) = indicator function of the constraint set
    - G = [I; A_1; A_2; ...; A_m] (stacked constraint matrices)
    - z = [z_space; z_1; z_2; ...; z_m] (auxiliary variables)

ADMM iterations:
    e^{k+1} = argmin_e f(e) + (rho/2)||Ge - z^k + u^k||_2^2
    z^{k+1} = proj_C(Ge^{k+1} + u^k)
    u^{k+1} = u^k + Ge^{k+1} - z^{k+1}

Configuration File Format (JSON): Same as multi_operator_projection.py

Usage:
    python GPU_solver/multi_operator_projection_gpu.py config.json

    # With specific device
    python GPU_solver/multi_operator_projection_gpu.py config.json --device cuda:0

    # Force CPU (for comparison)
    python GPU_solver/multi_operator_projection_gpu.py config.json --device cpu
"""

import argparse
import numpy as np
from scipy.sparse import load_npz, csr_matrix
import os
import json
import time
from typing import Dict, List, Tuple, Optional

import torch
import torch.sparse

# Check CUDA availability
HAS_CUDA = torch.cuda.is_available()


def scipy_sparse_to_torch(scipy_sparse: csr_matrix, device: torch.device) -> torch.Tensor:
    """Convert scipy sparse CSR matrix to PyTorch sparse tensor."""
    coo = scipy_sparse.tocoo()
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
    return sparse_tensor.to(device).coalesce()


def read_raw_float32(filepath: str, m: int, n: int) -> np.ndarray:
    """Read raw binary file as float32 array."""
    data = np.fromfile(filepath, dtype=np.float32)
    if data.size != m * n:
        raise ValueError(f"File size mismatch: expected {m*n} floats, got {data.size}")
    return data.astype(np.float64)


def write_raw_float32(filepath: str, data: np.ndarray):
    """Write array to raw binary file as float32."""
    data.astype(np.float32).tofile(filepath)


class ADMMSolver:
    """
    ADMM solver for constrained optimization on GPU.

    Solves: minimize ||e - e*||_p subject to box constraints on e and Ae.
    """

    def __init__(self,
                 e_star: torch.Tensor,
                 operators: List[Dict],
                 space_bounds: torch.Tensor,
                 norm: str = 'L2',
                 rho: float = 1.0,
                 max_iter: int = 1000,
                 tol_abs: float = 1e-6,
                 tol_rel: float = 1e-4,
                 adaptive_rho: bool = True,
                 verbose: bool = True,
                 print_interval: int = 10,
                 device: torch.device = None):
        """
        Initialize ADMM solver.

        Args:
            e_star: Initial error vector (n,)
            operators: List of dicts with 'matrix' (torch sparse), 'bounds' (torch tensor)
            space_bounds: Per-element space bounds (n,)
            norm: 'L1' or 'L2'
            rho: ADMM penalty parameter
            max_iter: Maximum iterations
            tol_abs: Absolute tolerance for convergence
            tol_rel: Relative tolerance for convergence
            adaptive_rho: Whether to adaptively update rho
            verbose: Print progress
            print_interval: Print status every N iterations
            device: torch device
        """
        self.device = device if device else torch.device('cuda' if HAS_CUDA else 'cpu')
        self.n = len(e_star)
        self.norm = norm
        self.rho = rho
        self.max_iter = max_iter
        self.tol_abs = tol_abs
        self.tol_rel = tol_rel
        self.adaptive_rho = adaptive_rho
        self.verbose = verbose
        self.print_interval = print_interval

        # Move data to device
        self.e_star = e_star.to(self.device)
        self.b = space_bounds.to(self.device)

        # Store operators
        self.operators = []
        self.total_constraints = self.n  # Start with space constraints

        for op in operators:
            op_dict = {
                'name': op['name'],
                'matrix': op['matrix'].to(self.device),
                'bounds': op['bounds'].to(self.device),
                'size': op['matrix'].shape[0]
            }
            self.operators.append(op_dict)
            self.total_constraints += op_dict['size']

        # Build stacked constraint matrix G = [I; A_1; A_2; ...; A_m]
        self._build_constraint_matrix()

        # Initialize variables
        self.e = self.e_star.clone()
        self.z = torch.zeros(self.total_constraints, device=self.device)
        self.u = torch.zeros(self.total_constraints, device=self.device)

        # Precompute for e-update
        self._precompute_e_update()

    def _build_constraint_matrix(self):
        """Build the stacked constraint matrix G."""
        # G = [I; A_1; A_2; ...; A_m]
        # We store this implicitly to avoid memory issues
        # For sparse matrix-vector products, we compute separately

        # Build lower and upper bounds for z
        lower = [-self.b]
        upper = [self.b]

        for op in self.operators:
            lower.append(-op['bounds'])
            upper.append(op['bounds'])

        self.z_lower = torch.cat(lower)
        self.z_upper = torch.cat(upper)

    def _apply_G(self, e: torch.Tensor) -> torch.Tensor:
        """Apply G = [I; A_1; ...; A_m] to vector e."""
        results = [e]  # I @ e = e

        for op in self.operators:
            # Sparse matrix-vector product
            Ae = torch.sparse.mm(op['matrix'], e.unsqueeze(1)).squeeze(1)
            results.append(Ae)

        return torch.cat(results)

    def _apply_G_transpose(self, z: torch.Tensor) -> torch.Tensor:
        """Apply G^T = [I, A_1^T, ..., A_m^T] to vector z."""
        # Split z into components
        idx = 0
        result = z[idx:idx + self.n].clone()  # I^T @ z_space
        idx += self.n

        for op in self.operators:
            z_k = z[idx:idx + op['size']]
            # A_k^T @ z_k
            AtZ = torch.sparse.mm(op['matrix'].t(), z_k.unsqueeze(1)).squeeze(1)
            result = result + AtZ
            idx += op['size']

        return result

    def _precompute_e_update(self):
        """Precompute matrices for the e-update step."""
        # For L2 norm: e-update solves (2I + rho * G^T G) e = 2e* + rho * G^T(z - u)
        # For L1 norm: use soft thresholding with proximal operator

        # G^T G = I + sum_k A_k^T A_k
        # We compute this implicitly using conjugate gradient

        # For efficiency with sparse matrices, we use CG solver
        self.use_cg = True
        self.cg_max_iter = 100
        self.cg_tol = 1e-8

    def _compute_GtG(self, e: torch.Tensor) -> torch.Tensor:
        """Compute G^T G @ e = e + sum_k A_k^T A_k @ e."""
        result = e.clone()

        for op in self.operators:
            Ae = torch.sparse.mm(op['matrix'], e.unsqueeze(1)).squeeze(1)
            AtAe = torch.sparse.mm(op['matrix'].t(), Ae.unsqueeze(1)).squeeze(1)
            result = result + AtAe

        return result

    def _cg_solve(self, b: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """
        Solve (2I + rho * G^T G) x = b using Conjugate Gradient.

        For L2: coefficient is 2 (from objective)
        """
        coeff = 2.0 if self.norm == 'L2' else 0.0

        def matvec(x):
            return coeff * x + self.rho * self._compute_GtG(x)

        x = x0.clone()
        r = b - matvec(x)
        p = r.clone()
        rsold = torch.dot(r, r)

        for i in range(self.cg_max_iter):
            Ap = matvec(p)
            alpha = rsold / (torch.dot(p, Ap) + 1e-10)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.dot(r, r)

            if torch.sqrt(rsnew) < self.cg_tol:
                break

            p = r + (rsnew / (rsold + 1e-10)) * p
            rsold = rsnew

        return x

    def _e_update_L2(self):
        """Update e for L2 norm using CG."""
        # Solve: (2I + rho * G^T G) e = 2e* + rho * G^T(z - u)
        rhs = 2.0 * self.e_star + self.rho * self._apply_G_transpose(self.z - self.u)
        self.e = self._cg_solve(rhs, self.e)

    def _e_update_L1(self):
        """
        Update e for L1 norm using proximal gradient method (linearized ADMM).

        We need to solve:
            minimize ||e - e*||_1 + (rho/2)||Ge - z + u||_2^2

        Using proximal gradient with step size 1/L where L = rho * ||G^T G||:
            e^{k+1} = prox_{(1/L)||·-e*||_1}(e^k - (1/L) * rho * G^T(Ge^k - z + u))

        The proximal operator of ||· - e*||_1 with parameter tau is:
            prox_{tau||·-e*||_1}(v) = e* + soft_threshold(v - e*, tau)
        """
        def soft_threshold(x, threshold):
            return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0)

        # Estimate Lipschitz constant L = rho * ||G^T G||
        # For G = [I; A_1; ...; A_m], ||G^T G|| = ||I + sum_k A_k^T A_k||
        # Conservative estimate: L = rho * (1 + sum of operator norms)
        # We use a fixed estimate based on problem structure
        if not hasattr(self, '_L_estimate'):
            # Estimate using power iteration on G^T G
            v = torch.randn(self.n, device=self.device)
            v = v / torch.norm(v)
            for _ in range(20):
                Gv = self._compute_GtG(v)
                norm_Gv = torch.norm(Gv)
                if norm_Gv > 1e-10:
                    v = Gv / norm_Gv
            self._L_estimate = self.rho * torch.dot(v, self._compute_GtG(v)).item()
            self._L_estimate = max(self._L_estimate, self.rho)  # At least rho

        L = self._L_estimate
        tau = 1.0 / L  # Proximal parameter

        # Run proximal gradient iterations
        e = self.e.clone()
        target = self.z - self.u

        for _ in range(self.cg_max_iter):
            # Gradient of (rho/2)||Ge - target||^2 is rho * G^T(Ge - target)
            Ge = self._apply_G(e)
            grad = self.rho * self._apply_G_transpose(Ge - target)

            # Gradient step
            v = e - (1.0 / L) * grad

            # Proximal step: prox_{tau||·-e*||_1}(v) = e* + soft_threshold(v - e*, tau)
            e_new = self.e_star + soft_threshold(v - self.e_star, tau)

            # Check convergence
            diff = torch.norm(e_new - e)
            e = e_new

            if diff < self.cg_tol * torch.norm(e):
                break

        self.e = e

    def _z_update(self):
        """Update z by projecting onto box constraints."""
        Ge = self._apply_G(self.e)
        v = Ge + self.u
        self.z = torch.clamp(v, self.z_lower, self.z_upper)

    def _u_update(self):
        """Update dual variable u."""
        Ge = self._apply_G(self.e)
        self.u = self.u + Ge - self.z

    def _compute_residuals(self) -> Tuple[float, float]:
        """Compute primal and dual residuals."""
        Ge = self._apply_G(self.e)

        # Primal residual: ||Ge - z||
        primal = torch.norm(Ge - self.z).item()

        # Dual residual: ||rho * G^T(z - z_prev)||
        # Approximated as ||rho * G^T z||
        dual = self.rho * torch.norm(self._apply_G_transpose(self.z)).item()

        return primal, dual

    def _update_rho(self, primal_res: float, dual_res: float):
        """Adaptively update rho based on residuals."""
        mu = 10.0
        tau_incr = 2.0
        tau_decr = 2.0

        old_rho = self.rho

        if primal_res > mu * dual_res:
            self.rho *= tau_incr
            self.u /= tau_incr
        elif dual_res > mu * primal_res:
            self.rho /= tau_decr
            self.u *= tau_decr

        # Reset Lipschitz estimate if rho changed (for L1)
        if self.rho != old_rho and hasattr(self, '_L_estimate'):
            del self._L_estimate

    def _count_violations(self) -> Tuple[int, int]:
        """Count constraint violations for space and gradient bounds."""
        # Space violations
        space_viol = torch.sum(torch.abs(self.e) > self.b).item()

        # Gradient violations
        grad_viol = 0
        idx = self.n
        Ge = self._apply_G(self.e)
        for op in self.operators:
            size = op['size']
            Ae = Ge[idx:idx + size]
            grad_viol += torch.sum(torch.abs(Ae) > op['bounds']).item()
            idx += size

        return int(space_viol), int(grad_viol)

    def solve(self) -> torch.Tensor:
        """Run ADMM to solve the optimization problem."""
        if self.verbose:
            print(f"  ADMM solver on {self.device}")
            print(f"  Variables: {self.n:,}")
            print(f"  Constraints: {self.total_constraints:,}")
            print(f"  Initial rho: {self.rho}")
            print(f"  Print interval: every {self.print_interval} iterations")
            print()
            print(f"  {'Iter':>6} | {'Primal':>10} | {'Dual':>10} | {'Objective':>12} | "
                  f"{'Space Viol':>10} | {'Grad Viol':>10} | {'Rho':>8} | {'Time':>6}")
            print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}")

        start_time = time.time()
        last_print_time = start_time
        z_prev = self.z.clone()

        for iteration in range(self.max_iter):
            # E-update
            if self.norm == 'L2':
                self._e_update_L2()
            else:
                self._e_update_L1()

            # Z-update
            z_prev = self.z.clone()
            self._z_update()

            # U-update
            self._u_update()

            # Check convergence
            primal_res, dual_res = self._compute_residuals()

            # Compute tolerances
            Ge = self._apply_G(self.e)
            eps_pri = np.sqrt(self.total_constraints) * self.tol_abs + \
                      self.tol_rel * max(torch.norm(Ge).item(), torch.norm(self.z).item())
            eps_dual = np.sqrt(self.n) * self.tol_abs + \
                       self.tol_rel * torch.norm(self.u).item()

            # Print status at specified interval
            if self.verbose and (iteration + 1) % self.print_interval == 0:
                obj_val = self._compute_objective()
                space_viol, grad_viol = self._count_violations()
                elapsed = time.time() - start_time
                print(f"  {iteration+1:>6} | {primal_res:>10.2e} | {dual_res:>10.2e} | "
                      f"{obj_val:>12.4e} | {space_viol:>10,} | {grad_viol:>10,} | "
                      f"{self.rho:>8.2e} | {elapsed:>5.1f}s")

            # Check convergence
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

            # Adaptive rho
            if self.adaptive_rho and (iteration + 1) % 10 == 0:
                self._update_rho(primal_res, dual_res)

        solve_time = time.time() - start_time

        if self.verbose:
            print(f"  Total solve time: {solve_time:.2f}s")
            print(f"  Final objective: {self._compute_objective():.6e}")

        return self.e

    def _compute_objective(self) -> float:
        """Compute the objective function value."""
        diff = self.e - self.e_star
        if self.norm == 'L2':
            return torch.sum(diff ** 2).item()
        else:
            return torch.sum(torch.abs(diff)).item()


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
        raise ValueError("Must specify either 'rel' or 'abs' in bounds")


def load_config(config_path: str) -> Dict:
    """Load and validate JSON configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    required = ['ground_truth', 'error_field', 'dimensions', 'space_bounds', 'operators']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    if 'm' not in config['dimensions'] or 'n' not in config['dimensions']:
        raise ValueError("dimensions must contain 'm' and 'n'")

    if len(config['operators']) == 0:
        raise ValueError("At least one operator must be specified")

    config.setdefault('norm', 'L2')
    config.setdefault('output_dir', f"result/{config['norm']}_multi_gpu")
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
    parser = argparse.ArgumentParser(
        description='GPU-accelerated multi-operator projection using ADMM.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('config', help='Path to JSON configuration file')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, cuda:0, cuda:1, etc.')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if HAS_CUDA else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("GPU Multi-Operator Projection (ADMM)")
    print("=" * 70)
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    print(f"\n[1/7] Loading configuration from: {args.config}")
    config = load_config(args.config)

    verbose = config['verbose']
    m = config['dimensions']['m']
    n = config['dimensions']['n']
    norm = config['norm']
    admm_config = config['admm']

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
    print(f"  Loaded {m}x{n} fields ({m*n:,} pixels) in {time.time()-start_time:.2f}s")

    # Load operators
    print(f"\n[3/7] Loading {len(config['operators'])} operator(s) and converting to GPU format...")

    operators = []
    operators_np = []  # Keep numpy versions for output computation

    for i, op_config in enumerate(config['operators'], 1):
        op_name = op_config['name']
        op_matrix_path = op_config['matrix']

        if verbose:
            print(f"  [{i}/{len(config['operators'])}] {op_name}: Loading...")

        op_start = time.time()
        A_scipy = load_npz(op_matrix_path)

        if A_scipy.shape[0] != m * n or A_scipy.shape[1] != m * n:
            raise ValueError(f"Matrix {op_name} shape {A_scipy.shape} doesn't match field size")

        # Convert to PyTorch sparse tensor
        A_torch = scipy_sparse_to_torch(A_scipy, device)

        if verbose:
            print(f"      Matrix: {A_scipy.shape[0]:,} x {A_scipy.shape[1]:,}, {A_scipy.nnz:,} non-zeros")

        # Compute ground truth gradient
        Ax = A_scipy @ x
        Ax_star = A_scipy @ x_star

        # Compute error bounds
        bounds_config = op_config['bounds']
        c = compute_error_bounds(Ax, rel=bounds_config.get('rel'), abs_val=bounds_config.get('abs'))

        if verbose:
            print(f"      Error bound: {c[0]:.6e}")
            print(f"      Converted to GPU in {time.time()-op_start:.2f}s")

        operators.append({
            'name': op_name,
            'matrix': A_torch,
            'bounds': torch.from_numpy(c).float(),
        })

        operators_np.append({
            'name': op_name,
            'matrix': A_scipy,
            'bounds': c,
            'ground_truth': Ax,
            'error_field': Ax_star,
        })

    # Compute space error bounds
    print(f"\n[4/7] Computing error bounds...")
    space_config = config['space_bounds']
    b = compute_error_bounds(x, rel=space_config.get('rel'), abs_val=space_config.get('abs'))

    if verbose:
        print(f"  Space error bound: {b[0]:.6e}")

    # Initial error
    e_star = x_star - x

    # Analyze violations before projection
    print(f"\n[5/7] Analyzing constraint violations...")
    space_violations_before = np.sum(np.abs(e_star) > b)

    print(f"\n  === Before Projection ===")
    print(f"  Space domain:")
    print(f"    Violations: {space_violations_before:,} / {len(e_star):,} pixels "
          f"({100*space_violations_before/len(e_star):.4f}%)")
    print(f"    Max error: {np.abs(e_star).max():.6e} (bound: {b[0]:.6e})")

    total_grad_violations_before = 0
    for op in operators_np:
        Ae_star = op['error_field'] - op['ground_truth']
        grad_violations = np.sum(np.abs(Ae_star) > op['bounds'])
        total_grad_violations_before += grad_violations

        print(f"  Operator [{op['name']}]:")
        print(f"    Violations: {grad_violations:,} / {len(Ae_star):,} pixels "
              f"({100*grad_violations/len(Ae_star):.4f}%)")
        print(f"    Max error: {np.abs(Ae_star).max():.6e} (bound: {op['bounds'][0]:.6e})")

    # Check if already feasible
    if space_violations_before == 0 and total_grad_violations_before == 0:
        print("\n  Field already satisfies all constraints. No projection needed.")
        e_proj = e_star
    else:
        # Solve with ADMM
        print(f"\n[6/7] Solving {norm} optimization with ADMM on {device}...")
        print(f"  Problem size: {len(e_star):,} variables")
        print(f"  ADMM parameters: rho={admm_config['rho']}, max_iter={admm_config['max_iter']}")

        # Convert data to torch
        e_star_torch = torch.from_numpy(e_star).float()
        b_torch = torch.from_numpy(b).float()

        # Create solver
        solver = ADMMSolver(
            e_star=e_star_torch,
            operators=operators,
            space_bounds=b_torch,
            norm=norm,
            rho=admm_config['rho'],
            max_iter=admm_config['max_iter'],
            tol_abs=admm_config['tol_abs'],
            tol_rel=admm_config['tol_rel'],
            adaptive_rho=admm_config['adaptive_rho'],
            verbose=verbose,
            print_interval=admm_config['print_interval'],
            device=device
        )

        solve_start = time.time()
        e_proj_torch = solver.solve()
        print(f"  Optimization completed in {time.time()-solve_start:.2f}s")

        # Move result back to CPU
        e_proj = e_proj_torch.cpu().numpy().astype(np.float64)

    # Compute projected field
    print(f"\n  Computing projected results...")
    x_proj = x + e_proj

    # Post-projection stats
    space_violations_after = np.sum(np.abs(e_proj) > b * (1 + 1e-6))

    change_threshold = 1e-10 * np.abs(x_star).max()
    change_mask = (np.abs(x_proj - x_star) > change_threshold).astype(np.float32)
    pixels_changed = int(change_mask.sum())

    l1_distance = np.sum(np.abs(x_proj - x_star))
    l2_distance = np.linalg.norm(x_proj - x_star)

    print(f"\n  === After Projection ===")
    print(f"  Space domain:")
    print(f"    Violations: {space_violations_after:,}")
    print(f"    Max error: {np.abs(e_proj).max():.6e} (bound: {b[0]:.6e})")

    total_grad_violations_after = 0
    for i, op in enumerate(operators_np):
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

    # Projected space error
    path = os.path.join(config['output_dir'], "projected_space_error.f32.raw")
    write_raw_float32(path, e_proj)
    files_written += 1
    if verbose:
        print(f"  [{files_written}] {path}")

    # Per-operator outputs
    for op in operators_np:
        op_name = op['name']

        path = os.path.join(config['output_dir'], f"ground_truth_{op_name}.f32.raw")
        write_raw_float32(path, op['ground_truth'])
        files_written += 1
        if verbose:
            print(f"  [{files_written}] {path}")

        path = os.path.join(config['output_dir'], f"error_{op_name}.f32.raw")
        write_raw_float32(path, op['error_field'])
        files_written += 1
        if verbose:
            print(f"  [{files_written}] {path}")

        grad_diff = op['error_field'] - op['ground_truth']
        path = os.path.join(config['output_dir'], f"{op_name}_diff.f32.raw")
        write_raw_float32(path, grad_diff)
        files_written += 1
        if verbose:
            print(f"  [{files_written}] {path}")

        grad_oob = (np.abs(grad_diff) > op['bounds']).astype(np.float32)
        path = os.path.join(config['output_dir'], f"{op_name}_oob_mask.f32.raw")
        write_raw_float32(path, grad_oob)
        files_written += 1
        if verbose:
            print(f"  [{files_written}] {path}")

        path = os.path.join(config['output_dir'], f"projected_{op_name}.f32.raw")
        write_raw_float32(path, op['projected_field'])
        files_written += 1
        if verbose:
            print(f"  [{files_written}] {path}")

        path = os.path.join(config['output_dir'], f"projected_{op_name}_error.f32.raw")
        write_raw_float32(path, op['projected_error'])
        files_written += 1
        if verbose:
            print(f"  [{files_written}] {path}")

    # Write statistics
    stats_path = os.path.join(config['output_dir'], "projection_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"=== {norm} Multi-Operator Projection (GPU ADMM) ===\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Config file: {args.config}\n")
        f.write(f"  Device: {device}\n")
        if device.type == 'cuda':
            f.write(f"  GPU: {torch.cuda.get_device_name(device)}\n")
        f.write(f"  Ground truth: {config['ground_truth']}\n")
        f.write(f"  Error field: {config['error_field']}\n")
        f.write(f"  Dimensions: {m} x {n}\n")
        f.write(f"  Norm: {norm}\n")
        f.write(f"\nADMM Parameters:\n")
        f.write(f"  rho: {admm_config['rho']}\n")
        f.write(f"  max_iter: {admm_config['max_iter']}\n")
        f.write(f"  tol_abs: {admm_config['tol_abs']}\n")
        f.write(f"  tol_rel: {admm_config['tol_rel']}\n")
        f.write(f"  adaptive_rho: {admm_config['adaptive_rho']}\n")
        f.write(f"\nSpace Bounds:\n")
        if 'rel' in space_config:
            f.write(f"  REL {space_config['rel']} = {b[0]:.6e} absolute\n")
        else:
            f.write(f"  ABS {space_config['abs']}\n")
        f.write(f"\nOperators ({len(operators_np)}):\n")
        for i, op_config in enumerate(config['operators']):
            op = operators_np[i]
            f.write(f"  [{op['name']}] Matrix: {op_config['matrix']}\n")
            bounds_cfg = op_config['bounds']
            if 'rel' in bounds_cfg:
                f.write(f"  [{op['name']}] Bounds: REL {bounds_cfg['rel']} = {op['bounds'][0]:.6e}\n")
            else:
                f.write(f"  [{op['name']}] Bounds: ABS {bounds_cfg['abs']}\n")
        f.write(f"\nBefore Projection:\n")
        f.write(f"  Space violations: {space_violations_before} ({100*space_violations_before/len(e_star):.4f}%)\n")
        f.write(f"  Max space error: {np.abs(e_star).max():.6e}\n")
        for op in operators_np:
            Ae_star = op['error_field'] - op['ground_truth']
            grad_violations = np.sum(np.abs(Ae_star) > op['bounds'])
            f.write(f"  [{op['name']}] violations: {grad_violations}\n")
            f.write(f"  [{op['name']}] max error: {np.abs(Ae_star).max():.6e}\n")
        f.write(f"\nAfter Projection:\n")
        f.write(f"  Space violations: {space_violations_after}\n")
        f.write(f"  Max space error: {np.abs(e_proj).max():.6e}\n")
        for op in operators_np:
            grad_violations = np.sum(np.abs(op['projected_error']) > op['bounds'] * (1 + 1e-6))
            f.write(f"  [{op['name']}] violations: {grad_violations}\n")
            f.write(f"  [{op['name']}] max error: {np.abs(op['projected_error']).max():.6e}\n")
        f.write(f"  Pixels changed: {pixels_changed} ({100*pixels_changed/len(x_star):.4f}%)\n")
        f.write(f"  Projection L1 distance: {l1_distance:.6e}\n")
        f.write(f"  Projection L2 distance: {l2_distance:.6e}\n")

    files_written += 1
    if verbose:
        print(f"  [{files_written}] {stats_path}")

    print(f"  Wrote {files_written} files in {time.time()-write_start:.2f}s")
    print("\n" + "=" * 70)
    print("Projection complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
