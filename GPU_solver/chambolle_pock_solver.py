#!/usr/bin/env python3
"""
Chambolle-Pock (PDHG) solver for L1 projection with linear constraints.

Implements the Primal-Dual Hybrid Gradient algorithm, which is highly efficient
for L1 minimization with linear constraints on GPU.

Problem:
    minimize    ||e - e*||₁
    subject to  -b ≤ e ≤ b           (space bounds)
                -c^(k) ≤ A_k e ≤ c^(k)  (operator bounds)

Reformulated as:
    minimize    f(e) + g(Ke)
    where:
        f(e) = ||e - e*||₁
        K = [I; A_1; ...; A_m]  (stacked constraint matrices)
        g(z) = indicator function of box constraints

Chambolle-Pock iterations:
    y^{k+1} = prox_{σg*}(y^k + σK ē^k)
    e^{k+1} = prox_{τf}(e^k - τK^T y^{k+1})
    ē^{k+1} = e^{k+1} + θ(e^{k+1} - e^k)

where:
    prox_{τf}(v) = e* + soft_threshold(v - e*, τ)
    prox_{σg*}(y) = y - σ·clip(y/σ, lower, upper)

References:
    Chambolle & Pock (2011): "A First-Order Primal-Dual Algorithm for Convex
    Problems with Applications to Imaging"

Usage:
    python GPU_solver/chambolle_pock_solver.py config.json
"""

import argparse
import numpy as np
from scipy.sparse import load_npz, csr_matrix
import os
import json
import time
from typing import Dict, List, Tuple

import torch
import torch.sparse

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
        raise ValueError(f"File size mismatch: expected {m*n}, got {data.size}")
    return data.astype(np.float64)


def write_raw_float32(filepath: str, data: np.ndarray):
    """Write array to raw binary file as float32."""
    data.astype(np.float32).tofile(filepath)


class ChambollePockSolver:
    """
    Chambolle-Pock (PDHG) solver for L1 minimization with linear constraints.

    This is a primal-dual algorithm that's highly efficient for L1 problems
    and naturally parallelizes on GPU.
    """

    def __init__(self,
                 e_star: torch.Tensor,
                 operators: List[Dict],
                 space_bounds: torch.Tensor,
                 tau: float = None,
                 sigma: float = None,
                 theta: float = 1.0,
                 max_iter: int = 5000,
                 tol: float = 1e-6,
                 bound_tol: float = 1e-6,
                 adaptive: bool = True,
                 verbose: bool = True,
                 print_interval: int = 50,
                 device: torch.device = None):
        """
        Initialize Chambolle-Pock solver.

        Args:
            e_star: Target error vector (what we want to stay close to)
            operators: List of dicts with 'matrix' (torch sparse), 'bounds' (tensor)
            space_bounds: Per-element space bounds
            tau: Primal step size (auto-computed if None)
            sigma: Dual step size (auto-computed if None)
            theta: Extrapolation parameter (default 1.0)
            max_iter: Maximum iterations
            tol: Convergence tolerance for primal variable change
            bound_tol: Tolerance for constraint violations (early stop when max violation < bound_tol)
            adaptive: Use adaptive step sizes
            verbose: Print progress
            print_interval: Print every N iterations
            device: torch device
        """
        self.device = device if device else torch.device('cuda' if HAS_CUDA else 'cpu')
        self.n = len(e_star)
        self.theta = theta
        self.max_iter = max_iter
        self.tol = tol
        self.bound_tol = bound_tol
        self.adaptive = adaptive
        self.verbose = verbose
        self.print_interval = print_interval

        # Move data to device
        self.e_star = e_star.to(self.device)
        self.b = space_bounds.to(self.device)

        # Store operators and compute total constraint dimension
        self.operators = []
        self.m_total = self.n  # Start with identity block (space constraints)

        for op in operators:
            op_dict = {
                'name': op['name'],
                'matrix': op['matrix'].to(self.device),
                'bounds': op['bounds'].to(self.device),
                'size': op['matrix'].shape[0]
            }
            self.operators.append(op_dict)
            self.m_total += op_dict['size']

        # Build bound vectors for the stacked system
        # K = [I; A_1; ...; A_m], so bounds are [b; c^(1); ...; c^(m)]
        lower_bounds = [-self.b]
        upper_bounds = [self.b]
        for op in self.operators:
            lower_bounds.append(-op['bounds'])
            upper_bounds.append(op['bounds'])

        self.lower = torch.cat(lower_bounds)
        self.upper = torch.cat(upper_bounds)

        # Estimate operator norm ||K|| for step size selection
        self._estimate_operator_norm()

        # Set step sizes
        if tau is None or sigma is None:
            # For convergence: tau * sigma * ||K||^2 < 1
            # Use tau = sigma = 0.99 / ||K||
            self.tau = 0.99 / self.K_norm
            self.sigma = 0.99 / self.K_norm
        else:
            self.tau = tau
            self.sigma = sigma

        # Initialize primal and dual variables
        self.e = self.e_star.clone()  # Primal variable
        self.e_bar = self.e.clone()   # Extrapolated primal
        self.y = torch.zeros(self.m_total, device=self.device)  # Dual variable

    def _estimate_operator_norm(self):
        """Estimate ||K|| using power iteration."""
        # K = [I; A_1; ...; A_m]
        # K^T K = I + sum_k A_k^T A_k
        # ||K||^2 = ||K^T K|| = largest eigenvalue of K^T K

        v = torch.randn(self.n, device=self.device)
        v = v / torch.norm(v)

        for _ in range(30):
            # Compute K^T K v = v + sum_k A_k^T A_k v
            Kv = v.clone()
            for op in self.operators:
                Av = torch.sparse.mm(op['matrix'], v.unsqueeze(1)).squeeze(1)
                AtAv = torch.sparse.mm(op['matrix'].t(), Av.unsqueeze(1)).squeeze(1)
                Kv = Kv + AtAv

            norm_Kv = torch.norm(Kv)
            if norm_Kv > 1e-10:
                v = Kv / norm_Kv

        # ||K||^2 ≈ v^T (K^T K) v
        Kv = v.clone()
        for op in self.operators:
            Av = torch.sparse.mm(op['matrix'], v.unsqueeze(1)).squeeze(1)
            AtAv = torch.sparse.mm(op['matrix'].t(), Av.unsqueeze(1)).squeeze(1)
            Kv = Kv + AtAv

        self.K_norm_sq = torch.dot(v, Kv).item()
        self.K_norm = np.sqrt(self.K_norm_sq)

        if self.verbose:
            print(f"  Estimated ||K|| = {self.K_norm:.4f}")

    def _apply_K(self, e: torch.Tensor) -> torch.Tensor:
        """Apply K = [I; A_1; ...; A_m] to e."""
        result = [e]
        for op in self.operators:
            Ae = torch.sparse.mm(op['matrix'], e.unsqueeze(1)).squeeze(1)
            result.append(Ae)
        return torch.cat(result)

    def _apply_Kt(self, y: torch.Tensor) -> torch.Tensor:
        """Apply K^T = [I, A_1^T, ..., A_m^T] to y."""
        idx = 0
        result = y[idx:idx + self.n].clone()
        idx += self.n

        for op in self.operators:
            y_k = y[idx:idx + op['size']]
            Aty = torch.sparse.mm(op['matrix'].t(), y_k.unsqueeze(1)).squeeze(1)
            result = result + Aty
            idx += op['size']

        return result

    def _prox_f(self, v: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Proximal operator for f(e) = ||e - e*||_1.

        prox_{τf}(v) = e* + soft_threshold(v - e*, τ)
        """
        diff = v - self.e_star
        shrink = torch.sign(diff) * torch.clamp(torch.abs(diff) - tau, min=0)
        return self.e_star + shrink

    def _prox_g_conj(self, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Proximal operator for σg* where g is indicator of box [lower, upper].

        prox_{σg*}(y) = y - σ · proj_C(y/σ)

        where proj_C is projection onto the box constraints.
        """
        # proj_C(y/σ) = clip(y/σ, lower, upper)
        proj = torch.clamp(y / sigma, self.lower, self.upper)
        return y - sigma * proj

    def _compute_primal_obj(self) -> float:
        """Compute primal objective ||e - e*||_1."""
        return torch.sum(torch.abs(self.e - self.e_star)).item()

    def _compute_constraint_violation(self, tol: float = 0.0) -> Tuple[float, float, int, int]:
        """
        Compute constraint violation and count.

        Args:
            tol: Tolerance for counting violations (violations within tol are not counted)

        Returns:
            total_viol: Sum of all constraint violations
            max_viol: Maximum violation across all constraints
            space_viol: Number of space bound violations (exceeding bound + tol)
            grad_viol: Number of gradient bound violations (exceeding bound + tol)
        """
        Ke = self._apply_K(self.e)

        # Violation amounts
        lower_viol = torch.clamp(self.lower - Ke, min=0)
        upper_viol = torch.clamp(Ke - self.upper, min=0)
        all_viol = lower_viol + upper_viol

        total_viol = torch.sum(all_viol).item()
        max_viol = torch.max(all_viol).item()

        # Count violations exceeding tolerance
        # Space violations: |e_i| > b_i + tol
        space_viol = int(torch.sum(torch.abs(self.e) > self.b * (1 + tol) + tol).item())

        # Gradient violations: |A_k e|_i > c^(k)_i + tol
        grad_viol = 0
        idx = self.n
        for op in self.operators:
            Ae = Ke[idx:idx + op['size']]
            grad_viol += int(torch.sum(torch.abs(Ae) > op['bounds'] * (1 + tol) + tol).item())
            idx += op['size']

        return total_viol, max_viol, space_viol, grad_viol

    def _compute_gap(self) -> float:
        """Compute primal-dual gap for convergence check."""
        Ke = self._apply_K(self.e)
        Kty = self._apply_Kt(self.y)

        # Primal: f(e) + g(Ke)
        primal = torch.sum(torch.abs(self.e - self.e_star))
        # Add indicator (0 if feasible, inf otherwise - we use violation as proxy)
        lower_viol = torch.sum(torch.clamp(self.lower - Ke, min=0))
        upper_viol = torch.sum(torch.clamp(Ke - self.upper, min=0))
        primal = primal + 1e6 * (lower_viol + upper_viol)

        # Dual: -f*(-K^T y) - g*(-y)
        # f*(z) = <e*, z> + indicator(||z||_inf <= 1)
        # g*(y) = sup_{l <= z <= u} <y, z> = sum max(y_i * u_i, y_i * l_i)
        dual_f = torch.sum(self.e_star * Kty)  # Simplified
        dual_g = torch.sum(torch.maximum(self.y * self.upper, self.y * self.lower))
        dual = -dual_f - dual_g

        gap = (primal - dual).item()
        return gap

    def solve(self) -> torch.Tensor:
        """Run Chambolle-Pock algorithm."""
        if self.verbose:
            print(f"  Chambolle-Pock solver on {self.device}")
            print(f"  Variables: {self.n:,}")
            print(f"  Constraints: {self.m_total:,}")
            print(f"  Step sizes: tau={self.tau:.6f}, sigma={self.sigma:.6f}")
            print(f"  theta={self.theta}, bound_tol={self.bound_tol:.2e}")
            print()
            print(f"  {'Iter':>6} | {'Objective':>12} | {'Max Viol':>12} | "
                  f"{'Space Viol':>10} | {'Grad Viol':>10} | {'Time':>6}")
            print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")

        start_time = time.time()

        for iteration in range(self.max_iter):
            e_old = self.e.clone()

            # Dual update: y^{k+1} = prox_{σg*}(y^k + σK ē^k)
            Ke_bar = self._apply_K(self.e_bar)
            self.y = self._prox_g_conj(self.y + self.sigma * Ke_bar, self.sigma)

            # Primal update: e^{k+1} = prox_{τf}(e^k - τK^T y^{k+1})
            Kty = self._apply_Kt(self.y)
            self.e = self._prox_f(self.e - self.tau * Kty, self.tau)

            # Extrapolation: ē^{k+1} = e^{k+1} + θ(e^{k+1} - e^k)
            self.e_bar = self.e + self.theta * (self.e - e_old)

            # Compute violations (with tolerance for counting)
            _, max_viol, space_viol, grad_viol = self._compute_constraint_violation(self.bound_tol)

            # Print progress
            if self.verbose and (iteration + 1) % self.print_interval == 0:
                obj = self._compute_primal_obj()
                elapsed = time.time() - start_time

                print(f"  {iteration+1:>6} | {obj:>12.4e} | {max_viol:>12.4e} | "
                      f"{space_viol:>10,} | {grad_viol:>10,} | {elapsed:>5.1f}s")

            # Early stop: all violations within bound_tol
            if max_viol < self.bound_tol:
                if self.verbose:
                    obj = self._compute_primal_obj()
                    elapsed = time.time() - start_time

                    print(f"  {iteration+1:>6} | {obj:>12.4e} | {max_viol:>12.4e} | "
                          f"{space_viol:>10,} | {grad_viol:>10,} | {elapsed:>5.1f}s")
                    print()
                    print(f"  Early stop: max violation ({max_viol:.2e}) < bound_tol ({self.bound_tol:.2e})")
                break

            # Check convergence based on primal variable change
            e_change = torch.norm(self.e - e_old) / (torch.norm(e_old) + 1e-10)

            if e_change < self.tol and space_viol == 0 and grad_viol == 0:
                if self.verbose:
                    obj = self._compute_primal_obj()
                    elapsed = time.time() - start_time

                    print(f"  {iteration+1:>6} | {obj:>12.4e} | {max_viol:>12.4e} | "
                          f"{space_viol:>10,} | {grad_viol:>10,} | {elapsed:>5.1f}s")
                    print()
                    print(f"  Converged at iteration {iteration + 1} (no violations, stable solution)")
                break

        solve_time = time.time() - start_time

        if self.verbose:
            print(f"  Total solve time: {solve_time:.2f}s")
            print(f"  Final objective: {self._compute_primal_obj():.6e}")

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
    """Load and validate JSON configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    required = ['ground_truth', 'error_field', 'dimensions', 'space_bounds', 'operators']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    config.setdefault('norm', 'L1')
    config.setdefault('output_dir', f"result/CP_L1")
    config.setdefault('verbose', True)
    config.setdefault('chambolle_pock', {})
    config['chambolle_pock'].setdefault('max_iter', 7000)
    config['chambolle_pock'].setdefault('tol', 1e-6)
    config['chambolle_pock'].setdefault('bound_tol', 1e-4)
    config['chambolle_pock'].setdefault('theta', 1.0)
    config['chambolle_pock'].setdefault('adaptive', True)
    config['chambolle_pock'].setdefault('print_interval', 50)

    return config


def main():
    parser = argparse.ArgumentParser(
        description='Chambolle-Pock (PDHG) solver for L1 projection.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('config', help='Path to JSON configuration file')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, cuda:0, etc.')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if HAS_CUDA else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("Chambolle-Pock (PDHG) L1 Projection Solver")
    print("=" * 70)
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    print(f"\n[1/7] Loading configuration from: {args.config}")
    config = load_config(args.config)

    verbose = config['verbose']
    m = config['dimensions']['m']
    n = config['dimensions']['n']
    cp_config = config['chambolle_pock']

    # Override output directory for this solver
    config['output_dir'] = config.get('output_dir', 'result/CP_L1').replace('L1_', 'CP_L1_').replace('L2_', 'CP_L1_')
    if 'CP_' not in config['output_dir']:
        config['output_dir'] = config['output_dir'].replace('result/', 'result/CP_')

    os.makedirs(config['output_dir'], exist_ok=True)

    # Read inputs
    print(f"\n[2/7] Reading input fields...")
    x = read_raw_float32(config['ground_truth'], m, n)
    x_star = read_raw_float32(config['error_field'], m, n)
    print(f"  Loaded {m}x{n} fields ({m*n:,} pixels)")

    # Load operators
    print(f"\n[3/7] Loading {len(config['operators'])} operator(s)...")
    operators = []
    operators_np = []

    for i, op_config in enumerate(config['operators'], 1):
        op_name = op_config['name']
        A_scipy = load_npz(op_config['matrix'])

        if verbose:
            print(f"  [{i}] {op_name}: {A_scipy.shape}, {A_scipy.nnz:,} nnz")

        A_torch = scipy_sparse_to_torch(A_scipy, device)
        Ax = A_scipy @ x
        Ax_star = A_scipy @ x_star

        bounds_config = op_config['bounds']
        c = compute_error_bounds(Ax, rel=bounds_config.get('rel'), abs_val=bounds_config.get('abs'))

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

    # Space bounds
    print(f"\n[4/7] Computing error bounds...")
    space_config = config['space_bounds']
    b = compute_error_bounds(x, rel=space_config.get('rel'), abs_val=space_config.get('abs'))
    if verbose:
        print(f"  Space bound: {b[0]:.6e}")

    e_star = x_star - x

    # Analyze violations
    print(f"\n[5/7] Analyzing constraint violations...")
    space_violations_before = np.sum(np.abs(e_star) > b)

    print(f"\n  === Before Projection ===")
    print(f"  Space domain:")
    print(f"    Violations: {space_violations_before:,} / {len(e_star):,} "
          f"({100*space_violations_before/len(e_star):.4f}%)")
    print(f"    Max error: {np.abs(e_star).max():.6e} (bound: {b[0]:.6e})")

    total_grad_violations = 0
    for op in operators_np:
        Ae = op['error_field'] - op['ground_truth']
        viol = np.sum(np.abs(Ae) > op['bounds'])
        total_grad_violations += viol
        print(f"  Operator [{op['name']}]:")
        print(f"    Violations: {viol:,} ({100*viol/len(Ae):.4f}%)")
        print(f"    Max error: {np.abs(Ae).max():.6e} (bound: {op['bounds'][0]:.6e})")

    if space_violations_before == 0 and total_grad_violations == 0:
        print("\n  Already feasible!")
        e_proj = e_star
    else:
        # Solve
        print(f"\n[6/7] Solving L1 optimization with Chambolle-Pock...")

        e_star_torch = torch.from_numpy(e_star).float()
        b_torch = torch.from_numpy(b).float()

        solver = ChambollePockSolver(
            e_star=e_star_torch,
            operators=operators,
            space_bounds=b_torch,
            max_iter=cp_config['max_iter'],
            tol=cp_config['tol'],
            bound_tol=cp_config['bound_tol'],
            theta=cp_config['theta'],
            adaptive=cp_config['adaptive'],
            verbose=verbose,
            print_interval=cp_config['print_interval'],
            device=device
        )

        solve_start = time.time()
        e_proj_torch = solver.solve()
        print(f"  Optimization completed in {time.time()-solve_start:.2f}s")

        e_proj = e_proj_torch.cpu().numpy().astype(np.float64)

    # Results
    x_proj = x + e_proj

    space_violations_after = np.sum(np.abs(e_proj) > b * (1 + 1e-6))
    change_mask = (np.abs(x_proj - x_star) > 1e-10 * np.abs(x_star).max()).astype(np.float32)
    pixels_changed = int(change_mask.sum())

    l1_dist = np.sum(np.abs(x_proj - x_star))
    l2_dist = np.linalg.norm(x_proj - x_star)

    print(f"\n  === After Projection ===")
    print(f"  Space domain:")
    print(f"    Violations: {space_violations_after}")
    print(f"    Max error: {np.abs(e_proj).max():.6e} (bound: {b[0]:.6e})")

    for op in operators_np:
        Ax_proj = op['matrix'] @ x_proj
        Ae_proj = Ax_proj - op['ground_truth']
        viol = np.sum(np.abs(Ae_proj) > op['bounds'] * (1 + 1e-6))
        op['projected_field'] = Ax_proj
        op['projected_error'] = Ae_proj
        print(f"  Operator [{op['name']}]:")
        print(f"    Violations: {viol}")
        print(f"    Max error: {np.abs(Ae_proj).max():.6e} (bound: {op['bounds'][0]:.6e})")

    print(f"\n  Summary:")
    print(f"    Pixels changed: {pixels_changed:,} ({100*pixels_changed/len(x_star):.4f}%)")
    print(f"    Projection L1 distance: {l1_dist:.6e}")
    print(f"    Projection L2 distance: {l2_dist:.6e}")

    # Write outputs
    print(f"\n[7/7] Writing outputs to {config['output_dir']}...")

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

        grad_diff = op['error_field'] - op['ground_truth']
        write_raw_float32(os.path.join(config['output_dir'], f"{name}_diff.f32.raw"), grad_diff)

        grad_oob = (np.abs(grad_diff) > op['bounds']).astype(np.float32)
        write_raw_float32(os.path.join(config['output_dir'], f"{name}_oob_mask.f32.raw"), grad_oob)

    # Stats file
    stats_path = os.path.join(config['output_dir'], "projection_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"=== Chambolle-Pock L1 Projection ===\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dimensions: {m} x {n}\n")
        f.write(f"Space violations after: {space_violations_after}\n")
        f.write(f"Pixels changed: {pixels_changed}\n")
        f.write(f"L1 distance: {l1_dist:.6e}\n")
        f.write(f"L2 distance: {l2_dist:.6e}\n")

    print(f"  Wrote output files")
    print("\n" + "=" * 70)
    print("Projection complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
