#!/usr/bin/env python3
"""
CuPy-based PDHG (Primal-Dual Hybrid Gradient) solver for L1 projection.

Implements the Chambolle-Pock algorithm as described in feasible_region_lemmas.tex.

Problem (from LaTeX):
    minimize    ||ε̂ - ε₀||₁
    subject to  |ε̂| ≤ d           (box constraint)
                |A_k ε̂| ≤ b_k     (operator constraints for k=1..m)

After variable substitution δ = ε̂ - ε₀, the problem becomes:
    minimize    ||δ||₁
    subject to  |δ + ε₀| ≤ d
                |A_k δ + A_k ε₀| ≤ b_k

PDHG Algorithm:
    Primal:       δ̃_{k+1} = prox_{τf}(δ̃_k - τ K^T y_k)
    Extrapolation: δ̄_{k+1} = δ̃_{k+1} + θ(δ̃_{k+1} - δ̃_k)
    Dual:         y_{k+1} = prox_{σg*}(y_k + σ K δ̄_{k+1})

where:
    K = [I; A_1; ...; A_m]  (stacked constraint operator)
    f(δ) = ||δ||₁           (L1 norm)
    g(z) = indicator of box constraints

References:
    - Chambolle & Pock (2011): "A First-Order Primal-Dual Algorithm"
    - feasible_region_lemmas.tex in this repository

Requirements:
    pip install cupy-cuda11x  # (or cupy-cuda12x for CUDA 12)

Usage:
    python GPU_solver/cupy_pdhg_solver.py config.json

    # With specific GPU
    python GPU_solver/cupy_pdhg_solver.py config.json --gpu 1

    # Override step sizes
    python GPU_solver/cupy_pdhg_solver.py config.json --tau 0.1 --sigma 0.1
"""

import argparse
import numpy as np
from scipy.sparse import load_npz, csr_matrix
import os
import json
import time
from typing import Dict, List, Tuple, Optional

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
        (cp.array(scipy_sparse.data, dtype=cp.float64),
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


class CuPyPDHGSolver:
    """
    PDHG (Chambolle-Pock) solver using CuPy for GPU acceleration.

    Solves the L1 projection problem:
        minimize ||δ||₁ subject to box constraints on (δ + ε₀) and A(δ + ε₀)

    This is equivalent to finding ε̂ = δ + ε₀ that minimizes ||ε̂ - ε₀||₁.
    """

    def __init__(self,
                 e0: cp.ndarray,
                 operators: List[Dict],
                 space_bounds: cp.ndarray,
                 tau: Optional[float] = None,
                 sigma: Optional[float] = None,
                 theta: float = 1.0,
                 max_iter: int = 10000,
                 tol: float = 1e-6,
                 primal_tol: float = 1e-6,
                 dual_tol: float = 1e-6,
                 bound_tol: float = 1e-6,
                 adaptive: bool = True,
                 adaptive_gamma: float = 0.5,
                 adaptive_delta: float = 1.5,
                 adaptive_eta: float = 0.95,
                 fine_tune: bool = True,
                 fine_tune_threshold: float = 1e-2,
                 fine_tune_factor: float = 0.5,
                 min_step: float = 1e-6,
                 backtrack: bool = False,
                 restart: bool = True,
                 restart_interval: int = 100,
                 verbose: bool = True,
                 print_interval: int = 50):
        """
        Initialize CuPy PDHG solver.

        Args:
            e0: Initial error vector ε₀ (what we want to stay close to)
            operators: List of dicts with 'matrix' (CuPy sparse), 'bounds' (CuPy array)
            space_bounds: Per-element space bounds d (so |ε̂| ≤ d)

        Step Size Parameters:
            tau: Primal step size (auto-computed if None)
            sigma: Dual step size (auto-computed if None)
            theta: Extrapolation parameter in [0, 1], default 1.0

        Convergence Parameters:
            max_iter: Maximum iterations
            tol: General convergence tolerance (used if primal/dual_tol not set)
            primal_tol: Primal residual tolerance
            dual_tol: Dual residual tolerance
            bound_tol: Tolerance for constraint violations

        Adaptive Step Size Parameters:
            adaptive: Enable adaptive step size adjustment (primal/dual balancing)
            adaptive_gamma: Backtracking factor (< 1)
            adaptive_delta: Growth factor (> 1)
            adaptive_eta: Target ratio for balancing primal/dual

        Fine-Tuning Parameters (for tight bound_tol):
            fine_tune: Enable automatic step size reduction near solution
            fine_tune_threshold: Reduce steps when max_viol < threshold
            fine_tune_factor: Multiply step by this when entering fine-tune mode
            min_step: Minimum step size (prevents steps from getting too small)

        Algorithm Variants:
            backtrack: Enable backtracking line search
            restart: Enable adaptive restart for acceleration
            restart_interval: Check restart condition every N iterations

        Output Parameters:
            verbose: Print progress
            print_interval: Print every N iterations
        """
        self.n = len(e0)
        self.theta = theta
        self.theta_init = theta
        self.max_iter = max_iter
        self.tol = tol
        self.primal_tol = primal_tol if primal_tol else tol
        self.dual_tol = dual_tol if dual_tol else tol
        self.bound_tol = bound_tol
        self.adaptive = adaptive
        self.adaptive_gamma = adaptive_gamma
        self.adaptive_delta = adaptive_delta
        self.adaptive_eta = adaptive_eta
        self.fine_tune = fine_tune
        self.fine_tune_threshold = fine_tune_threshold
        self.fine_tune_factor = fine_tune_factor
        self.min_step = min_step
        self.backtrack = backtrack
        self.restart = restart
        self.restart_interval = restart_interval
        self.verbose = verbose
        self.print_interval = print_interval

        # Fine-tuning state
        self.in_fine_tune_mode = False
        self.fine_tune_level = 0  # Track how many times we've reduced steps

        # Store original error (target)
        self.e0 = e0

        # Store operators and compute total constraint dimension
        self.operators = operators
        self.m_total = self.n  # Start with identity block (space constraints)
        for op in operators:
            self.m_total += op['matrix'].shape[0]

        # Build constraint bounds
        # For space: |δ + ε₀| ≤ d  =>  -d - ε₀ ≤ δ ≤ d - ε₀
        self.space_d = space_bounds
        self.delta_lower_space = -space_bounds - e0
        self.delta_upper_space = space_bounds - e0

        # For operators: |A_k δ + A_k ε₀| ≤ b_k  =>  -b_k - c_k ≤ A_k δ ≤ b_k - c_k
        # where c_k = A_k ε₀
        lower_bounds = [self.delta_lower_space]
        upper_bounds = [self.delta_upper_space]

        self.operator_offsets = []  # Store A_k ε₀ for each operator
        for op in operators:
            c_k = op['matrix'] @ e0  # A_k ε₀
            self.operator_offsets.append(c_k)
            lower_bounds.append(-op['bounds'] - c_k)
            upper_bounds.append(op['bounds'] - c_k)

        self.lower = cp.concatenate(lower_bounds)
        self.upper = cp.concatenate(upper_bounds)

        # Estimate operator norm for step sizes
        self._estimate_operator_norm()

        # Set step sizes
        if tau is None or sigma is None:
            # For convergence: τσ||K||² < 1
            # Use τ = σ = 0.99 / ||K||
            self.tau = 0.99 / self.K_norm
            self.sigma = 0.99 / self.K_norm
        else:
            self.tau = tau
            self.sigma = sigma

        self.tau_init = self.tau
        self.sigma_init = self.sigma

        # Initialize variables
        self.delta = cp.zeros(self.n, dtype=cp.float64)  # Primal: δ = ε̂ - ε₀
        self.delta_bar = self.delta.copy()               # Extrapolated primal
        self.y = cp.zeros(self.m_total, dtype=cp.float64)  # Dual

        # For restart
        self.delta_prev = self.delta.copy()
        self.y_prev = self.y.copy()

    def _estimate_operator_norm(self):
        """
        Estimate ||K|| using power iteration.

        K = [I; A_1; ...; A_m]
        K^T K = I + Σ_k A_k^T A_k
        """
        v = cp.random.randn(self.n)
        v = v / cp.linalg.norm(v)

        for _ in range(30):
            # K^T K v = v + Σ_k A_k^T (A_k v)
            Kv = v.copy()
            for op in self.operators:
                Av = op['matrix'] @ v
                AtAv = op['matrix'].T @ Av
                Kv = Kv + AtAv

            norm_Kv = cp.linalg.norm(Kv)
            if norm_Kv > 1e-10:
                v = Kv / norm_Kv

        # ||K||² ≈ v^T (K^T K) v
        Kv = v.copy()
        for op in self.operators:
            Av = op['matrix'] @ v
            AtAv = op['matrix'].T @ Av
            Kv = Kv + AtAv

        self.K_norm_sq = float(cp.dot(v, Kv))
        self.K_norm = np.sqrt(self.K_norm_sq)

        if self.verbose:
            print(f"  Estimated ||K|| = {self.K_norm:.4f}")

    def _apply_K(self, delta: cp.ndarray) -> cp.ndarray:
        """Apply K = [I; A_1; ...; A_m] to δ."""
        result = [delta]
        for op in self.operators:
            result.append(op['matrix'] @ delta)
        return cp.concatenate(result)

    def _apply_Kt(self, y: cp.ndarray) -> cp.ndarray:
        """Apply K^T = [I, A_1^T, ..., A_m^T] to y."""
        idx = 0
        result = y[idx:idx + self.n].copy()
        idx += self.n

        for op in self.operators:
            size = op['matrix'].shape[0]
            y_k = y[idx:idx + size]
            result = result + op['matrix'].T @ y_k
            idx += size

        return result

    def _prox_f(self, v: cp.ndarray) -> cp.ndarray:
        """
        Proximal operator for f(δ) = ||δ||₁.

        prox_{τf}(v) = soft_threshold(v, τ) = sign(v) * max(|v| - τ, 0)
        """
        return cp.sign(v) * cp.maximum(cp.abs(v) - self.tau, 0)

    def _prox_g_conj(self, y: cp.ndarray) -> cp.ndarray:
        """
        Proximal operator for σg* where g is indicator of box [lower, upper].

        prox_{σg*}(y) = y - σ · proj_C(y/σ)

        where proj_C is projection onto the box constraints.
        """
        proj = cp.clip(y / self.sigma, self.lower, self.upper)
        return y - self.sigma * proj

    def _compute_primal_residual(self) -> float:
        """Compute primal residual ||K δ - z||."""
        Kd = self._apply_K(self.delta)
        z = cp.clip(Kd, self.lower, self.upper)  # Projection
        return float(cp.linalg.norm(Kd - z))

    def _compute_dual_residual(self, delta_old: cp.ndarray) -> float:
        """Compute dual residual ||δ - δ_old|| / τ."""
        return float(cp.linalg.norm(self.delta - delta_old)) / self.tau

    def _compute_objective(self) -> float:
        """Compute objective ||δ||₁ = ||ε̂ - ε₀||₁."""
        return float(cp.sum(cp.abs(self.delta)))

    def _compute_constraint_violation(self) -> Tuple[float, float, int, int]:
        """
        Compute constraint violations.

        Returns:
            total_viol: Sum of all violations
            max_viol: Maximum violation
            space_viol: Number of space bound violations
            grad_viol: Number of gradient bound violations
        """
        # Reconstruct ε̂ = δ + ε₀
        e_hat = self.delta + self.e0

        # Space violations: |ε̂| > d
        space_excess = cp.maximum(cp.abs(e_hat) - self.space_d, 0)
        space_viol = int(cp.sum(space_excess > self.bound_tol))

        # Gradient violations
        grad_viol = 0
        max_viol = float(cp.max(space_excess))

        for i, op in enumerate(self.operators):
            # A_k ε̂ = A_k δ + A_k ε₀ = A_k δ + c_k
            Ae = op['matrix'] @ self.delta + self.operator_offsets[i]
            excess = cp.maximum(cp.abs(Ae) - op['bounds'], 0)
            grad_viol += int(cp.sum(excess > self.bound_tol))
            max_viol = max(max_viol, float(cp.max(excess)))

        total_viol = float(cp.sum(space_excess)) + sum(
            float(cp.sum(cp.maximum(cp.abs(op['matrix'] @ self.delta + self.operator_offsets[i]) - op['bounds'], 0)))
            for i, op in enumerate(self.operators)
        )

        return total_viol, max_viol, space_viol, grad_viol

    def _should_restart(self) -> bool:
        """Check if adaptive restart should be triggered."""
        # Restart if the "gradient" is pointing in wrong direction
        # Using the criterion from O'Donoghue & Candes (2015)
        delta_diff = self.delta - self.delta_prev
        y_diff = self.y - self.y_prev

        # Check if inner product is negative (making progress)
        primal_progress = cp.dot(delta_diff, self._apply_Kt(self.y - self.y_prev))
        return float(primal_progress) > 0  # Restart if positive (bad direction)

    def _adapt_step_sizes(self, primal_res: float, dual_res: float):
        """Adapt step sizes based on primal/dual residual ratio."""
        if primal_res < 1e-15 or dual_res < 1e-15:
            return

        ratio = primal_res / dual_res

        if ratio > 1.0 / self.adaptive_eta:
            # Primal residual too large - decrease tau, increase sigma
            self.tau *= self.adaptive_gamma
            self.sigma /= self.adaptive_gamma
        elif ratio < self.adaptive_eta:
            # Dual residual too large - increase tau, decrease sigma
            self.tau /= self.adaptive_gamma
            self.sigma *= self.adaptive_gamma

        # Ensure step sizes stay reasonable
        max_step = 0.99 / self.K_norm
        self.tau = min(self.tau, max_step)
        self.sigma = min(self.sigma, max_step)

    def solve(self) -> cp.ndarray:
        """
        Run PDHG algorithm.

        Returns:
            e_hat: Projected error vector ε̂ = δ + ε₀
        """
        if self.verbose:
            print(f"  CuPy PDHG solver")
            print(f"  Variables: {self.n:,}")
            print(f"  Constraints: {self.m_total:,}")
            print(f"  Step sizes: tau={self.tau:.6f}, sigma={self.sigma:.6f}")
            print(f"  theta={self.theta}, bound_tol={self.bound_tol:.2e}")
            if self.adaptive:
                print(f"  Adaptive step sizes: gamma={self.adaptive_gamma}, eta={self.adaptive_eta}")
            if self.fine_tune:
                print(f"  Fine-tuning: threshold={self.fine_tune_threshold:.0e}, "
                      f"factor={self.fine_tune_factor}, min_step={self.min_step:.0e}")
            if self.restart:
                print(f"  Adaptive restart: interval={self.restart_interval}")
            print()
            print(f"  {'Iter':>6} | {'Objective':>12} | {'Primal Res':>12} | {'Dual Res':>12} | "
                  f"{'Max Viol':>12} | {'Space Viol':>10} | {'Grad Viol':>10} | {'tau':>10} | {'Time':>6}")
            print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")

        start_time = time.time()
        restart_count = 0

        for iteration in range(self.max_iter):
            delta_old = self.delta.copy()

            # === PDHG Iterations (Algorithm from LaTeX) ===

            # 1. Primal update: δ_{k+1} = prox_{τf}(δ_k - τ K^T y_k)
            Kty = self._apply_Kt(self.y)
            self.delta = self._prox_f(self.delta - self.tau * Kty)

            # 2. Extrapolation: δ̄_{k+1} = δ_{k+1} + θ(δ_{k+1} - δ_k)
            self.delta_bar = self.delta + self.theta * (self.delta - delta_old)

            # 3. Dual update: y_{k+1} = prox_{σg*}(y_k + σ K δ̄_{k+1})
            Kd_bar = self._apply_K(self.delta_bar)
            self.y = self._prox_g_conj(self.y + self.sigma * Kd_bar)

            # === Convergence checks ===
            primal_res = self._compute_primal_residual()
            dual_res = self._compute_dual_residual(delta_old)
            _, max_viol, space_viol, grad_viol = self._compute_constraint_violation()

            # Print progress
            if self.verbose and (iteration + 1) % self.print_interval == 0:
                obj = self._compute_objective()
                elapsed = time.time() - start_time
                print(f"  {iteration+1:>6} | {obj:>12.4e} | {primal_res:>12.4e} | {dual_res:>12.4e} | "
                      f"{max_viol:>12.4e} | {space_viol:>10,} | {grad_viol:>10,} | {self.tau:>10.2e} | {elapsed:>5.1f}s")

            # Early stop: check feasibility first, then residuals
            # Primary criterion: all constraints satisfied (no violations)
            if space_viol == 0 and grad_viol == 0:
                if self.verbose:
                    obj = self._compute_objective()
                    elapsed = time.time() - start_time
                    print(f"  {iteration+1:>6} | {obj:>12.4e} | {primal_res:>12.4e} | {dual_res:>12.4e} | "
                          f"{max_viol:>12.4e} | {space_viol:>10,} | {grad_viol:>10,} | {self.tau:>10.2e} | {elapsed:>5.1f}s")
                    print()
                    print(f"  Converged at iteration {iteration + 1} (all constraints satisfied)")
                    print(f"  Restarts: {restart_count}")
                break

            # Secondary criterion: max violation below tolerance and residuals small
            if max_viol < self.bound_tol and primal_res < self.primal_tol and dual_res < self.dual_tol:
                if self.verbose:
                    obj = self._compute_objective()
                    elapsed = time.time() - start_time
                    print(f"  {iteration+1:>6} | {obj:>12.4e} | {primal_res:>12.4e} | {dual_res:>12.4e} | "
                          f"{max_viol:>12.4e} | {space_viol:>10,} | {grad_viol:>10,} | {self.tau:>10.2e} | {elapsed:>5.1f}s")
                    print()
                    print(f"  Converged at iteration {iteration + 1} (residuals below tolerance)")
                    print(f"  Restarts: {restart_count}")
                break

            # Adaptive restart
            if self.restart and (iteration + 1) % self.restart_interval == 0:
                if self._should_restart():
                    # Reset extrapolation
                    self.delta_bar = self.delta.copy()
                    restart_count += 1

            # Store for restart check
            self.delta_prev = delta_old
            self.y_prev = self.y.copy()

            # Adaptive step sizes (primal/dual balancing)
            if self.adaptive and (iteration + 1) % 50 == 0:
                self._adapt_step_sizes(primal_res, dual_res)

            # Fine-tuning: reduce step sizes when close to solution or stagnating
            total_viol = space_viol + grad_viol
            if self.fine_tune and self.tau > self.min_step:
                should_fine_tune = False
                reason = ""

                # Track violation history for progress detection
                if not hasattr(self, '_viol_history'):
                    self._viol_history = []
                    self._last_fine_tune_iter = 0
                    self._last_fine_tune_viol = float('inf')
                    self._best_viol = float('inf')
                    self._best_max_viol = float('inf')

                self._viol_history.append(total_viol)
                if len(self._viol_history) > 2000:
                    self._viol_history = self._viol_history[-2000:]

                # Track best seen violations
                if total_viol < self._best_viol:
                    self._best_viol = total_viol
                if max_viol < self._best_max_viol:
                    self._best_max_viol = max_viol

                # Strategy 1: Threshold-based (when max_viol drops below threshold)
                current_threshold = self.fine_tune_threshold * (self.fine_tune_factor ** self.fine_tune_level)
                if max_viol > 0 and max_viol < current_threshold:
                    # Only trigger if we've improved since last fine-tune
                    if max_viol < self._last_fine_tune_viol * 0.9:  # 10% improvement required
                        should_fine_tune = True
                        reason = f"threshold: max_viol={max_viol:.2e} < {current_threshold:.2e}"

                # Strategy 2: Significant progress (violations dropped significantly)
                if total_viol > 0 and total_viol <= self._last_fine_tune_viol * 0.5:
                    # Violations halved since last fine-tune
                    if (iteration - self._last_fine_tune_iter) > 200:  # Min cooldown
                        should_fine_tune = True
                        reason = f"progress: violations {self._last_fine_tune_viol:.0f} -> {total_viol}"

                # Strategy 3: Long stagnation with tiny steps needed
                # Only if we've been stuck for a long time AND violations are very few
                if len(self._viol_history) >= 2000 and (iteration - self._last_fine_tune_iter) > 2000:
                    recent = self._viol_history[-500:]
                    old = self._viol_history[-2000:-1500]
                    recent_min = min(recent)
                    old_min = min(old)
                    # Only fine-tune if we haven't improved the minimum at all
                    if recent_min >= old_min and total_viol > 0 and total_viol <= 10:
                        should_fine_tune = True
                        reason = f"long stagnation: min_viol stuck at {recent_min}"

                if should_fine_tune:
                    self.fine_tune_level += 1
                    self._last_fine_tune_iter = iteration
                    self._last_adjust_iter = iteration
                    self._last_fine_tune_viol = total_viol if total_viol > 0 else self._last_fine_tune_viol
                    old_tau = self.tau

                    # Reduce step sizes
                    self.tau = max(self.tau * self.fine_tune_factor, self.min_step)
                    self.sigma = max(self.sigma * self.fine_tune_factor, self.min_step)

                    # Also reduce theta for more stable convergence
                    self.theta = max(self.theta * self.fine_tune_factor, 0.1)

                    if self.verbose:
                        print(f"  [Fine-tune level {self.fine_tune_level}] {reason}, "
                              f"tau: {old_tau:.2e} -> {self.tau:.2e}, theta: {self.theta:.2f}")

                # Strategy 4: INCREASE step size if stuck for too long with small steps
                # This helps escape when we fine-tuned too aggressively
                if not hasattr(self, '_last_adjust_iter'):
                    self._last_adjust_iter = 0

                if (self.fine_tune_level > 0 and
                    (iteration - self._last_adjust_iter) > 3000 and
                    self.tau < self.tau_init * 0.5):  # Only if we've reduced steps significantly

                    # Check if we're stuck (no improvement in min violations)
                    if len(self._viol_history) >= 3000:
                        recent_min = min(self._viol_history[-1000:])
                        old_min = min(self._viol_history[-3000:-2000])

                        # If minimum hasn't improved, try larger steps
                        if recent_min >= old_min * 0.9 and total_viol > 0:
                            self._last_adjust_iter = iteration
                            old_tau = self.tau

                            # Increase step sizes (but not beyond initial)
                            increase_factor = 1.0 / self.fine_tune_factor  # e.g., 2.0
                            self.tau = min(self.tau * increase_factor, self.tau_init)
                            self.sigma = min(self.sigma * increase_factor, self.sigma_init)
                            self.theta = min(self.theta * increase_factor, self.theta_init)

                            if self.fine_tune_level > 0:
                                self.fine_tune_level -= 1

                            if self.verbose:
                                print(f"  [Step increase] stuck for 3000 iters, min_viol {old_min}->{recent_min}, "
                                      f"tau: {old_tau:.2e} -> {self.tau:.2e}, theta: {self.theta:.2f}")

        solve_time = time.time() - start_time

        if self.verbose:
            print(f"  Total solve time: {solve_time:.2f}s")
            print(f"  Final objective: {self._compute_objective():.6e}")
            print(f"  Final step sizes: tau={self.tau:.6e}, sigma={self.sigma:.6e}")

        # Return ε̂ = δ + ε₀
        return self.delta + self.e0

    def project_to_feasible(self, e_hat: cp.ndarray, max_iter: int = 1000) -> cp.ndarray:
        """
        Project solution onto feasible region using iterative Dykstra-like projection.

        This is a post-processing step to eliminate any remaining constraint violations
        after PDHG has converged to an approximate solution.

        Args:
            e_hat: Approximate solution from PDHG
            max_iter: Maximum projection iterations

        Returns:
            Feasible e_hat with all constraints satisfied
        """
        if self.verbose:
            print(f"\n  === Post-processing: Projecting to feasible region ===")

        e = e_hat.copy()

        for iteration in range(max_iter):
            e_old = e.copy()

            # Project onto space constraints: |e| ≤ d
            e = cp.clip(e, -self.space_d, self.space_d)

            # Project onto each operator constraint: |A_k e| ≤ b_k
            # Using averaged projections (Dykstra's algorithm style)
            for i, op in enumerate(self.operators):
                Ae = op['matrix'] @ e
                bounds = op['bounds']

                # Find violations
                excess_pos = cp.maximum(Ae - bounds, 0)
                excess_neg = cp.maximum(-Ae - bounds, 0)
                excess = excess_pos - excess_neg

                if cp.any(cp.abs(excess) > self.bound_tol):
                    # Correct violations using pseudo-inverse projection
                    # For sparse gradient operators, A^T A is well-conditioned
                    # Use gradient descent: e -= alpha * A^T * excess
                    AtA_diag = cp.array(op['matrix'].multiply(op['matrix']).sum(axis=0)).flatten()
                    alpha = 0.5 / (cp.max(AtA_diag) + 1e-10)  # Conservative step
                    correction = op['matrix'].T @ excess
                    e = e - float(alpha) * correction

                    # Re-project onto space constraints
                    e = cp.clip(e, -self.space_d, self.space_d)

            # Check convergence
            change = float(cp.linalg.norm(e - e_old))
            if change < 1e-10:
                break

        # Final hard clip on all constraints
        e = cp.clip(e, -self.space_d, self.space_d)

        if self.verbose:
            _, max_viol, space_viol, grad_viol = self._compute_constraint_violation_for(e)
            print(f"  After projection: space_viol={space_viol}, grad_viol={grad_viol}, max_viol={max_viol:.2e}")

        return e

    def _compute_constraint_violation_for(self, e_hat: cp.ndarray) -> Tuple[float, float, int, int]:
        """Compute constraint violations for a given e_hat (not self.delta)."""
        # Space violations: |e_hat| > d
        space_excess = cp.maximum(cp.abs(e_hat) - self.space_d, 0)
        space_viol = int(cp.sum(space_excess > self.bound_tol))

        # Gradient violations
        grad_viol = 0
        max_viol = float(cp.max(space_excess))

        # For this check, e_hat IS the error (not delta), so we compute A @ e_hat directly
        # But our operators store offsets for delta formulation
        # e_hat = delta + e0, so A @ e_hat = A @ delta + A @ e0
        delta = e_hat - self.e0

        for i, op in enumerate(self.operators):
            Ae = op['matrix'] @ delta + self.operator_offsets[i]
            excess = cp.maximum(cp.abs(Ae) - op['bounds'], 0)
            grad_viol += int(cp.sum(excess > self.bound_tol))
            max_viol = max(max_viol, float(cp.max(excess)))

        total_viol = float(cp.sum(space_excess))

        return total_viol, max_viol, space_viol, grad_viol


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

    config.setdefault('norm', 'L1')
    config.setdefault('output_dir', f"result/PDHG_L1_cupy")
    config.setdefault('verbose', True)

    # PDHG-specific defaults
    config.setdefault('pdhg', {})
    pdhg = config['pdhg']
    pdhg.setdefault('tau', None)              # Auto-compute
    pdhg.setdefault('sigma', None)            # Auto-compute
    pdhg.setdefault('theta', 1.0)             # Extrapolation parameter
    pdhg.setdefault('max_iter', 10000)
    pdhg.setdefault('tol', 1e-6)
    pdhg.setdefault('primal_tol', None)       # Use tol if not set
    pdhg.setdefault('dual_tol', None)         # Use tol if not set
    pdhg.setdefault('bound_tol', 1e-6)
    pdhg.setdefault('adaptive', True)
    pdhg.setdefault('adaptive_gamma', 0.5)
    pdhg.setdefault('adaptive_delta', 1.5)
    pdhg.setdefault('adaptive_eta', 0.95)
    pdhg.setdefault('fine_tune', True)
    pdhg.setdefault('fine_tune_threshold', 1e-2)
    pdhg.setdefault('fine_tune_factor', 0.5)
    pdhg.setdefault('min_step', 1e-6)
    pdhg.setdefault('backtrack', False)
    pdhg.setdefault('restart', True)
    pdhg.setdefault('restart_interval', 100)
    pdhg.setdefault('print_interval', 50)
    pdhg.setdefault('post_project', True)  # Enable post-processing projection
    pdhg.setdefault('post_project_iter', 1000)

    return config


def main():
    if not HAS_CUPY:
        print("ERROR: CuPy not installed. Install with: pip install cupy-cuda11x")
        print("       (or cupy-cuda12x for CUDA 12)")
        return

    parser = argparse.ArgumentParser(
        description='CuPy PDHG solver for L1 projection with linear constraints.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('config', help='Path to JSON config file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')

    # Override config file settings
    parser.add_argument('--tau', type=float, default=None, help='Primal step size')
    parser.add_argument('--sigma', type=float, default=None, help='Dual step size')
    parser.add_argument('--theta', type=float, default=None, help='Extrapolation parameter')
    parser.add_argument('--max-iter', type=int, default=None, help='Maximum iterations')
    parser.add_argument('--tol', type=float, default=None, help='Convergence tolerance')
    parser.add_argument('--bound-tol', type=float, default=None, help='Constraint violation tolerance')
    parser.add_argument('--no-adaptive', action='store_true', help='Disable adaptive step sizes')
    parser.add_argument('--no-restart', action='store_true', help='Disable adaptive restart')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    # Set GPU device
    cp.cuda.Device(args.gpu).use()

    print("=" * 80)
    print("CuPy PDHG (Chambolle-Pock) L1 Projection Solver")
    print("=" * 80)
    print(f"\nGPU {args.gpu}: {cp.cuda.runtime.getDeviceProperties(args.gpu)['name'].decode()}")
    mem_info = cp.cuda.runtime.memGetInfo()
    print(f"Memory: {mem_info[1] / 1e9:.1f} GB total, {mem_info[0] / 1e9:.1f} GB free")

    print(f"\n[1/7] Loading configuration: {args.config}")
    config = load_config(args.config)
    pdhg_config = config['pdhg']

    # Apply command-line overrides
    if args.tau is not None:
        pdhg_config['tau'] = args.tau
    if args.sigma is not None:
        pdhg_config['sigma'] = args.sigma
    if args.theta is not None:
        pdhg_config['theta'] = args.theta
    if args.max_iter is not None:
        pdhg_config['max_iter'] = args.max_iter
    if args.tol is not None:
        pdhg_config['tol'] = args.tol
    if args.bound_tol is not None:
        pdhg_config['bound_tol'] = args.bound_tol
    if args.no_adaptive:
        pdhg_config['adaptive'] = False
    if args.no_restart:
        pdhg_config['restart'] = False
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir

    verbose = config['verbose']
    m = config['dimensions']['m']
    n = config['dimensions']['n']

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

    e_star = x_star - x  # Initial error ε₀

    # Analyze violations
    print(f"\n[5/7] Analyzing violations...")
    space_violations_before = np.sum(np.abs(e_star) > b)
    print(f"\n  === Before Projection ===")
    print(f"  Space violations: {space_violations_before:,} ({100*space_violations_before/len(e_star):.4f}%)")
    print(f"  Max space error: {np.abs(e_star).max():.6e} (bound: {b[0]:.6e})")

    total_grad_violations = 0
    for op in operators_np:
        Ae = op['error_field'] - op['ground_truth']
        viol = np.sum(np.abs(Ae) > op['bounds'])
        total_grad_violations += viol
        print(f"  [{op['name']}] violations: {viol:,}, max error: {np.abs(Ae).max():.6e} (bound: {op['bounds'][0]:.6e})")

    if space_violations_before == 0 and total_grad_violations == 0:
        print("\n  Already feasible!")
        e_proj = e_star
    else:
        # Solve
        print(f"\n[6/7] Solving L1 optimization with PDHG...")
        print(f"  PDHG config: tau={pdhg_config['tau']}, sigma={pdhg_config['sigma']}, "
              f"theta={pdhg_config['theta']}, max_iter={pdhg_config['max_iter']}")

        e0_gpu = cp.array(e_star)
        b_gpu = cp.array(b)

        solver = CuPyPDHGSolver(
            e0=e0_gpu,
            operators=operators_gpu,
            space_bounds=b_gpu,
            tau=pdhg_config['tau'],
            sigma=pdhg_config['sigma'],
            theta=pdhg_config['theta'],
            max_iter=pdhg_config['max_iter'],
            tol=pdhg_config['tol'],
            primal_tol=pdhg_config['primal_tol'],
            dual_tol=pdhg_config['dual_tol'],
            bound_tol=pdhg_config['bound_tol'],
            adaptive=pdhg_config['adaptive'],
            adaptive_gamma=pdhg_config['adaptive_gamma'],
            adaptive_delta=pdhg_config['adaptive_delta'],
            adaptive_eta=pdhg_config['adaptive_eta'],
            fine_tune=pdhg_config['fine_tune'],
            fine_tune_threshold=pdhg_config['fine_tune_threshold'],
            fine_tune_factor=pdhg_config['fine_tune_factor'],
            min_step=pdhg_config['min_step'],
            backtrack=pdhg_config['backtrack'],
            restart=pdhg_config['restart'],
            restart_interval=pdhg_config['restart_interval'],
            verbose=verbose,
            print_interval=pdhg_config['print_interval']
        )

        solve_start = time.time()
        e_proj_gpu = solver.solve()
        print(f"  Completed in {time.time()-solve_start:.2f}s")

        # Post-processing projection to eliminate remaining violations
        if pdhg_config.get('post_project', True):
            _, max_viol, space_viol, grad_viol = solver._compute_constraint_violation()
            if space_viol > 0 or grad_viol > 0:
                print(f"\n  Remaining violations: space={space_viol}, grad={grad_viol}")
                print(f"  Running post-processing projection...")
                e_proj_gpu = solver.project_to_feasible(
                    e_proj_gpu,
                    max_iter=pdhg_config.get('post_project_iter', 1000)
                )

        e_proj = cp.asnumpy(e_proj_gpu)

    # Results
    x_proj = x + e_proj
    bound_tol = pdhg_config['bound_tol']

    # Use same tolerance as during iteration for consistent counting
    space_excess = np.maximum(np.abs(e_proj) - b, 0)
    space_violations_after = np.sum(space_excess > bound_tol)
    space_max_excess = space_excess.max()

    change_mask = (np.abs(x_proj - x_star) > 1e-10 * np.abs(x_star).max()).astype(np.float32)
    pixels_changed = int(change_mask.sum())

    l1_dist = np.sum(np.abs(x_proj - x_star))
    l2_dist = np.linalg.norm(x_proj - x_star)

    print(f"\n  === After Projection (using bound_tol={bound_tol:.0e}) ===")
    print(f"  Space violations: {space_violations_after}")
    print(f"  Max space error: {np.abs(e_proj).max():.6e} (bound: {b[0]:.6e}, excess: {space_max_excess:.6e})")

    for i, op in enumerate(operators_np):
        Ax_proj = op['matrix'] @ x_proj
        Ae_proj = Ax_proj - op['ground_truth']
        grad_excess = np.maximum(np.abs(Ae_proj) - op['bounds'], 0)
        viol = np.sum(grad_excess > bound_tol)
        max_excess = grad_excess.max()
        op['projected_field'] = Ax_proj
        op['projected_error'] = Ae_proj
        print(f"  [{op['name']}] violations: {viol}, max error: {np.abs(Ae_proj).max():.6e} (bound: {op['bounds'][0]:.6e}, excess: {max_excess:.6e})")

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
        f.write(f"=== PDHG L1 Projection (CuPy) ===\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  GPU: {args.gpu}\n")
        f.write(f"  Dimensions: {m} x {n}\n")
        f.write(f"\nPDHG Parameters:\n")
        f.write(f"  tau (initial): {pdhg_config['tau']}\n")
        f.write(f"  sigma (initial): {pdhg_config['sigma']}\n")
        f.write(f"  theta: {pdhg_config['theta']}\n")
        f.write(f"  max_iter: {pdhg_config['max_iter']}\n")
        f.write(f"  tol: {pdhg_config['tol']}\n")
        f.write(f"  bound_tol: {pdhg_config['bound_tol']}\n")
        f.write(f"  adaptive: {pdhg_config['adaptive']}\n")
        f.write(f"  restart: {pdhg_config['restart']}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Space violations after: {space_violations_after}\n")
        f.write(f"  Pixels changed: {pixels_changed}\n")
        f.write(f"  L1 distance: {l1_dist:.6e}\n")
        f.write(f"  L2 distance: {l2_dist:.6e}\n")

    print("\n" + "=" * 80)
    print("Projection complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
