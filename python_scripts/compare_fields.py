#!/usr/bin/env python3
"""
Compare two raw binary fields and analyze error bounds.

Usage:
    python python_scripts/compare_fields.py <ground_truth> <test_field> <m> <n> <output_prefix> [--rel <rel_bound>]

Examples:
    # Compute difference only
    python python_scripts/compare_fields.py result/Nyx_original_cd_grad_x.raw result/Nyx_rel_1e-3_cd_grad_x.raw 512 512 result/compare_grad_x

    # Compute difference and out-of-bound mask with REL 1e-3
    python python_scripts/compare_fields.py result/Nyx_original_cd_grad_x.raw result/Nyx_rel_1e-3_cd_grad_x.raw 512 512 result/compare_grad_x --rel 1e-3

Output:
    <output_prefix>_diff.f32.raw      - Difference field (test - ground_truth), float32
    <output_prefix>_oob_mask.f32.raw  - Out-of-bound mask (1.0=exceeded, 0.0=within), float32 (only if --rel specified)
"""

import argparse
import numpy as np
import sys


def read_raw_float32(filepath: str, m: int, n: int) -> np.ndarray:
    """Read raw binary file as float32 array with shape (m, n)."""
    data = np.fromfile(filepath, dtype=np.float32)
    if data.size != m * n:
        raise ValueError(f"File size mismatch: expected {m*n} floats, got {data.size}")
    return data.reshape((m, n))


def write_raw_float32(filepath: str, data: np.ndarray):
    """Write float32 array to raw binary file."""
    data.astype(np.float32).tofile(filepath)




def compare_fields(ground_truth: np.ndarray, test_field: np.ndarray, rel_bound: float = None):
    """
    Compare two fields and compute error statistics.

    Args:
        ground_truth: Reference field
        test_field: Field to compare against reference
        rel_bound: Relative error bound (same as SZ3: rel_bound * value_range)

    Returns:
        diff: Difference field (test - ground_truth)
        oob_mask: Out-of-bound mask (1=exceeded, 0=within), None if rel_bound not specified
        stats: Dictionary of error statistics
    """
    diff = test_field - ground_truth

    # Compute statistics
    value_range = ground_truth.max() - ground_truth.min()
    abs_diff = np.abs(diff)

    stats = {
        'value_range': value_range,
        'min_error': abs_diff.min(),
        'max_error': abs_diff.max(),
        'mean_error': abs_diff.mean(),
        'rmse': np.sqrt(np.mean(diff**2)),
        'max_rel_error': abs_diff.max() / value_range if value_range > 0 else float('inf'),
    }

    oob_mask = None
    if rel_bound is not None:
        abs_error_bound = rel_bound * value_range
        stats['abs_error_bound'] = abs_error_bound
        oob_mask = (abs_diff > abs_error_bound).astype(np.float32)
        stats['oob_count'] = int(oob_mask.sum())
        stats['oob_percent'] = 100.0 * stats['oob_count'] / oob_mask.size

    return diff, oob_mask, stats


def main():
    parser = argparse.ArgumentParser(
        description='Compare two raw binary fields and analyze error bounds.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('ground_truth', help='Path to ground truth field (float32 raw)')
    parser.add_argument('test_field', help='Path to test field (float32 raw)')
    parser.add_argument('m', type=int, help='Number of rows')
    parser.add_argument('n', type=int, help='Number of columns')
    parser.add_argument('output_prefix', help='Output file prefix')
    parser.add_argument('--rel', type=float, default=None,
                        help='Relative error bound (e.g., 1e-3). Bound = rel * (max - min)')

    args = parser.parse_args()

    # Read input fields
    print(f"Reading ground truth: {args.ground_truth}")
    ground_truth = read_raw_float32(args.ground_truth, args.m, args.n)

    print(f"Reading test field: {args.test_field}")
    test_field = read_raw_float32(args.test_field, args.m, args.n)

    # Compare fields
    diff, oob_mask, stats = compare_fields(ground_truth, test_field, args.rel)

    # Write difference field
    diff_path = f"{args.output_prefix}_diff.f32.raw"
    write_raw_float32(diff_path, diff)
    print(f"Wrote difference field: {diff_path}")

    # Write out-of-bound mask if rel bound specified
    if oob_mask is not None:
        oob_path = f"{args.output_prefix}_oob_mask.f32.raw"
        write_raw_float32(oob_path, oob_mask)
        print(f"Wrote out-of-bound mask: {oob_path}")

    # Print statistics
    print("\n=== Error Statistics ===")
    print(f"Ground truth value range: {stats['value_range']:.6e}")
    print(f"Min absolute error:       {stats['min_error']:.6e}")
    print(f"Max absolute error:       {stats['max_error']:.6e}")
    print(f"Mean absolute error:      {stats['mean_error']:.6e}")
    print(f"RMSE:                     {stats['rmse']:.6e}")
    print(f"Max relative error:       {stats['max_rel_error']:.6e}")

    if args.rel is not None:
        print(f"\n=== Error Bound Analysis (REL {args.rel}) ===")
        print(f"Absolute error bound:     {stats['abs_error_bound']:.6e}")
        print(f"Out-of-bound pixels:      {stats['oob_count']} / {args.m * args.n}")
        print(f"Out-of-bound percentage:  {stats['oob_percent']:.4f}%")


if __name__ == '__main__':
    main()
