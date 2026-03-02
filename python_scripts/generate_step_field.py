"""
Generate a 512x512 2D field with step functions.
Upper half (y): 0
Lower half (y): 7 bands along x-axis with values 0, 1, 10, 100, 1000, 10000, 100000
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_step_field(size=512, output_path=None):
    """
    Generate a 2D field with step functions.

    Parameters:
        size: Grid size (default 512x512)
        output_path: Path to save the raw float32 file (optional)

    Returns:
        2D numpy array with the field values
    """
    # Values for each band along x-axis in lower half
    values = [0, 1, 10, 100, 1000, 10000, 100000]
    n_bands = len(values)

    # Initialize field with zeros
    field = np.zeros((size, size), dtype=np.float32)

    # Lower half: y >= size/2 (in array coordinates, row >= size/2)
    # Each band takes equal space along x-axis
    band_width = size // n_bands

    for i, val in enumerate(values):
        x_start = i * band_width
        x_end = (i + 1) * band_width if i < n_bands - 1 else size

        # Fill lower half (row >= size/2) for this x-band
        # In imshow with origin='lower', lower half means rows 0 to size/2-1
        field[0:size//2, x_start:x_end] = val

    # Save to raw file if output path provided
    if output_path:
        field.astype(np.float32).tofile(output_path)
        print(f"Saved field to {output_path}")
        print(f"Shape: {field.shape}, dtype: float32, row-major order")

    return field


def visualize_field(field):
    """Visualize the field using matplotlib."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Linear scale
    im1 = axes[0].imshow(field, origin='lower', cmap='viridis')
    axes[0].set_title('Linear Scale')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0], label='Value')

    # Log scale (add small offset to handle zeros)
    field_log = np.log10(field + 1e-10)
    im2 = axes[1].imshow(field_log, origin='lower', cmap='viridis')
    axes[1].set_title('Log10 Scale')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1], label='log10(Value)')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import os

    # Default output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    output_path = os.path.join(repo_root, "result", "step_field_512x512.f32.raw")

    # Generate and save the field
    field = generate_step_field(size=512, output_path=output_path)

    # Print some statistics
    print(f"\nField statistics:")
    print(f"  Min: {field.min():.6e}")
    print(f"  Max: {field.max():.6e}")
    print(f"  Mean: {field.mean():.6e}")

    # Visualize
    visualize_field(field)
