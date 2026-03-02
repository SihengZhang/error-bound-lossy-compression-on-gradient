"""
Generate a 512x512 2D field with 6 cone functions along the diagonal.
Cones are distributed from bottom-left to top-right with peaks:
1, 10, 100, 1000, 10000, 100000
"""

import numpy as np
import matplotlib.pyplot as plt


def cone_2d(x, y, cx, cy, radius, amplitude):
    """Generate a 2D cone (linear falloff) centered at (cx, cy)."""
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    cone = amplitude * np.maximum(0, 1 - dist / radius)
    return cone


def generate_cone_field(size=512, output_path=None):
    """
    Generate a 2D field with 6 cone functions along the diagonal.

    Parameters:
        size: Grid size (default 512x512)
        output_path: Path to save the raw float32 file (optional)

    Returns:
        2D numpy array with the field values
    """
    # Peak amplitudes for each cone
    peaks = [1, 10, 100, 1000, 10000, 100000]
    n_cones = len(peaks)

    # Create coordinate grids
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)

    # Initialize field with zeros
    field = np.zeros((size, size), dtype=np.float32)

    # Cone radius - chosen so cones don't overlap
    radius = size / 15  # ~34 pixels

    # Padding from corners (as fraction of size)
    padding = 0.1  # 10% padding on each end

    # Distribute cones evenly along diagonal from bottom-left to top-right
    # With padding at both ends
    for i, peak in enumerate(peaks):
        # Parameter t goes from padding to (1-padding)
        t = padding + i / (n_cones - 1) * (1 - 2 * padding)

        # Center coordinates (col increases left to right, row decreases bottom to top)
        cx = t * (size - 1)  # column
        cy = (1 - t) * (size - 1)  # row

        # Add cone to field
        field += cone_2d(X, Y, cx, cy, radius, peak)

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
    output_path = os.path.join(repo_root, "result", "cone_field_512x512.f32.raw")

    # Generate and save the field
    field = generate_cone_field(size=512, output_path=output_path)

    # Print some statistics
    print(f"\nField statistics:")
    print(f"  Min: {field.min():.6e}")
    print(f"  Max: {field.max():.6e}")
    print(f"  Mean: {field.mean():.6e}")

    # Visualize
    visualize_field(field)
