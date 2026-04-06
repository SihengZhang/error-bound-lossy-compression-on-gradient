#!/bin/bash
# Setup script for gradient-compression environment with GPU support

set -e

ENV_NAME="gradient-compression"

echo "=========================================="
echo "Setting up $ENV_NAME environment"
echo "=========================================="

# Check if environment exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Removing existing environment..."
    conda env remove -n $ENV_NAME -y
fi

# Create base environment
echo ""
echo "[1/4] Creating conda environment..."
conda env create -f environment.yml

# Activate environment
echo ""
echo "[2/4] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install PyTorch with CUDA
echo ""
echo "[3/4] Installing PyTorch with CUDA 12.1..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install CuPy
echo ""
echo "[4/4] Installing CuPy for CUDA 12..."
pip install cupy-cuda12x

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "
import torch
import cupy as cp
import numpy as np
import scipy
import osqp
import cvxpy

print(f'NumPy:   {np.__version__}')
print(f'SciPy:   {scipy.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'CuPy:    {cp.__version__}')
print(f'  GPUs: {cp.cuda.runtime.getDeviceCount()}')
print(f'OSQP:    {osqp.__version__}')
print(f'CVXPY:   {cvxpy.__version__}')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo "=========================================="
