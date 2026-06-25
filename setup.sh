#!/bin/bash
# ============================================================================
# SHROOM-Visions Evaluation Setup Script
# ============================================================================
# Run this on your remote A40 server to set up the environment.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ============================================================================

set -e

echo "============================================"
echo "  SHROOM-Visions Evaluation Setup"
echo "============================================"

# ── Check Python ──
echo ""
echo "[1/5] Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found. Please install Python 3.9+"
    exit 1
fi
echo "  Python: $($PYTHON --version)"

# ── Check CUDA ──
echo ""
echo "[2/5] Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "  NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "  WARNING: nvidia-smi not found. GPU may not be available."
fi

if $PYTHON -c "import torch; print(f'  PyTorch CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
    :
else
    echo "  PyTorch not yet installed, will install next."
fi

# ── Create virtual environment ──
echo ""
echo "[3/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    echo "  Created venv/"
else
    echo "  venv/ already exists, reusing."
fi
source venv/bin/activate
echo "  Activated venv"

# ── Install dependencies ──
echo ""
echo "[4/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "  Dependencies installed."

# ── Verify setup ──
echo ""
echo "[5/5] Verifying setup..."
python -c "
import torch
import transformers
print(f'  PyTorch:      {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory:   {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "To run the evaluation:"
echo "  source venv/bin/activate"
echo ""
echo "  # Quick test (10 samples):"
echo "  python evaluate.py --max_samples 10"
echo ""
echo "  # Full evaluation:"
echo "  python evaluate.py"
echo ""
echo "  # Resume if interrupted:"
echo "  python evaluate.py --resume"
echo "============================================"
