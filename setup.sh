#!/bin/bash
# ============================================================================
# SHROOM-Visions Evaluation Setup Script
# ============================================================================
# Run this on your remote A40 server to set up the environment.
# Downloads data, images, installs dependencies, and verifies CUDA.
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
echo "[1/7] Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found. Please install Python 3.9+"
    exit 1
fi
echo "  Python: $($PYTHON --version)"

# ── Check GPU & CUDA driver ──
echo ""
echo "[2/7] Checking GPU & CUDA driver..."
if command -v nvidia-smi &> /dev/null; then
    echo "  NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    # Extract CUDA version from nvidia-smi
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
    echo "  CUDA Driver Version: $CUDA_VERSION"
else
    echo "  WARNING: nvidia-smi not found."
    CUDA_VERSION=""
fi

# ── Download data ──
echo ""
echo "[3/7] Downloading SHROOM-Visions data..."

DATA_URL="https://a3s.fi/mickusti-2007780-pub/shroom-visions-data.zip"
IMAGES_URL="https://a3s.fi/mickusti-2007780-pub/shroom-visions-images.tar.gz"

if [ ! -d "shroom-visions-data/distrib" ]; then
    echo "  Downloading data zip..."
    wget -q --show-progress -O shroom-visions-data.zip "$DATA_URL"
    echo "  Extracting data..."
    rm -rf distrib
    $PYTHON -c "import zipfile; zipfile.ZipFile('shroom-visions-data.zip').extractall('shroom-visions-data')"
    rm -f shroom-visions-data.zip
    echo "  Data extracted to shroom-visions-data/"
else
    echo "  Data already exists, skipping download."
fi

if [ ! -d "shroom-visions-images" ]; then
    echo "  Downloading images tar.gz..."
    wget -q --show-progress -O shroom-visions-images.tar.gz "$IMAGES_URL"
    echo "  Extracting images (this may take a while)..."
    $PYTHON -c "import tarfile; tarfile.open('shroom-visions-images.tar.gz').extractall()"
    rm -f shroom-visions-images.tar.gz
    echo "  Images extracted."
else
    echo "  Images already exist, skipping download."
fi

# ── Create virtual environment ──
echo ""
echo "[4/7] Creating virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    echo "  Created venv/"
else
    echo "  venv/ already exists, reusing."
fi
source venv/bin/activate
echo "  Activated venv"

# ── Install PyTorch with correct CUDA version ──
echo ""
echo "[5/7] Installing PyTorch with correct CUDA version..."
pip install --upgrade pip

# Determine the right PyTorch CUDA build
if [ -n "$CUDA_VERSION" ]; then
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    echo "  Detected CUDA: ${CUDA_MAJOR}.${CUDA_MINOR}"

    if [ "$CUDA_MAJOR" -ge 13 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        echo "  Using CUDA 12.8 PyTorch build (driver supports CUDA $CUDA_VERSION)"
    elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        echo "  Using CUDA 12.8 PyTorch build"
    elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 6 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu126"
        echo "  Using CUDA 12.6 PyTorch build"
    elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        echo "  Using CUDA 12.4 PyTorch build"
    elif [ "$CUDA_MAJOR" -eq 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        echo "  Using CUDA 12.1 PyTorch build"
    elif [ "$CUDA_MAJOR" -eq 11 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        echo "  Using CUDA 11.8 PyTorch build"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        echo "  Defaulting to CUDA 12.1 PyTorch build"
    fi

    pip install --force-reinstall --no-cache-dir torch --index-url "$TORCH_INDEX"
else
    echo "  No CUDA detected, installing CPU PyTorch"
    pip install --force-reinstall --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
fi

# ── Install other dependencies ──
echo ""
echo "[6/7] Installing other dependencies..."
pip install -r requirements.txt
echo "  Dependencies installed."

# ── Verify setup ──
echo ""
echo "[7/7] Verifying setup..."
python -c "
import torch
import transformers
print(f'  PyTorch:      {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory:   {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  WARNING: CUDA is NOT available! Check your NVIDIA driver.')
"

# Check data is present
echo ""
if [ -f "shroom-visions-data/distrib/shroom-vision.train.en.labeled.jsonl" ]; then
    echo "  ✓ English train data found"
else
    echo "  ✗ English train data NOT found!"
fi

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
