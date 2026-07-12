#!/bin/bash
# ============================================================================
# Unified SFT Finetuning Script for SHROOM-Visions
# ============================================================================
# Run this on your remote A40 server to automate the entire pipeline:
#   1. Virtual environment setup & dependency installation
#   2. Dataset & images download/extraction
#   3. HuggingFace CLI login using token (passed as arg, env var, or interactive)
#   4. Unsloth-powered SFT finetuning & Model Hub upload
# ============================================================================

set -e

echo "=========================================================="
echo "Starting SHROOM SFT Unified Pipeline"
echo "=========================================================="

# ── 1. Virtual Environment Setup ──
echo ""
echo "[1/4] Setting up python virtual environment..."
if [ ! -d "venv_finetune" ]; then
    python3 -m venv venv_finetune
    echo "  Created venv_finetune/"
else
    echo "  venv_finetune/ already exists, reusing."
fi
source venv_finetune/bin/activate
echo "  Activated venv_finetune"

echo "  Installing/upgrading dependencies..."
pip install --upgrade pip
pip install -r requirements_finetune.txt
echo "  Dependencies installed."

# ── 2. Data & Image Retrieval ──
echo ""
echo "[2/4] Retrieving SHROOM-Visions data and images..."

DATA_URL="https://a3s.fi/mickusti-2007780-pub/shroom-visions-data.zip"
IMAGES_URL="https://a3s.fi/mickusti-2007780-pub/shroom-visions-images.tar.gz"

if [ ! -d "shroom-visions-data/distrib" ]; then
    echo "  Downloading data zip..."
    wget -q --show-progress -O shroom-visions-data.zip "$DATA_URL"
    echo "  Extracting data..."
    python3 -c "import zipfile; zipfile.ZipFile('shroom-visions-data.zip').extractall('shroom-visions-data')"
    rm -f shroom-visions-data.zip
else
    echo "  SHROOM data already present."
fi

if [ ! -d "shroom-vis-images" ]; then
    echo "  Downloading images tar.gz..."
    wget -q --show-progress -O shroom-visions-images.tar.gz "$IMAGES_URL"
    echo "  Extracting images (this may take a moment)..."
    python3 -c "import tarfile; tarfile.open('shroom-visions-images.tar.gz').extractall()"
    rm -f shroom-visions-images.tar.gz
else
    echo "  SHROOM images already present."
fi

# ── 3. HuggingFace Authentication ──
echo ""
echo "[3/4] Authenticating with Hugging Face..."
if [ -z "$HF_TOKEN" ]; then
    if [ -n "$1" ]; then
        HF_TOKEN="$1"
    else
        read -sp "Enter your Hugging Face Token: " HF_TOKEN
        echo ""
    fi
fi

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Hugging Face token is required."
    exit 1
fi

huggingface-cli login --token "$HF_TOKEN"

# ── 4. SFT Finetuning Execution ──
echo ""
echo "[4/4] Launching Qwen3.5-4B SFT Finetuning..."
python finetune.py \
  --model_id unsloth/Qwen3.5-4B \
  --data_file shroom-visions-data/distrib/shroom-vision.train.en.labeled.jsonl \
  --images_dir shroom-vis-images \
  --output_dir ./checkpoints/qwen35-4b-shroom-sft \
  --hub_model_id amanuelbyte/Qwen3.5-4B-SHROOM-SFT \
  --hub_token "$HF_TOKEN" \
  --num_epochs 3 \
  --batch_size 2 \
  --grad_accum 4 \
  --lr 2e-4 \
  --lora_rank 16 \
  --max_seq_length 2048 \
  --push_to_hub

echo ""
echo "=========================================================="
echo "  SFT Finetuning & Upload Completed Successfully!"
echo "=========================================================="

