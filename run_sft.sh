#!/bin/bash
# ============================================================================
# Unified SFT Pipeline for SHROOM-Visions on RunPod
# ============================================================================
# Self-contained script — run on a fresh RunPod A40 instance.
# Handles everything: clone → setup → download → train → push → evaluate.
#
# Usage (pick one):
#
#   # Option A: Clone first, then run
#   git clone https://github.com/Aman-byte1/Hallucination-Detection-in-LVLMs.git
#   cd Hallucination-Detection-in-LVLMs
#   bash run_sft.sh <HF_TOKEN>
#
#   # Option B: One-liner (pass token as env var)
#   HF_TOKEN=hf_xxx bash -c 'git clone https://github.com/Aman-byte1/Hallucination-Detection-in-LVLMs.git && cd Hallucination-Detection-in-LVLMs && bash run_sft.sh'
#
# The HuggingFace token can be provided via:
#   1. First CLI argument:   bash run_sft.sh hf_xxx
#   2. Environment variable: HF_TOKEN=hf_xxx bash run_sft.sh
#   3. Interactive prompt:   bash run_sft.sh  (will ask)
#
# Data split note:
#   Both finetune.py and evaluate.py use seed=42 with eval_ratio=0.10 to
#   deterministically split the same 10% of training data as the test set.
#   This guarantees the test set is NEVER in the training data.
# ============================================================================

set -euo pipefail

REPO_URL="https://github.com/Aman-byte1/Hallucination-Detection-in-LVLMs.git"
REPO_DIR="Hallucination-Detection-in-LVLMs"

DATA_URL="https://a3s.fi/mickusti-2007780-pub/shroom-visions-data.zip"
IMAGES_URL="https://a3s.fi/mickusti-2007780-pub/shroom-visions-images.tar.gz"

# Training config (A40 48GB)
MODEL_ID="unsloth/Qwen3.5-4B"
BASE_MODEL_ID="Qwen/Qwen3.5-4B"
HUB_MODEL_ID="amanuelbyte/Qwen3.5-4B-SHROOM-SFT"
OUTPUT_DIR="./checkpoints/qwen35-4b-shroom-sft"
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=4
LR="2e-4"
LORA_RANK=16
MAX_SEQ_LENGTH=2048
SEED=42

echo ""
echo "============================================================"
echo "  SHROOM-Visions SFT Pipeline (RunPod A40)"
echo "============================================================"
echo ""

# ============================================================================
# Step 1: Clone the repository (skip if already inside it)
# ============================================================================
echo "[1/7] Checking repository..."

if [ -f "finetune.py" ] && [ -f "evaluate.py" ]; then
    echo "  Already inside the repo directory. Skipping clone."
elif [ -d "$REPO_DIR" ]; then
    echo "  Repo directory exists. Entering $REPO_DIR/"
    cd "$REPO_DIR"
else
    echo "  Cloning $REPO_URL ..."
    git clone "$REPO_URL"
    cd "$REPO_DIR"
    echo "  Cloned and entered $REPO_DIR/"
fi

echo "  Working directory: $(pwd)"

# ============================================================================
# Step 2: Python venv + CUDA-aware PyTorch + dependencies
# ============================================================================
echo ""
echo "[2/7] Setting up Python environment..."

# Create venv
if [ ! -d "venv_finetune" ]; then
    python3 -m venv venv_finetune
    echo "  Created venv_finetune/"
else
    echo "  venv_finetune/ already exists, reusing."
fi
source venv_finetune/bin/activate
echo "  Activated venv_finetune (Python: $(python --version))"

pip install --upgrade pip --quiet

# ── Install PyTorch with correct CUDA version ──
echo "  Detecting CUDA version..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
    echo "  CUDA Driver: $CUDA_VERSION"

    if [ "$CUDA_MAJOR" -ge 13 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 6 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu126"
    elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    elif [ "$CUDA_MAJOR" -eq 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    elif [ "$CUDA_MAJOR" -eq 11 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    fi
    echo "  Using PyTorch index: $TORCH_INDEX"
else
    echo "  WARNING: nvidia-smi not found — installing CPU PyTorch"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

# Check if working PyTorch with CUDA is already installed
CUDA_OK=$(python -c "
try:
    import torch
    print(torch.cuda.is_available())
except ImportError:
    print('False')
" 2>/dev/null || echo "False")

if [ "$CUDA_OK" = "True" ]; then
    echo "  ✓ PyTorch with CUDA already installed. Skipping reinstall."
else
    echo "  Installing PyTorch + torchvision..."
    pip install --force-reinstall --no-cache-dir torch torchvision --index-url "$TORCH_INDEX"
fi

# ── Install remaining dependencies ──
echo "  Installing finetuning dependencies..."
pip install -r requirements_finetune.txt --quiet

# Verify GPU
python -c "
import torch
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ============================================================================
# Step 3: Download SHROOM-Visions data and images
# ============================================================================
echo ""
echo "[3/7] Downloading SHROOM-Visions data and images..."

if [ ! -d "shroom-visions-data/distrib" ]; then
    echo "  Downloading data zip..."
    wget -q --show-progress -O shroom-visions-data.zip "$DATA_URL"
    echo "  Extracting data..."
    python -c "import zipfile; zipfile.ZipFile('shroom-visions-data.zip').extractall('shroom-visions-data')"
    rm -f shroom-visions-data.zip
    echo "  ✓ Data extracted to shroom-visions-data/"
else
    echo "  ✓ SHROOM data already present."
fi

if [ ! -d "shroom-vis-images" ]; then
    echo "  Downloading images tar.gz (this may take a while)..."
    wget -q --show-progress -O shroom-visions-images.tar.gz "$IMAGES_URL"
    echo "  Extracting images..."
    python -c "import tarfile; tarfile.open('shroom-visions-images.tar.gz').extractall()"
    rm -f shroom-visions-images.tar.gz
    echo "  ✓ Images extracted."
else
    echo "  ✓ SHROOM images already present."
fi

# Quick sanity check
TRAIN_COUNT=$(wc -l < shroom-visions-data/distrib/shroom-vision.train.en.labeled.jsonl)
echo "  Training samples: $TRAIN_COUNT"

# ============================================================================
# Step 4: HuggingFace authentication
# ============================================================================
echo ""
echo "[4/7] Authenticating with HuggingFace..."

if [ -z "${HF_TOKEN:-}" ]; then
    if [ -n "${1:-}" ]; then
        HF_TOKEN="$1"
    else
        read -sp "  Enter your HuggingFace Token: " HF_TOKEN
        echo ""
    fi
fi

if [ -z "$HF_TOKEN" ]; then
    echo "  ERROR: HuggingFace token is required for pushing the model."
    echo "  Provide via: HF_TOKEN=hf_xxx bash run_sft.sh"
    echo "           or: bash run_sft.sh hf_xxx"
    exit 1
fi

pip install huggingface_hub --quiet
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
echo "  ✓ Authenticated with HuggingFace."


# ============================================================================
# Step 5: SFT Finetuning + Push to Hub
# ============================================================================
echo ""
echo "[5/7] Starting SFT Finetuning..."
echo "  Model:          $MODEL_ID"
echo "  Hub upload:     $HUB_MODEL_ID"
echo "  Epochs:         $NUM_EPOCHS"
echo "  Batch:          ${BATCH_SIZE} × ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM))"
echo "  LR:             $LR"
echo "  LoRA rank:      $LORA_RANK"
echo "  Seed:           $SEED"
echo ""

python finetune.py \
  --model_id "$MODEL_ID" \
  --data_file shroom-visions-data/distrib/shroom-vision.train.en.labeled.jsonl \
  --images_dir shroom-vis-images \
  --output_dir "$OUTPUT_DIR" \
  --hub_model_id "$HUB_MODEL_ID" \
  --hub_token "$HF_TOKEN" \
  --num_epochs "$NUM_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --lora_rank "$LORA_RANK" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --seed "$SEED" \
  --push_to_hub

echo ""
echo "  ✓ SFT Finetuning complete. Model pushed to $HUB_MODEL_ID"

# ============================================================================
# Step 6: Evaluate the finetuned model on held-out 10% test split
# ============================================================================
echo ""
echo "[6/7] Evaluating FINETUNED model on held-out test split..."
echo "  Model:  ${OUTPUT_DIR}/merged"
echo "  Seed:   $SEED (same split as training — test set was excluded)"
echo ""

# Install evaluation dependencies (tabulate, scipy, tqdm already in requirements)
pip install tabulate scipy tqdm --quiet

python evaluate.py \
  --model_id "${OUTPUT_DIR}/merged" \
  --no_think

echo ""
echo "  ✓ Finetuned model evaluation complete."
echo "  Results saved to: outputs/"

# ============================================================================
# Step 7: Evaluate the baseline (unfinetuned) model for comparison
# ============================================================================
echo ""
echo "[7/7] Evaluating BASELINE model for comparison..."
echo "  Model:  $BASE_MODEL_ID"
echo ""

# Rename finetuned outputs so they don't get overwritten
if [ -d "outputs" ]; then
    mv outputs outputs_finetuned
    echo "  Moved finetuned results to outputs_finetuned/"
fi

python evaluate.py \
  --model_id "$BASE_MODEL_ID" \
  --no_think

# Rename baseline outputs
if [ -d "outputs" ]; then
    mv outputs outputs_baseline
    echo "  Moved baseline results to outputs_baseline/"
fi

echo ""
echo "  ✓ Baseline model evaluation complete."

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "  ✓ SHROOM-Visions SFT Pipeline Complete!"
echo "============================================================"
echo ""
echo "  Finetuned model:"
echo "    Local:      ${OUTPUT_DIR}/merged"
echo "    HuggingFace: https://huggingface.co/${HUB_MODEL_ID}"
echo ""
echo "  Evaluation results:"
echo "    Finetuned:  outputs_finetuned/metrics_en.json"
echo "    Baseline:   outputs_baseline/metrics_en.json"
echo ""
echo "  Compare with:"
echo "    python -c \""
echo "      import json"
echo "      ft = json.load(open('outputs_finetuned/metrics_en.json'))"
echo "      bl = json.load(open('outputs_baseline/metrics_en.json'))"
echo "      print(f'Finetuned IoU: {ft[\\\"metrics\\\"][\\\"overall\\\"][\\\"iou_mean\\\"]:.4f}')"
echo "      print(f'Baseline  IoU: {bl[\\\"metrics\\\"][\\\"overall\\\"][\\\"iou_mean\\\"]:.4f}')"
echo "    \""
echo ""
echo "============================================================"
