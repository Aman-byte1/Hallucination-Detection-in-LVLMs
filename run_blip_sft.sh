#!/bin/bash
# ============================================================================
# BLIP-VQA-Base SFT Pipeline for SHROOM-Visions on RunPod
# ============================================================================
# Self-contained script — run on a fresh RunPod A40 instance.
# Handles: clone → setup → download → train → push → evaluate.
#
# Usage:
#   git clone https://github.com/Aman-byte1/Hallucination-Detection-in-LVLMs.git
#   cd Hallucination-Detection-in-LVLMs
#   bash run_blip_sft.sh <HF_TOKEN>
# ============================================================================

set -euo pipefail

export HF_HUB_DISABLE_XET=1
export HF_HOME="/workspace/huggingface_cache"

REPO_URL="https://github.com/Aman-byte1/Hallucination-Detection-in-LVLMs.git"
REPO_DIR="Hallucination-Detection-in-LVLMs"

DATA_URL="https://a3s.fi/mickusti-2007780-pub/shroom-visions-data.zip"
IMAGES_URL="https://a3s.fi/mickusti-2007780-pub/shroom-visions-images.tar.gz"

# Training config (A40 48GB — BLIP is tiny, uses minimal VRAM)
MODEL_ID="Salesforce/blip-vqa-base"
HUB_MODEL_ID="amanuelbyte/BLIP-VQA-Base-SHROOM-SFT"
OUTPUT_DIR="./checkpoints/blip-vqa-shroom-sft"
NUM_EPOCHS=5
BATCH_SIZE=8
GRAD_ACCUM=2
LR="5e-5"
MAX_QUESTION_LENGTH=512
MAX_ANSWER_LENGTH=128
SEED=42

echo ""
echo "============================================================"
echo "  SHROOM-Visions SFT Pipeline — BLIP-VQA-Base (RunPod A40)"
echo "============================================================"
echo ""

# ============================================================================
# Step 1: Clone the repository
# ============================================================================
echo "[1/7] Checking repository..."

if [ -f "finetune_blip.py" ] && [ -f "evaluate_blip.py" ]; then
    echo "  Already inside the repo directory."
elif [ -d "$REPO_DIR" ]; then
    echo "  Repo directory exists. Entering $REPO_DIR/"
    cd "$REPO_DIR"
else
    echo "  Cloning $REPO_URL ..."
    git clone "$REPO_URL"
    cd "$REPO_DIR"
fi
echo "  Working directory: $(pwd)"

# ============================================================================
# Step 2: Python venv + dependencies
# ============================================================================
echo ""
echo "[2/7] Setting up Python environment..."

if [ ! -d "venv_vqa" ]; then
    python3 -m venv --system-site-packages venv_vqa
    echo "  Created venv_vqa with --system-site-packages"
else
    echo "  venv_vqa/ already exists, reusing."
fi
source venv_vqa/bin/activate
echo "  Activated venv_vqa (Python: $(python --version))"

# Install uv for ultra-fast dependency resolution
pip install uv --quiet 2>/dev/null || true

echo "  Installing VQA dependencies fast with uv..."
if command -v uv &> /dev/null; then
    uv pip install -r requirements_vqa.txt --quiet
else
    pip install -r requirements_vqa.txt --quiet
fi

python -c "
import torch
print(f'  PyTorch:   {torch.__version__}')
print(f'  CUDA:      {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:       {torch.cuda.get_device_name(0)}')
    print(f'  GPU Mem:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ============================================================================
# Step 3: Download data and images
# ============================================================================
echo ""
echo "[3/7] Downloading SHROOM-Visions data and images..."

if [ ! -d "shroom-visions-data/distrib" ]; then
    wget -q --show-progress -O shroom-visions-data.zip "$DATA_URL"
    python -c "import zipfile; zipfile.ZipFile('shroom-visions-data.zip').extractall('shroom-visions-data')"
    rm -f shroom-visions-data.zip
    echo "  ✓ Data extracted."
else
    echo "  ✓ Data already present."
fi

if [ ! -d "shroom-vis-images" ]; then
    wget -q --show-progress -O shroom-visions-images.tar.gz "$IMAGES_URL"
    python -c "import tarfile; tarfile.open('shroom-visions-images.tar.gz').extractall()"
    rm -f shroom-visions-images.tar.gz
    echo "  ✓ Images extracted."
else
    echo "  ✓ Images already present."
fi

# ============================================================================
# Step 4: HuggingFace auth
# ============================================================================
echo ""
echo "[4/7] Authenticating with HuggingFace..."

if [ -z "${HF_TOKEN:-}" ]; then
    if [ -n "${1:-}" ]; then
        HF_TOKEN="$1"
    else
        read -sp "  Enter HuggingFace Token: " HF_TOKEN
        echo ""
    fi
fi

if [ -z "$HF_TOKEN" ]; then
    echo "  ERROR: HF token required. Usage: bash run_blip_sft.sh hf_xxx"
    exit 1
fi

pip install huggingface_hub --quiet
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
echo "  ✓ Authenticated."

# ============================================================================
# Step 5: SFT Finetuning
# ============================================================================
echo ""
echo "[5/7] Starting BLIP-VQA-Base SFT..."
echo "  Model:      $MODEL_ID"
echo "  Hub:        $HUB_MODEL_ID"
echo "  Epochs:     $NUM_EPOCHS"
echo "  Batch:      ${BATCH_SIZE} × ${GRAD_ACCUM}"
echo "  LR:         $LR"
echo ""

python finetune_blip.py \
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
  --max_question_length "$MAX_QUESTION_LENGTH" \
  --max_answer_length "$MAX_ANSWER_LENGTH" \
  --seed "$SEED" \
  --push_to_hub

echo "  ✓ SFT complete."

# ============================================================================
# Step 6: Evaluate finetuned model
# ============================================================================
echo ""
echo "[6/7] Evaluating FINETUNED BLIP-VQA-Base..."

python evaluate_blip.py \
  --model_id "${OUTPUT_DIR}/final"

echo "  ✓ Finetuned evaluation complete."

# ============================================================================
# Step 7: Evaluate baseline (unfinetuned) model
# ============================================================================
echo ""
echo "[7/7] Evaluating BASELINE BLIP-VQA-Base..."

if [ -d "outputs_blip" ]; then
    mv outputs_blip outputs_blip_finetuned
fi

python evaluate_blip.py \
  --model_id "$MODEL_ID"

if [ -d "outputs_blip" ]; then
    mv outputs_blip outputs_blip_baseline
fi

echo ""
echo "============================================================"
echo "  ✓ BLIP-VQA-Base Pipeline Complete!"
echo "============================================================"
echo ""
echo "  Finetuned: ${OUTPUT_DIR}/final"
echo "  Hub:       https://huggingface.co/${HUB_MODEL_ID}"
echo ""
echo "  Results:"
echo "    Finetuned: outputs_blip_finetuned/metrics_en.json"
echo "    Baseline:  outputs_blip_baseline/metrics_en.json"
echo ""
echo "  Compare all three models:"
echo "    python -c \""
echo "      import json"
echo "      for name, path in [('Qwen SFT', 'outputs_finetuned/metrics_en.json'),"
echo "                          ('MiniCPM SFT', 'outputs_minicpm_finetuned/metrics_en.json'),"
echo "                          ('BLIP SFT', 'outputs_blip_finetuned/metrics_en.json')]:"
echo "        try:"
echo "          m = json.load(open(path))"
echo "          print(f'{name:15} IoU: {m[\\\"metrics\\\"][\\\"overall\\\"][\\\"iou_mean\\\"]:.4f}')"
echo "        except: print(f'{name:15} not found')"
echo "    \""
echo "============================================================"
