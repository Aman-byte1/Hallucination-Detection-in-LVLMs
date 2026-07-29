#!/bin/bash
# ============================================================================
# Multilingual SFT Pipeline for SHROOM-Visions (EN + FR + IT + ZH)
# ============================================================================
# Self-contained script — run on a fresh RunPod A40 instance.
# Trains a SINGLE multilingual model on all 4 languages (~15K samples).
#
# Usage:
#   # One-liner from scratch:
#   HF_TOKEN=hf_xxx bash -c 'git clone https://github.com/Aman-byte1/Hallucination-Detection-in-LVLMs.git && cd Hallucination-Detection-in-LVLMs && bash run_sft_multilingual.sh'
#
#   # Or if already cloned:
#   bash run_sft_multilingual.sh <HF_TOKEN>
#
# Strategy:
#   Concatenates all 4 language train files (en, fr, it, zh) into a single
#   JSONL file and trains one unified model. This works because:
#   - Qwen3.5 is natively multilingual
#   - The data format is identical across languages
#   - More data (15K vs 3.8K) = better generalization
# ============================================================================

set -euo pipefail

# Disable HF xet downloads — they stall on some RunPod instances
export HF_HUB_DISABLE_XET=1
export HF_HOME="/workspace/huggingface_cache"

REPO_URL="https://github.com/Aman-byte1/Hallucination-Detection-in-LVLMs.git"
REPO_DIR="Hallucination-Detection-in-LVLMs"

DATA_URL="https://a3s.fi/mickusti-2007780-pub/shroom-visions-data.zip"
IMAGES_URL="https://a3s.fi/mickusti-2007780-pub/shroom-visions-images.tar.gz"

# Training config (A40 48GB) — adjusted for 4× more data
MODEL_ID="unsloth/Qwen3.5-4B"
BASE_MODEL_ID="Qwen/Qwen3.5-4B"
HUB_MODEL_ID="amanuelbyte/Qwen3.5-4B-SHROOM-SFT-Multilingual"
OUTPUT_DIR="./checkpoints/qwen35-4b-shroom-sft-multilingual"
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=4
LR="2e-4"
LORA_RANK=16
MAX_SEQ_LENGTH=2048
SEED=42

# Languages to include
LANGUAGES=("en" "fr" "it" "zh")
DATA_DIR="shroom-visions-data/distrib"
MERGED_FILE="${DATA_DIR}/shroom-vision.train.all.labeled.jsonl"

echo ""
echo "============================================================"
echo "  SHROOM-Visions MULTILINGUAL SFT Pipeline (RunPod A40)"
echo "  Languages: ${LANGUAGES[*]}"
echo "============================================================"
echo ""

# ============================================================================
# Step 1: Clone the repository (skip if already inside it)
# ============================================================================
echo "[1/8] Checking repository..."

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
echo "[2/8] Setting up Python environment..."

# Store venv OUTSIDE the repo so 'rm -rf' of the repo doesn't nuke it
VENV_DIR="/workspace/venv_shroom_sft"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Created $VENV_DIR/"
else
    echo "  $VENV_DIR/ already exists, reusing (no reinstall needed)."
fi
source "$VENV_DIR/bin/activate"
echo "  Activated venv (Python: $(python --version))"

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
echo "[3/8] Downloading SHROOM-Visions data and images..."

if [ ! -d "$DATA_DIR" ]; then
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

# ============================================================================
# Step 4: Merge all language files into one JSONL
# ============================================================================
echo ""
echo "[4/8] Merging multilingual training data..."

# Remove old merged file if it exists
rm -f "$MERGED_FILE"

TOTAL_LINES=0
for LANG in "${LANGUAGES[@]}"; do
    LANG_FILE="${DATA_DIR}/shroom-vision.train.${LANG}.labeled.jsonl"
    if [ -f "$LANG_FILE" ]; then
        LANG_COUNT=$(wc -l < "$LANG_FILE")
        TOTAL_LINES=$((TOTAL_LINES + LANG_COUNT))
        cat "$LANG_FILE" >> "$MERGED_FILE"
        echo "  ✓ ${LANG}: ${LANG_COUNT} samples"
    else
        echo "  ✗ ${LANG}: FILE NOT FOUND ($LANG_FILE)"
        exit 1
    fi
done

# Shuffle the merged file so languages are interleaved during training
python -c "
import random
random.seed(42)
with open('$MERGED_FILE', 'r', encoding='utf-8') as f:
    lines = f.readlines()
random.shuffle(lines)
with open('$MERGED_FILE', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print(f'  Shuffled {len(lines)} samples')
"

echo "  ────────────────────────────"
echo "  Total: ${TOTAL_LINES} samples → ${MERGED_FILE}"

# ============================================================================
# Step 5: HuggingFace authentication
# ============================================================================
echo ""
echo "[5/8] Authenticating with HuggingFace..."

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
    echo "  Provide via: HF_TOKEN=hf_xxx bash run_sft_multilingual.sh"
    echo "           or: bash run_sft_multilingual.sh hf_xxx"
    exit 1
fi

pip install huggingface_hub --quiet
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
echo "  ✓ Authenticated with HuggingFace."

# ============================================================================
# Step 6: SFT Finetuning on merged multilingual data + Push to Hub
# ============================================================================
echo ""
echo "[6/8] Starting MULTILINGUAL SFT Finetuning..."
echo "  Model:          $MODEL_ID"
echo "  Data:           $MERGED_FILE (${TOTAL_LINES} samples, ${#LANGUAGES[@]} languages)"
echo "  Hub upload:     $HUB_MODEL_ID"
echo "  Epochs:         $NUM_EPOCHS"
echo "  Batch:          ${BATCH_SIZE} × ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM))"
echo "  LR:             $LR"
echo "  LoRA rank:      $LORA_RANK"
echo "  Seed:           $SEED"
echo ""

python finetune.py \
  --model_id "$MODEL_ID" \
  --data_file "$MERGED_FILE" \
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
echo "  ✓ Multilingual SFT Finetuning complete. Model pushed to $HUB_MODEL_ID"

# ============================================================================
# Step 7: Evaluate the finetuned model on each language separately
# ============================================================================
echo ""
echo "[7/8] Evaluating finetuned model on each language..."

pip install tabulate scipy tqdm --quiet

mkdir -p outputs_multilingual_finetuned

for LANG in "${LANGUAGES[@]}"; do
    echo ""
    echo "  ── Evaluating: ${LANG} ──"

    # Set the data file for this language
    LANG_TRAIN_FILE="${DATA_DIR}/shroom-vision.train.${LANG}.labeled.jsonl"

    python evaluate.py \
      --model_id "${OUTPUT_DIR}/merged" \
      --data_file "$LANG_TRAIN_FILE" \
      --no_think

    # Move outputs to language-specific names
    if [ -d "outputs" ]; then
        for f in outputs/*; do
            base=$(basename "$f")
            ext="${base##*.}"
            name="${base%.*}"
            cp "$f" "outputs_multilingual_finetuned/${name}_${LANG}.${ext}"
        done
        rm -rf outputs
    fi

    echo "  ✓ ${LANG} evaluation complete"
done

echo ""
echo "  ✓ All language evaluations saved to outputs_multilingual_finetuned/"

# ============================================================================
# Step 8: Evaluate baseline (unfinetuned) model for comparison
# ============================================================================
echo ""
echo "[8/8] Evaluating BASELINE model on English for comparison..."

python evaluate.py \
  --model_id "$BASE_MODEL_ID" \
  --no_think

if [ -d "outputs" ]; then
    mv outputs outputs_baseline
    echo "  ✓ Baseline results saved to outputs_baseline/"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "  ✓ SHROOM-Visions MULTILINGUAL SFT Pipeline Complete!"
echo "============================================================"
echo ""
echo "  Languages trained on: ${LANGUAGES[*]}"
echo "  Total training samples: ${TOTAL_LINES}"
echo ""
echo "  Finetuned model:"
echo "    Local:       ${OUTPUT_DIR}/merged"
echo "    HuggingFace: https://huggingface.co/${HUB_MODEL_ID}"
echo ""
echo "  Evaluation results:"
echo "    Finetuned (per-language): outputs_multilingual_finetuned/"
echo "    Baseline (English):      outputs_baseline/"
echo ""
echo "  Quick compare (Python):"
echo "    import json"
echo "    for lang in ['en', 'fr', 'it', 'zh']:"
echo "        m = json.load(open(f'outputs_multilingual_finetuned/metrics_en_{lang}.json'))"
echo "        print(f'{lang}: IoU={m[\"metrics\"][\"overall\"][\"iou_mean\"]:.4f}')"
echo ""
echo "============================================================"
