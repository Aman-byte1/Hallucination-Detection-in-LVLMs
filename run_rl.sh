#!/bin/bash
# ============================================================================
# SHROOM-Visions GRPO RL Pipeline for RunPod A40
# ============================================================================
# Self-contained script — run on a fresh RunPod A40 instance.
# Handles everything: clone → setup → RL train → push → evaluate.
#
# Budget: ~$2 out of $3 (4 hours @ $0.50/hr on A40)
#   - Setup:     ~15 min
#   - RL train:  ~3 hours   (500 steps GRPO, 4 generations each)
#   - Eval RL:   ~20 min
#   - Eval SFT:  ~20 min
#
# Usage:
#   # Option A: Clone first, then run
#   git clone https://github.com/Aman-byte1/Hallucination-Detection-in-LVLMs.git
#   cd Hallucination-Detection-in-LVLMs
#   bash run_rl.sh <HF_TOKEN>
#
#   # Option B: One-liner
#   HF_TOKEN=hf_xxx bash -c 'git clone https://github.com/Aman-byte1/Hallucination-Detection-in-LVLMs.git && cd Hallucination-Detection-in-LVLMs && bash run_rl.sh'
# ============================================================================

set -euo pipefail

# Disable HF xet downloads — they stall on some RunPod instances
export HF_HUB_DISABLE_XET=1
export HF_HOME="/workspace/huggingface_cache"

REPO_URL="https://github.com/Aman-byte1/Hallucination-Detection-in-LVLMs.git"
REPO_DIR="Hallucination-Detection-in-LVLMs"

DATA_URL="https://a3s.fi/mickusti-2007780-pub/shroom-visions-data.zip"
IMAGES_URL="https://a3s.fi/mickusti-2007780-pub/shroom-visions-images.tar.gz"

# GRPO Config
SFT_MODEL_ID="amanuelbyte/Qwen3.5-4B-SHROOM-SFT"
HUB_MODEL_ID="amanuelbyte/Qwen3.5-4B-SHROOM-GRPO"
OUTPUT_DIR="./checkpoints/qwen35-4b-shroom-grpo"
MAX_STEPS=500
NUM_GENERATIONS=4
MAX_COMPLETION_LENGTH=256
LR="5e-6"
LORA_RANK=16
GRAD_ACCUM=4

echo ""
echo "============================================================"
echo "  SHROOM-Visions GRPO RL Pipeline (RunPod A40)"
echo "============================================================"
echo ""

# ============================================================================
# Step 1: Clone the repository (skip if already inside it)
# ============================================================================
echo "[1/9] Checking repository..."

if [ -f "rl_grpo.py" ] && [ -f "evaluate.py" ]; then
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
echo "[2/9] Setting up Python environment..."

if [ ! -d "venv_rl" ]; then
    python3 -m venv venv_rl
    echo "  Created venv_rl/"
else
    echo "  venv_rl/ already exists, reusing."
fi
source venv_rl/bin/activate
echo "  Activated venv_rl (Python: $(python --version))"

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

# Check if dependencies are already installed
DEPS_OK=$(python -c "
try:
    import unsloth
    import trl
    print('True')
except ImportError:
    print('False')
" 2>/dev/null || echo "False")

if [ "$DEPS_OK" = "True" ]; then
    echo "  ✓ Dependencies already installed. Skipping pip install."
else
    if [ "$CUDA_OK" = "True" ]; then
        echo "  ✓ PyTorch with CUDA already installed. Skipping reinstall."
    else
        echo "  Installing PyTorch + torchvision..."
        pip install --force-reinstall --no-cache-dir torch torchvision --index-url "$TORCH_INDEX"
    fi

    # ── Install RL + evaluation dependencies ──
    echo "  Installing Unsloth + RL dependencies..."
    pip install --upgrade unsloth unsloth_zoo
    pip install "trl>=0.17.0" "peft>=0.15.0" datasets accelerate bitsandbytes
    pip install scipy numpy tqdm tabulate pillow qwen-vl-utils
    pip install huggingface_hub
fi

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
# Step 3: Download SHROOM data + images
# ============================================================================
echo ""
echo "[3/9] Downloading SHROOM-Visions data and images..."

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

TRAIN_COUNT=$(wc -l < shroom-visions-data/distrib/shroom-vision.train.en.labeled.jsonl)
echo "  Training samples: $TRAIN_COUNT"

# ============================================================================
# Step 4: HuggingFace authentication
# ============================================================================
echo ""
echo "[4/9] Authenticating with HuggingFace..."

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
    echo "  Provide via: HF_TOKEN=hf_xxx bash run_rl.sh"
    echo "           or: bash run_rl.sh hf_xxx"
    exit 1
fi

python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
echo "  ✓ Authenticated with HuggingFace."

# ============================================================================
# Step 5: GRPO RL Training
# ============================================================================
echo ""
echo "[5/9] Starting GRPO RL Training..."
echo "  SFT Model:      $SFT_MODEL_ID"
echo "  Hub upload:      $HUB_MODEL_ID"
echo "  Max steps:       $MAX_STEPS"
echo "  Generations:     $NUM_GENERATIONS"
echo "  Completion len:  $MAX_COMPLETION_LENGTH"
echo "  Grad accum:      $GRAD_ACCUM"
echo "  LR:              $LR"
echo "  LoRA rank:       $LORA_RANK"
echo ""

python rl_grpo.py \
  --model_id "$SFT_MODEL_ID" \
  --output_dir "$OUTPUT_DIR" \
  --hub_model_id "$HUB_MODEL_ID" \
  --hub_token "$HF_TOKEN" \
  --max_steps "$MAX_STEPS" \
  --num_generations "$NUM_GENERATIONS" \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --grad_accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --lora_rank "$LORA_RANK" \
  --push_to_hub

echo ""
echo "  ✓ GRPO training complete. Model pushed to $HUB_MODEL_ID"

# ============================================================================
# Step 6: Evaluate the GRPO model on held-out 10% test split
# ============================================================================
echo ""
echo "[6/9] Evaluating GRPO model on held-out test split..."
echo "  Model:  ${OUTPUT_DIR}/merged"
echo ""

# Install eval deps (some may already be installed)
pip install tabulate scipy tqdm --quiet

python evaluate.py \
  --model_id "${OUTPUT_DIR}/merged" \
  --no_think

echo ""
echo "  ✓ GRPO model evaluation complete."

# Rename outputs so they don't get overwritten
if [ -d "outputs" ]; then
    mv outputs outputs_grpo
    echo "  Moved GRPO results to outputs_grpo/"
fi

# ============================================================================
# Step 7: Evaluate the SFT baseline for comparison
# ============================================================================
echo ""
echo "[7/9] Evaluating SFT baseline for comparison..."
echo "  Model:  $SFT_MODEL_ID"
echo ""

python evaluate.py \
  --model_id "$SFT_MODEL_ID" \
  --no_think

# Rename baseline outputs
if [ -d "outputs" ]; then
    mv outputs outputs_sft
    echo "  Moved SFT results to outputs_sft/"
fi

echo ""
echo "  ✓ SFT baseline evaluation complete."

# ============================================================================
# Step 8: Compare results
# ============================================================================
echo ""
echo "[8/9] Comparing GRPO vs SFT results..."

python -c "
import json

try:
    grpo = json.load(open('outputs_grpo/metrics_en.json'))
    sft  = json.load(open('outputs_sft/metrics_en.json'))

    print()
    print('=' * 65)
    print('  GRPO (RL) vs SFT Comparison')
    print('=' * 65)
    print()
    print(f'{\"Metric\":<30} {\"SFT\":>12} {\"GRPO (RL)\":>12} {\"Delta\":>10}')
    print('-' * 65)

    rows = [
        ('Overall IoU',         'overall',              'iou_mean'),
        ('Overall Calibration', 'overall',              'calibration_mean'),
        ('Hallucinated IoU',    'hallucinated_samples', 'iou_mean'),
        ('Clean IoU',           'clean_samples',        'iou_mean'),
    ]
    for name, cat, key in rows:
        s = sft['metrics'][cat][key]
        g = grpo['metrics'][cat][key]
        d = g - s
        arrow = '↑' if d > 0 else '↓' if d < 0 else '='
        print(f'{name:<30} {s:>12.4f} {g:>12.4f} {d:>+9.4f} {arrow}')

    s_acc = sft['metrics']['detection_stats']['detection_accuracy']
    g_acc = grpo['metrics']['detection_stats']['detection_accuracy']
    d_acc = g_acc - s_acc
    arrow = '↑' if d_acc > 0 else '↓' if d_acc < 0 else '='
    print(f'{\"Detection Accuracy\":<30} {s_acc:>12.4f} {g_acc:>12.4f} {d_acc:>+9.4f} {arrow}')

    print()
    print('=' * 65)

except FileNotFoundError as e:
    print(f'Could not find results file: {e}')
except Exception as e:
    print(f'Error comparing results: {e}')
"

# ============================================================================
# Step 9: Upload model card with evaluation results to HuggingFace
# ============================================================================
echo ""
echo "[9/9] Uploading model card with eval results to HuggingFace..."

export HUB_MODEL_ID="$HUB_MODEL_ID"
export HF_TOKEN="$HF_TOKEN"

python << 'MODELCARD_SCRIPT'
import json
import os
from datetime import datetime
from huggingface_hub import HfApi

hub_model_id = os.environ.get("HUB_MODEL_ID", "amanuelbyte/Qwen3.5-4B-SHROOM-GRPO")
hf_token = os.environ.get("HF_TOKEN", "")

# ── Load metrics ──
try:
    grpo = json.load(open("outputs_grpo/metrics_en.json"))
except FileNotFoundError:
    grpo = None
    print("  WARNING: outputs_grpo/metrics_en.json not found")

try:
    sft = json.load(open("outputs_sft/metrics_en.json"))
except FileNotFoundError:
    sft = None
    print("  WARNING: outputs_sft/metrics_en.json not found")

# ── Helper to safely get nested metric ──
def get_metric(data, *keys, default="N/A"):
    if data is None:
        return default
    obj = data
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
        else:
            return default
    return obj

# ── Build the model card ──
def fmt(val, decimals=4):
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return str(val)

now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

readme = f"""---
license: apache-2.0
tags:
  - hallucination-detection
  - vision-language-model
  - grpo
  - reinforcement-learning
  - unsloth
  - qwen3.5
  - shroom
base_model: amanuelbyte/Qwen3.5-4B-SHROOM-SFT
datasets:
  - shroom-visions
language:
  - en
pipeline_tag: image-text-to-text
model-index:
  - name: Qwen3.5-4B-SHROOM-GRPO
    results:
      - task:
          type: hallucination-detection
          name: Hallucination Detection in VLM Outputs
        dataset:
          name: SHROOM-Visions (English, 10% held-out)
          type: shroom-visions
        metrics:
          - name: Overall IoU
            type: iou
            value: {get_metric(grpo, 'metrics', 'overall', 'iou_mean', default=0)}
          - name: Detection Accuracy
            type: accuracy
            value: {get_metric(grpo, 'metrics', 'detection_stats', 'detection_accuracy', default=0)}
          - name: Overall Calibration
            type: calibration
            value: {get_metric(grpo, 'metrics', 'overall', 'calibration_mean', default=0)}
---

# 🧠 Qwen3.5-4B-SHROOM-GRPO

**Hallucination detector for Vision-Language Model outputs**, trained with GRPO (Group Relative Policy Optimization) reinforcement learning on top of the SFT-finetuned [Qwen3.5-4B-SHROOM-SFT](https://huggingface.co/amanuelbyte/Qwen3.5-4B-SHROOM-SFT).

## 🔬 Model Description

This model detects factual errors (hallucinations) in text descriptions of images generated by Vision-Language Models. Given an image and a text response, it outputs a JSON array of detected hallucinated spans with their types and confidence scores.

**Training pipeline:**
1. **Base model:** `Qwen/Qwen3.5-4B`
2. **SFT (Supervised Fine-Tuning):** Trained on SHROOM-Visions labeled data → `amanuelbyte/Qwen3.5-4B-SHROOM-SFT`
3. **GRPO (Reinforcement Learning):** Further trained with 4 custom reward functions → **this model**

### Reward Functions

| Reward Function | Max Score | Description |
|---|---|---|
| `format_reward` | +2.0 | Valid JSON array output |
| `detection_reward` | +3.0 | Correctly identifies hallucination presence |
| `span_iou_reward` | +4.0 | Character-level IoU between predicted and gold spans |
| `label_accuracy_reward` | +1.0 | Uses correct category labels |

### Training Configuration

| Parameter | Value |
|---|---|
| RL Algorithm | Dr-GRPO (via Unsloth) |
| Max steps | 500 |
| Generations per prompt | 4 |
| Max completion length | 256 tokens |
| Learning rate | 5e-6 |
| LoRA rank | 16 |
| Gradient accumulation | 4 |
| Precision | bf16 |
| GPU | NVIDIA A40 (48GB) |

## 📊 Evaluation Results

Evaluated on 10% held-out split of SHROOM-Visions English data (seed=42).
"""

# ── Add comparison table ──
if grpo is not None:
    gm = grpo["metrics"]
    readme += f"""
### GRPO Model Results

| Metric | Value |
|---|---|
| Overall IoU (mean ± std) | {fmt(gm['overall']['iou_mean'])} ± {fmt(gm['overall']['iou_std'])} |
| Overall IoU (median) | {fmt(gm['overall']['iou_median'])} |
| Overall Calibration (mean ± std) | {fmt(gm['overall']['calibration_mean'])} ± {fmt(gm['overall']['calibration_std'])} |
| Overall Calibration (median) | {fmt(gm['overall']['calibration_median'])} |
| Hallucinated IoU (mean) | {fmt(gm['hallucinated_samples']['iou_mean'])} |
| Clean IoU (mean) | {fmt(gm['clean_samples']['iou_mean'])} |
| Detection Accuracy | {fmt(gm['detection_stats']['detection_accuracy'])} |
"""
    det = gm["detection_stats"]
    readme += f"""| Clean Correct | {det['clean_correct']}/{det['total_clean']} ({fmt(det['clean_correct']/max(det['total_clean'],1)*100, 1)}%) |
| Halluc Correct | {det['halluc_correct']}/{det['total_halluc']} ({fmt(det['halluc_correct']/max(det['total_halluc'],1)*100, 1)}%) |
"""

if grpo is not None and sft is not None:
    gm = grpo["metrics"]
    sm = sft["metrics"]
    readme += f"""
### Comparison: GRPO vs SFT Baseline

| Metric | SFT | GRPO (RL) | Delta |
|---|---|---|---|
| Overall IoU | {fmt(sm['overall']['iou_mean'])} | {fmt(gm['overall']['iou_mean'])} | {fmt(gm['overall']['iou_mean'] - sm['overall']['iou_mean'], 4)} |
| Overall Calibration | {fmt(sm['overall']['calibration_mean'])} | {fmt(gm['overall']['calibration_mean'])} | {fmt(gm['overall']['calibration_mean'] - sm['overall']['calibration_mean'], 4)} |
| Hallucinated IoU | {fmt(sm['hallucinated_samples']['iou_mean'])} | {fmt(gm['hallucinated_samples']['iou_mean'])} | {fmt(gm['hallucinated_samples']['iou_mean'] - sm['hallucinated_samples']['iou_mean'], 4)} |
| Clean IoU | {fmt(sm['clean_samples']['iou_mean'])} | {fmt(gm['clean_samples']['iou_mean'])} | {fmt(gm['clean_samples']['iou_mean'] - sm['clean_samples']['iou_mean'], 4)} |
| Detection Accuracy | {fmt(sm['detection_stats']['detection_accuracy'])} | {fmt(gm['detection_stats']['detection_accuracy'])} | {fmt(gm['detection_stats']['detection_accuracy'] - sm['detection_stats']['detection_accuracy'], 4)} |
"""

readme += f"""
## 🚀 Usage

```python
from unsloth import FastVisionModel
from PIL import Image

model, tokenizer = FastVisionModel.from_pretrained(
    "{hub_model_id}",
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)

image = Image.open("your_image.jpg")

messages = [
    {{
        "role": "user",
        "content": [
            {{"type": "image", "image": image}},
            {{"type": "text", "text": "IMAGE QUESTION: Describe this image.\\n\\nRESPONSE TO CHECK: The image shows a blue car.\\n\\nFind factual errors. Output JSON array or []."}},
        ],
    }}
]

inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

output = model.generate(input_ids=inputs, max_new_tokens=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 📝 Output Format

```json
[{{{"text": "blue", "label": "mischaracterization", "prob": 0.9}}}}]
```

Supported hallucination labels:
- `invention` — object/entity that doesn't exist in the image
- `mischaracterization` — wrong attribute (color, shape, size, etc.)
- `miscounting` — wrong number/count
- `OCR` — misquoted text from the image

## 🏋️ Training Framework

- [Unsloth](https://github.com/unslothai/unsloth) for efficient vision model training
- [TRL](https://github.com/huggingface/trl) GRPOTrainer for reinforcement learning
- [SHROOM-Visions](https://github.com/Aman-byte1/Hallucination-Detection-in-LVLMs) dataset and evaluation

---
*Model card auto-generated on {now}*
"""

# ── Upload to HuggingFace ──
api = HfApi()
try:
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=hub_model_id,
        repo_type="model",
        token=hf_token,
        commit_message="Update model card with evaluation results",
    )
    print(f"  ✓ Model card uploaded to https://huggingface.co/{hub_model_id}")
except Exception as e:
    print(f"  WARNING: Failed to upload model card: {e}")
    # Save locally as fallback
    with open("model_card_README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    print("  Saved locally to model_card_README.md")
MODELCARD_SCRIPT

echo ""
echo "============================================================"
echo "  ✓ SHROOM-Visions GRPO RL Pipeline Complete!"
echo "============================================================"
echo ""
echo "  GRPO model:"
echo "    Local:       ${OUTPUT_DIR}/merged"
echo "    HuggingFace: https://huggingface.co/${HUB_MODEL_ID}"
echo ""
echo "  Evaluation results:"
echo "    GRPO:  outputs_grpo/metrics_en.json"
echo "    SFT:   outputs_sft/metrics_en.json"
echo ""
echo "============================================================"
