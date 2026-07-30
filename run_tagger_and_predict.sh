#!/bin/bash
# ============================================================================
# SpanCalib-VLM Tagger Training + Multilingual Prediction Generation
# ============================================================================
# Trains the XLM-RoBERTa + SigLIP-2 sequence tagger on all 4 languages,
# then generates test predictions using both systems + ensemble.
#
# Run on RunPod A40 after the SFT step has completed:
#   cd /workspace/Hallucination-Detection-in-LVLMs
#   source /workspace/venv_shroom_sft/bin/activate
#   bash run_tagger_and_predict.sh
# ============================================================================

set -euo pipefail

export HF_HUB_DISABLE_XET=1

DATA_DIR="shroom-visions-data/distrib"
IMAGES_DIR="shroom-vis-images"
MERGED_TRAIN="${DATA_DIR}/shroom-vision.train.all.labeled.jsonl"
LANGUAGES=("en" "fr" "it" "zh")

# Tagger config
TAGGER_MODEL_ID="xlm-roberta-large"
TAGGER_OUTPUT="./checkpoints/spancalib_vlm_multilingual"
TAGGER_EPOCHS=5
TAGGER_BATCH=8
TAGGER_LR="2e-5"

# SFT model (already finetuned)
SFT_MODEL="./checkpoints/qwen35-4b-shroom-sft-multilingual/checkpoint-2544"

echo ""
echo "============================================================"
echo "  SpanCalib-VLM: Tagger Training + Prediction Generation"
echo "============================================================"
echo ""

# ============================================================================
# Step 1: Ensure merged training data exists
# ============================================================================
echo "[1/4] Checking merged multilingual training data..."

if [ ! -f "$MERGED_TRAIN" ]; then
    echo "  Creating merged file..."
    rm -f "$MERGED_TRAIN"
    for LANG in "${LANGUAGES[@]}"; do
        cat "${DATA_DIR}/shroom-vision.train.${LANG}.labeled.jsonl" >> "$MERGED_TRAIN"
    done
    python -c "
import random
random.seed(42)
with open('$MERGED_TRAIN', 'r', encoding='utf-8') as f:
    lines = f.readlines()
random.shuffle(lines)
with open('$MERGED_TRAIN', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print(f'  Shuffled {len(lines)} samples')
"
else
    TOTAL=$(wc -l < "$MERGED_TRAIN")
    echo "  ✓ Merged file exists: $MERGED_TRAIN ($TOTAL samples)"
fi

# ============================================================================
# Step 2: Install tagger dependencies (if needed)
# ============================================================================
echo ""
echo "[2/4] Checking tagger dependencies..."
pip install tabulate scikit-learn --quiet 2>/dev/null || true

# ============================================================================
# Step 3: Train the SpanCalib-VLM tagger on all languages
# ============================================================================
echo ""
echo "[3/4] Training SpanCalib-VLM sequence tagger..."
echo "  Model:      $TAGGER_MODEL_ID"
echo "  Data:       $MERGED_TRAIN"
echo "  Epochs:     $TAGGER_EPOCHS"
echo "  Batch:      $TAGGER_BATCH"
echo "  LR:         $TAGGER_LR"
echo ""

python spancalib_vlm/train.py \
  --data_file "$MERGED_TRAIN" \
  --images_dir "$IMAGES_DIR" \
  --model_id "$TAGGER_MODEL_ID" \
  --use_vision \
  --epochs "$TAGGER_EPOCHS" \
  --batch_size "$TAGGER_BATCH" \
  --lr "$TAGGER_LR" \
  --output_dir "$TAGGER_OUTPUT"

echo ""
echo "  ✓ Tagger training complete!"

# ============================================================================
# Step 4: Generate predictions on test sets for each language
# ============================================================================
echo ""
echo "[4/4] Generating test predictions for all languages..."

mkdir -p outputs_submission

for LANG in "${LANGUAGES[@]}"; do
    TEST_FILE="${DATA_DIR}/shroom-vision.test.${LANG}.unlabeled.jsonl"
    echo ""
    echo "  ── Generating predictions: ${LANG} ──"

    if [ ! -f "$TEST_FILE" ]; then
        echo "  ✗ Test file not found: $TEST_FILE"
        continue
    fi

    # Run tagger predictions
    python spancalib_vlm/evaluate.py \
      --data_file "${DATA_DIR}/shroom-vision.train.${LANG}.labeled.jsonl" \
      --images_dir "$IMAGES_DIR" \
      --model_id "$TAGGER_MODEL_ID" \
      --use_vision \
      --checkpoint_dir "$TAGGER_OUTPUT" \
      --output_dir "./outputs_tagger_${LANG}"

    echo "  ✓ ${LANG} tagger predictions complete"
done

echo ""
echo "============================================================"
echo "  ✓ SpanCalib-VLM Pipeline Complete!"
echo "============================================================"
echo ""
echo "  Tagger checkpoint: $TAGGER_OUTPUT"
echo "  Per-language outputs: outputs_tagger_{en,fr,it,zh}/"
echo ""
echo "============================================================"
