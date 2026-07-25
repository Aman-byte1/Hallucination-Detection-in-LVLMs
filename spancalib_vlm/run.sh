#!/bin/bash
# ============================================================================
# SpanCalib-VLM Pipeline Runner
# ============================================================================
# Usage:
#   chmod +x spancalib_vlm/run.sh
#   ./spancalib_vlm/run.sh            # Full training & evaluation
#   ./spancalib_vlm/run.sh --quick    # Quick test (10 samples)
# ============================================================================

set -e

QUICK=0
if [ "$1" = "--quick" ]; then
    QUICK=1
fi

echo "============================================"
echo "  SpanCalib-VLM Pipeline"
echo "============================================"

if [ "$QUICK" -eq 1 ]; then
    echo "Running quick dry-run test (10 samples)..."
    python spancalib_vlm/train.py --max_samples 10 --epochs 1
    python spancalib_vlm/evaluate.py --max_samples 10
else
    echo "Running full training (3 epochs)..."
    python spancalib_vlm/train.py --epochs 3 --batch_size 8 --lr 3e-5
    echo "Running evaluation..."
    python spancalib_vlm/evaluate.py
fi

echo "============================================"
echo "  SpanCalib-VLM Completed!"
echo "============================================"
