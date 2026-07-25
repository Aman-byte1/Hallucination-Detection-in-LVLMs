"""
SpanCalib-VLM Ensemble Pipeline
================================
Ensembles predictions from SpanCalib-VLM sequence tagger and Qwen / VLM generative models
using weighted character-level probability fusion and thresholding.

Usage:
    python spancalib_vlm/ensemble.py \
        --spancalib_file outputs_spancalib_vlm/predictions_en.jsonl \
        --vlm_file outputs/predictions_en.jsonl \
        --weight_spancalib 0.55 \
        --weight_vlm 0.45 \
        --threshold 0.50
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent.parent))

from spancalib_vlm.evaluate import (
    labels_to_char_binary,
    labels_to_char_probs,
    compute_iou,
    compute_calibration,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_jsonl_predictions(filepath: Path) -> dict[str, dict]:
    preds_map = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                preds_map[item["id"]] = item
    return preds_map


def probs_to_contiguous_spans(
    char_probs: np.ndarray,
    threshold: float = 0.50,
    default_label: str = "invention",
) -> list[dict]:
    """Convert a character-level probability array into contiguous span objects."""
    resp_len = len(char_probs)
    if resp_len == 0:
        return []

    spans = []
    in_span = False
    start_idx = 0

    for i in range(resp_len):
        if char_probs[i] >= threshold and not in_span:
            in_span = True
            start_idx = i
        elif char_probs[i] < threshold and in_span:
            in_span = False
            end_idx = i
            avg_prob = float(np.mean(char_probs[start_idx:end_idx]))
            spans.append({
                "start": start_idx,
                "end": end_idx,
                "label": default_label,
                "prob": round(avg_prob, 4),
            })

    if in_span:
        end_idx = resp_len
        avg_prob = float(np.mean(char_probs[start_idx:end_idx]))
        spans.append({
            "start": start_idx,
            "end": end_idx,
            "label": default_label,
            "prob": round(avg_prob, 4),
        })

    return spans


def run_ensemble(args):
    spancalib_path = Path(args.spancalib_file)
    vlm_path = Path(args.vlm_file)

    if not spancalib_path.exists():
        logger.error(f"SpanCalib predictions file not found: {spancalib_path}")
        sys.exit(1)
    if not vlm_path.exists():
        logger.error(f"VLM predictions file not found: {vlm_path}")
        sys.exit(1)

    spancalib_preds = load_jsonl_predictions(spancalib_path)
    vlm_preds = load_jsonl_predictions(vlm_path)

    common_ids = sorted(list(set(spancalib_preds.keys()).intersection(set(vlm_preds.keys()))))
    logger.info(f"Ensembling {len(common_ids)} samples across both models...")
    logger.info(f"Weights: SpanCalib-VLM={args.weight_spancalib:.2f}, VLM={args.weight_vlm:.2f}")

    # Normalization of weights
    w1 = args.weight_spancalib
    w2 = args.weight_vlm
    total_w = w1 + w2
    w1, w2 = w1 / total_w, w2 / total_w

    ensemble_metrics = []
    ensemble_jsonl = []

    for sample_id in common_ids:
        p1 = spancalib_preds[sample_id]
        p2 = vlm_preds[sample_id]

        response_text = p1["response"]
        resp_len = len(response_text)
        gold_labels = p1["gold_labels"]

        # Character probability vectors
        probs1 = labels_to_char_probs(p1.get("pred_labels", []), resp_len)
        probs2 = labels_to_char_probs(p2.get("pred_labels", []), resp_len)

        # Weighted Character Probability Fusion
        fused_probs = w1 * probs1 + w2 * probs2

        # Reconstruct Ensembled Spans
        ens_spans = probs_to_contiguous_spans(fused_probs, threshold=args.threshold)

        iou = compute_iou(gold_labels, ens_spans, resp_len)
        cal = compute_calibration(gold_labels, ens_spans, resp_len)

        has_g = len(gold_labels) > 0
        has_p = len(ens_spans) > 0

        ensemble_metrics.append({
            "id": sample_id,
            "iou": iou,
            "calibration": cal,
            "has_gold": has_g,
            "has_pred": has_p,
        })

        ensemble_jsonl.append({
            "id": sample_id,
            "prompt": p1.get("prompt", ""),
            "response": response_text,
            "gold_labels": gold_labels,
            "pred_labels": ens_spans,
        })

    # Summary Statistics
    ious = [m["iou"] for m in ensemble_metrics]
    cals = [m["calibration"] for m in ensemble_metrics if m["calibration"] is not None]
    halluc_ious = [m["iou"] for m in ensemble_metrics if m["has_gold"]]
    clean_ious = [m["iou"] for m in ensemble_metrics if not m["has_gold"]]

    n_total = len(ensemble_metrics)
    n_gold_halluc = sum(1 for m in ensemble_metrics if m["has_gold"])
    n_clean = n_total - n_gold_halluc
    n_correct_clean = sum(1 for m in ensemble_metrics if not m["has_gold"] and not m["has_pred"])
    n_correct_halluc = sum(1 for m in ensemble_metrics if m["has_gold"] and m["has_pred"])

    overall_iou = float(np.mean(ious)) if ious else 0.0
    mean_cal = float(np.mean(cals)) if cals else 0.0
    halluc_iou_val = float(np.mean(halluc_ious)) if halluc_ious else 0.0
    clean_iou_val = float(np.mean(clean_ious)) if clean_ious else 0.0
    det_acc = (n_correct_clean + n_correct_halluc) / n_total if n_total > 0 else 0.0

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in ensemble_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("  SHROOM-Visions Evaluation Summary — ENSEMBLE MODEL")
    print(f"  Model 1: SpanCalib-VLM (weight: {w1:.2f})")
    print(f"  Model 2: Generative VLM (weight: {w2:.2f})")
    print(f"  Eval Samples: {n_total} (Hallucinated: {n_gold_halluc}, Clean: {n_clean})")
    print("=" * 70)

    summary_table = [
        [f"Overall IoU (t={args.threshold:.2f})", f"{overall_iou:.3f}"],
        ["Hallucinated IoU", f"{halluc_iou_val:.3f}"],
        ["Clean IoU", f"{clean_iou_val:.3f}"],
        ["Calibration Pearson", f"{mean_cal:.3f}"],
        ["Detection Accuracy", f"{det_acc:.3f}"],
        ["Clean correct", f"{n_correct_clean}/{n_clean} ({100*n_correct_clean/n_clean:.1f}%)"],
        ["Halluc correct", f"{n_correct_halluc}/{n_gold_halluc} ({100*n_correct_halluc/n_gold_halluc:.1f}%)"],
    ]

    print("\n🏆 Ensemble Benchmark Metrics:")
    print(tabulate(summary_table, headers=["Metric", "Value"], tablefmt="rounded_outline"))
    print(f"\n📁 Ensembled predictions saved to: {output_file}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Ensemble of SpanCalib-VLM + VLM Model")
    parser.add_argument(
        "--spancalib_file",
        default="outputs_spancalib_vlm/predictions_en.jsonl",
        help="Path to SpanCalib-VLM predictions JSONL",
    )
    parser.add_argument(
        "--vlm_file",
        default="outputs/predictions_en.jsonl",
        help="Path to VLM (Qwen/MiniCPM/BLIP) predictions JSONL",
    )
    parser.add_argument("--output_file", default="outputs_ensemble/predictions_en.jsonl")
    parser.add_argument("--weight_spancalib", type=float, default=0.55)
    parser.add_argument("--weight_vlm", type=float, default=0.45)
    parser.add_argument("--threshold", type=float, default=0.50)
    return parser.parse_args()


if __name__ == "__main__":
    run_ensemble(parse_args())
