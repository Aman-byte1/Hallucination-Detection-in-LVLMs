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
    # Cache fused probability vectors per sample
    cached_fused_samples = []
    for sample_id in common_ids:
        p1 = spancalib_preds[sample_id]
        p2 = vlm_preds[sample_id]
        response_text = p1["response"]
        resp_len = len(response_text)
        gold_labels = p1["gold_labels"]

        probs1 = labels_to_char_probs(p1.get("pred_labels", []), resp_len)
        probs2 = labels_to_char_probs(p2.get("pred_labels", []), resp_len)

        if args.fusion_mode == "union_calibrated":
            # Uses SpanCalib-VLM continuous probabilities guided by Qwen span agreement
            fused_probs = np.where(probs2 > 0, np.maximum(probs1, 0.45), probs1 * 0.35)
        elif args.fusion_mode == "intersection":
            fused_probs = np.where((probs1 >= args.threshold) & (probs2 > 0), (w1 * probs1 + w2 * probs2), 0.0)
        else:
            fused_probs = w1 * probs1 + w2 * probs2

        cached_fused_samples.append({
            "id": sample_id,
            "prompt": p1.get("prompt", ""),
            "response": response_text,
            "gold_labels": gold_labels,
            "fused_probs": fused_probs,
        })

    # Threshold Grid Search
    threshold_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    best_t_iou = -1.0
    best_t = args.threshold
    grid_results = []

    n_total = len(cached_fused_samples)
    n_gold_halluc = sum(1 for item in cached_fused_samples if len(item["gold_labels"]) > 0)
    n_clean = n_total - n_gold_halluc

    for t in threshold_grid:
        t_ious = []
        t_halluc_ious = []
        t_clean_ious = []
        t_clean_corr = 0
        t_halluc_corr = 0

        for item in cached_fused_samples:
            resp_len = len(item["response"])
            g_labels = item["gold_labels"]
            ens_spans = probs_to_contiguous_spans(item["fused_probs"], threshold=t)
            iou = compute_iou(g_labels, ens_spans, resp_len)
            t_ious.append(iou)

            has_g = len(g_labels) > 0
            has_p = len(ens_spans) > 0

            if has_g:
                t_halluc_ious.append(iou)
                if has_p:
                    t_halluc_corr += 1
            else:
                t_clean_ious.append(iou)
                if not has_p:
                    t_clean_corr += 1

        mean_iou = float(np.mean(t_ious))
        grid_results.append([
            f"{t:.2f}",
            f"{mean_iou:.3f}",
            f"{np.mean(t_halluc_ious):.3f}" if t_halluc_ious else "0.0",
            f"{np.mean(t_clean_ious):.3f}" if t_clean_ious else "0.0",
            f"{(t_clean_corr + t_halluc_corr) / n_total:.3f}",
            f"{t_clean_corr}/{n_clean}",
            f"{t_halluc_corr}/{n_gold_halluc}",
        ])

        if mean_iou > best_t_iou:
            best_t_iou = mean_iou
            best_t = t

    # Target threshold evaluation
    ensemble_metrics = []
    ensemble_jsonl = []

    for item in cached_fused_samples:
        response_text = item["response"]
        resp_len = len(response_text)
        gold_labels = item["gold_labels"]
        ens_spans = probs_to_contiguous_spans(item["fused_probs"], threshold=args.threshold)

        iou = compute_iou(gold_labels, ens_spans, resp_len)
        cal = compute_calibration(gold_labels, ens_spans, resp_len)

        has_g = len(gold_labels) > 0
        has_p = len(ens_spans) > 0

        ensemble_metrics.append({
            "id": item["id"], "iou": iou, "calibration": cal, "has_gold": has_g, "has_pred": has_p,
        })
        ensemble_jsonl.append({
            "id": item["id"], "prompt": item["prompt"], "response": response_text,
            "gold_labels": gold_labels, "pred_labels": ens_spans,
        })

    # Summary Statistics
    ious = [m["iou"] for m in ensemble_metrics]
    cals = [m["calibration"] for m in ensemble_metrics if m["calibration"] is not None]
    halluc_ious = [m["iou"] for m in ensemble_metrics if m["has_gold"]]
    clean_ious = [m["iou"] for m in ensemble_metrics if not m["has_gold"]]

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

    print(f"\n🔍 Threshold Grid Search (Best threshold: {best_t:.2f} -> Peak Overall IoU: {best_t_iou:.3f}):")
    print(tabulate(grid_results, headers=["Threshold", "Overall IoU", "Halluc IoU", "Clean IoU", "Det Acc", "Clean Correct", "Halluc Correct"], tablefmt="rounded_outline"))

    summary_table = [
        [f"Overall IoU (t={args.threshold:.2f})", f"{overall_iou:.3f}"],
        ["Peak Overall IoU", f"{best_t_iou:.3f} (at t={best_t:.2f})"],
        ["Hallucinated IoU", f"{halluc_iou_val:.3f}"],
        ["Clean IoU", f"{clean_iou_val:.3f}"],
        ["Calibration Pearson", f"{mean_cal:.3f}"],
        ["Detection Accuracy", f"{det_acc:.3f}"],
        ["Clean correct", f"{n_correct_clean}/{n_clean} ({100*n_correct_clean/n_clean:.1f}%)"],
        ["Halluc correct", f"{n_correct_halluc}/{n_gold_halluc} ({100*n_correct_halluc/n_gold_halluc:.1f}%)"],
    ]

    print(f"\n🏆 Ensemble Benchmark Metrics for selected threshold ({args.threshold:.2f}):")
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
    parser.add_argument("--fusion_mode", choices=["weighted", "union_calibrated", "intersection"], default="union_calibrated")
    parser.add_argument("--weight_spancalib", type=float, default=0.55)
    parser.add_argument("--weight_vlm", type=float, default=0.45)
    parser.add_argument("--threshold", type=float, default=0.40)
    return parser.parse_args()


if __name__ == "__main__":
    run_ensemble(parse_args())
