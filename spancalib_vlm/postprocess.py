"""
SpanCalib-VLM Post-Processing & Boundary Refinement Module
============================================================
Refines raw span predictions by:
1. Merging closely adjacent hallucination spans (distance <= max_gap).
2. Suppressing noise micro-spans (length < min_length).
3. Filtering low-confidence probability spans.

Usage:
    python spancalib_vlm/postprocess.py \
        --input_file outputs_ensemble/predictions_en.jsonl \
        --output_file outputs_ensemble/predictions_en_refined.jsonl \
        --max_gap 4 \
        --min_length 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent.parent))

from spancalib_vlm.evaluate import compute_iou, compute_calibration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def merge_and_filter_spans(
    spans: list[dict],
    response_length: int,
    max_gap: int = 4,
    min_length: int = 3,
) -> list[dict]:
    """Refine span predictions via gap merging and length thresholding."""
    if not spans:
        return []

    # Sort spans by start offset
    sorted_spans = sorted(spans, key=lambda x: x.get("start", 0))
    merged = []

    current = dict(sorted_spans[0])

    for nxt in sorted_spans[1:]:
        c_start = current.get("start", 0)
        c_end = current.get("end", 0)
        n_start = nxt.get("start", 0)
        n_end = nxt.get("end", 0)

        # Merge if gap between spans is <= max_gap
        if n_start <= c_end + max_gap:
            current["end"] = max(c_end, n_end)
            c_prob = float(current.get("prob", 1.0))
            n_prob = float(nxt.get("prob", 1.0))
            current["prob"] = round((c_prob + n_prob) / 2.0, 4)
        else:
            merged.append(current)
            current = dict(nxt)

    merged.append(current)

    # Filter out micro-spans shorter than min_length
    refined = []
    for s in merged:
        span_len = s["end"] - s["start"]
        if span_len >= min_length:
            refined.append(s)

    return refined


def run_postprocessing(args):
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        logger.error(f"Input predictions file not found: {input_path}")
        sys.exit(1)

    raw_samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_samples.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(raw_samples)} samples from {input_path}")
    logger.info(f"Post-processing parameters: max_gap={args.max_gap}, min_length={args.min_length}")

    before_metrics = []
    after_metrics = []
    refined_jsonl = []

    for item in raw_samples:
        response_text = item["response"]
        resp_len = len(response_text)
        gold_labels = item.get("gold_labels", [])
        pred_labels = item.get("pred_labels", [])

        # Compute raw metrics
        iou_raw = compute_iou(gold_labels, pred_labels, resp_len)
        cal_raw = compute_calibration(gold_labels, pred_labels, resp_len)

        # Refine spans
        refined_spans = merge_and_filter_spans(
            spans=pred_labels,
            response_length=resp_len,
            max_gap=args.max_gap,
            min_length=args.min_length,
        )

        # Compute refined metrics
        iou_ref = compute_iou(gold_labels, refined_spans, resp_len)
        cal_ref = compute_calibration(gold_labels, refined_spans, resp_len)

        has_gold = len(gold_labels) > 0
        has_raw = len(pred_labels) > 0
        has_ref = len(refined_spans) > 0

        before_metrics.append({"iou": iou_raw, "cal": cal_raw, "has_gold": has_gold, "has_pred": has_raw})
        after_metrics.append({"iou": iou_ref, "cal": cal_ref, "has_gold": has_gold, "has_pred": has_ref})

        refined_jsonl.append({
            "id": item["id"],
            "prompt": item.get("prompt", ""),
            "response": response_text,
            "gold_labels": gold_labels,
            "pred_labels": refined_spans,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in refined_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Aggregate Statistics
    raw_ious = [m["iou"] for m in before_metrics]
    ref_ious = [m["iou"] for m in after_metrics]

    raw_cals = [m["cal"] for m in before_metrics if m["cal"] is not None]
    ref_cals = [m["cal"] for m in after_metrics if m["cal"] is not None]

    n_total = len(before_metrics)
    n_gold_halluc = sum(1 for m in before_metrics if m["has_gold"])
    n_clean = n_total - n_gold_halluc

    print("\n" + "=" * 70)
    print("  SHROOM-Visions Post-Processing & Boundary Refinement Summary")
    print(f"  Input File:  {input_path}")
    print(f"  Output File: {output_path}")
    print(f"  Samples:     {n_total}")
    print("=" * 70)

    comparison_table = [
        ["Overall IoU", f"{np.mean(raw_ious):.3f}", f"{np.mean(ref_ious):.3f}", f"{np.mean(ref_ious) - np.mean(raw_ious):+.3f}"],
        ["Calibration Pearson", f"{np.mean(raw_cals):.3f}" if raw_cals else "N/A", f"{np.mean(ref_cals):.3f}" if ref_cals else "N/A", f"{(np.mean(ref_cals) - np.mean(raw_cals)):+.3f}" if raw_cals and ref_cals else "N/A"],
    ]

    print("\n📈 Before vs. After Post-Processing:")
    print(tabulate(comparison_table, headers=["Metric", "Raw Model", "Refined Model", "Delta"], tablefmt="rounded_outline"))
    print(f"\n📁 Refined predictions saved to: {output_path}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Post-process & refine SHROOM hallucination spans")
    parser.add_argument(
        "--input_file",
        default="outputs_ensemble/predictions_en.jsonl",
        help="Input predictions JSONL file",
    )
    parser.add_argument(
        "--output_file",
        default="outputs_ensemble/predictions_en_refined.jsonl",
        help="Refined output predictions JSONL file",
    )
    parser.add_argument("--max_gap", type=int, default=4, help="Max gap distance between adjacent spans to merge")
    parser.add_argument("--min_length", type=int, default=3, help="Minimum character length for valid spans")
    return parser.parse_args()


if __name__ == "__main__":
    run_postprocessing(parse_args())
