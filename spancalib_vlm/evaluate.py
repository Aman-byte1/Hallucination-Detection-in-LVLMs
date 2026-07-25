"""
SpanCalib-VLM Evaluation Pipeline
==================================
Evaluates trained SpanCalib-VLM model on SHROOM-Visions test data,
reconstructs character-level spans from predicted token probabilities,
and outputs standardized metrics table.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from spancalib_vlm.dataset import REVERSE_CATEGORY_MAP, find_image
from spancalib_vlm.model import SpanCalibVLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Standardized Metrics (same as main evaluate.py)
# ============================================================================

def labels_to_char_binary(labels: list[dict], response_length: int) -> np.ndarray:
    arr = np.zeros(response_length, dtype=np.float64)
    for label in labels:
        start = max(0, label["start"])
        end = min(response_length, label["end"])
        arr[start:end] = 1.0
    return arr


def labels_to_char_probs(labels: list[dict], response_length: int) -> np.ndarray:
    arr = np.zeros(response_length, dtype=np.float64)
    for label in labels:
        start = max(0, label["start"])
        end = min(response_length, label["end"])
        prob = label.get("prob", 1.0)
        arr[start:end] = np.maximum(arr[start:end], prob)
    return arr


def compute_iou(gold_labels: list[dict], pred_labels: list[dict], response_length: int) -> float:
    gold_arr = labels_to_char_binary(gold_labels, response_length)
    pred_arr = labels_to_char_binary(pred_labels, response_length)
    intersection = np.sum(gold_arr * pred_arr)
    union = np.sum(np.maximum(gold_arr, pred_arr))
    return 1.0 if union == 0 else float(intersection / union)


def compute_calibration(gold_labels: list[dict], pred_labels: list[dict], response_length: int) -> float | None:
    gold_probs = labels_to_char_probs(gold_labels, response_length)
    pred_probs = labels_to_char_probs(pred_labels, response_length)
    if np.std(gold_probs) == 0 and np.std(pred_probs) == 0:
        return 1.0
    if np.std(gold_probs) == 0 or np.std(pred_probs) == 0:
        return 0.0
    corr, _ = pearsonr(gold_probs, pred_probs)
    return float(corr) if not np.isnan(corr) else 0.0


# ============================================================================
# Token-to-Character Span Reconstruction
# ============================================================================

def reconstruct_spans_from_tokens(
    token_probs: np.ndarray,
    token_cats: np.ndarray,
    offsets: list[tuple[int, int]],
    response_text: str,
    threshold: float = 0.50,
) -> list[dict]:
    """Reconstruct contiguous character spans from predicted token probabilities."""
    resp_len = len(response_text)
    if resp_len == 0:
        return []

    # Map token probabilities to character array
    char_probs = np.zeros(resp_len, dtype=np.float32)
    char_cats = np.zeros(resp_len, dtype=np.int64)

    for i, (s, e) in enumerate(offsets):
        if s >= e or s >= resp_len:
            continue
        e_clamped = min(resp_len, e)
        if token_probs[i] >= threshold:
            char_probs[s:e_clamped] = np.maximum(char_probs[s:e_clamped], token_probs[i])
            char_cats[s:e_clamped] = token_cats[i]

    # Find contiguous spans where char_probs >= threshold
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
            cat_idx = int(np.bincount(char_cats[start_idx:end_idx]).argmax())
            cat_name = REVERSE_CATEGORY_MAP.get(cat_idx, "invention")
            spans.append({
                "start": start_idx,
                "end": end_idx,
                "label": cat_name,
                "prob": round(avg_prob, 4),
            })

    if in_span:
        end_idx = resp_len
        avg_prob = float(np.mean(char_probs[start_idx:end_idx]))
        cat_idx = int(np.bincount(char_cats[start_idx:end_idx]).argmax())
        cat_name = REVERSE_CATEGORY_MAP.get(cat_idx, "invention")
        spans.append({
            "start": start_idx,
            "end": end_idx,
            "label": cat_name,
            "prob": round(avg_prob, 4),
        })

    return spans


# ============================================================================
# Main Evaluation Loop
# ============================================================================

def load_data(filepath: Path) -> list[dict]:
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def split_eval_data(samples: list[dict], ratio: float = 0.10, seed: int = 42) -> list[dict]:
    rng = np.random.RandomState(seed)
    n_eval = max(1, int(len(samples) * ratio))
    indices = rng.choice(len(samples), size=n_eval, replace=False)
    return [samples[i] for i in sorted(indices)]


def evaluate(args):
    start_time = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "predictions_en.csv"
    json_path = output_dir / "metrics_en.json"
    predictions_jsonl_path = output_dir / "predictions_en.jsonl"

    logger.info("=" * 60)
    logger.info("  SHROOM-Visions Evaluation: SpanCalib-VLM")
    logger.info("=" * 60)

    # 1. Load Data
    all_samples = load_data(Path(args.data_file))
    eval_samples = split_eval_data(all_samples, ratio=0.10, seed=42)
    images_dir = Path(args.images_dir)

    if args.max_samples and args.max_samples < len(eval_samples):
        eval_samples = eval_samples[:args.max_samples]

    logger.info(f"Evaluating {len(eval_samples)} samples...")

    # 2. Load Model & Tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    image_processor = None
    if args.use_vision:
        from transformers import SiglipImageProcessor
        image_processor = SiglipImageProcessor.from_pretrained(args.vision_model_id)

    model = SpanCalibVLM(model_id=args.model_id, use_vision=args.use_vision)

    checkpoint_arg = getattr(args, "checkpoint", None) or args.checkpoint_dir
    checkpoint_path = Path(checkpoint_arg)
    if checkpoint_path.is_file():
        checkpoint_file = checkpoint_path
    else:
        checkpoint_file = checkpoint_path / "best_model.pt"
        if not checkpoint_file.exists():
            checkpoint_file = checkpoint_path / "final_model.pt"

    if checkpoint_file.exists():
        logger.info(f"Loading trained weights from {checkpoint_file}")
        model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_file}. Running with initialized weights.")

    model.to(device)
    model.eval()

    # 3. Inference Loop
    predictions = []
    per_sample_metrics = []

    for sample in tqdm(eval_samples, desc="Evaluating"):
        prompt_text = sample["prompt"]
        response_text = sample["response"]
        gold_labels = sample.get("labels", [])
        resp_len = len(response_text)

        formatted_prompt = f"Question: {prompt_text}\nResponse: {response_text}"

        encoded = tokenizer(
            formatted_prompt,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        offsets = encoded["offset_mapping"].squeeze(0).tolist()

        pixel_values = None
        if args.use_vision and image_processor is not None:
            img_name = sample.get("image_name", "")
            img_path = find_image(img_name, images_dir)
            if img_path and img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224), (255, 255, 255))
            else:
                img = Image.new("RGB", (224, 224), (255, 255, 255))
            pixel_values = image_processor(images=img, return_tensors="pt")["pixel_values"].to(device)

        resp_start = formatted_prompt.find(response_text)
        if resp_start == -1:
            resp_start = 0

        adjusted_offsets = []
        for s, e in offsets:
            if s >= resp_start and e > resp_start:
                adjusted_offsets.append((s - resp_start, e - resp_start))
            else:
                adjusted_offsets.append((0, 0))

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            pred_probs = outputs["pred_probs"].squeeze(0).cpu().numpy()
            cat_logits = outputs["cat_logits"].squeeze(0).cpu().numpy()
            token_cats = cat_logits.argmax(axis=-1)

        # Reconstruct spans
        pred_labels = reconstruct_spans_from_tokens(
            token_probs=pred_probs,
            token_cats=token_cats,
            offsets=adjusted_offsets,
            response_text=response_text,
            threshold=args.threshold,
        )

        iou = compute_iou(gold_labels, pred_labels, resp_len)
        cal = compute_calibration(gold_labels, pred_labels, resp_len)

        has_gold = len(gold_labels) > 0
        has_pred = len(pred_labels) > 0

        per_sample_metrics.append({
            "id": sample["id"],
            "iou": iou,
            "calibration": cal,
            "gold_span_count": len(gold_labels),
            "pred_span_count": len(pred_labels),
            "has_gold_hallucination": has_gold,
            "has_pred_hallucination": has_pred,
            "response_length": resp_len,
        })

        # Save raw predictions per sample for threshold grid search
        predictions.append({
            "id": sample["id"],
            "prompt": prompt_text,
            "image_name": sample.get("image_name", ""),
            "response": response_text,
            "gold_labels": gold_labels,
            "token_probs": pred_probs,
            "token_cats": token_cats,
            "adjusted_offsets": adjusted_offsets,
        })

    # 4. Threshold Grid Search & Optimization
    n_total = len(predictions)
    n_gold_halluc = sum(1 for p in predictions if len(p["gold_labels"]) > 0)
    n_clean = n_total - n_gold_halluc

    threshold_grid = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    best_t_iou = -1.0
    best_t = args.threshold
    grid_results = []

    for t in threshold_grid:
        t_ious = []
        t_halluc_ious = []
        t_clean_ious = []
        t_clean_corr = 0
        t_halluc_corr = 0

        for p in predictions:
            resp_len = len(p["response"])
            g_labels = p["gold_labels"]
            p_spans = reconstruct_spans_from_tokens(
                token_probs=p["token_probs"],
                token_cats=p["token_cats"],
                offsets=p["adjusted_offsets"],
                response_text=p["response"],
                threshold=t,
            )

            iou = compute_iou(g_labels, p_spans, resp_len)
            t_ious.append(iou)

            has_g = len(g_labels) > 0
            has_p = len(p_spans) > 0

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
            f"{(t_clean_corr + t_halluc_corr) / len(predictions):.3f}",
            f"{t_clean_corr}/{n_clean}",
            f"{t_halluc_corr}/{n_gold_halluc}",
        ])

        if mean_iou > best_t_iou:
            best_t_iou = mean_iou
            best_t = t

    # Compute metrics for target threshold
    per_sample_metrics = []
    final_preds = []
    for p in predictions:
        resp_len = len(p["response"])
        g_labels = p["gold_labels"]
        p_spans = reconstruct_spans_from_tokens(
            token_probs=p["token_probs"],
            token_cats=p["token_cats"],
            offsets=p["adjusted_offsets"],
            response_text=p["response"],
            threshold=args.threshold,
        )
        iou = compute_iou(g_labels, p_spans, resp_len)
        cal = compute_calibration(g_labels, p_spans, resp_len)
        per_sample_metrics.append({
            "id": p["id"], "iou": iou, "calibration": cal,
            "gold_span_count": len(g_labels), "pred_span_count": len(p_spans),
            "has_gold_hallucination": len(g_labels) > 0,
            "has_pred_hallucination": len(p_spans) > 0,
        })
        final_preds.append({
            "id": p["id"], "prompt": p["prompt"], "image_name": p["image_name"],
            "response": p["response"], "gold_labels": g_labels, "pred_labels": p_spans,
        })

    # 5. Aggregate Metrics
    iou_scores = [m["iou"] for m in per_sample_metrics]
    cal_scores = [m["calibration"] for m in per_sample_metrics if m["calibration"] is not None]
    halluc_iou = [m["iou"] for m in per_sample_metrics if m["has_gold_hallucination"]]
    clean_iou = [m["iou"] for m in per_sample_metrics if not m["has_gold_hallucination"]]

    n_total = len(per_sample_metrics)
    n_gold_halluc = sum(1 for m in per_sample_metrics if m["has_gold_hallucination"])
    n_clean = n_total - n_gold_halluc
    n_correct_clean = sum(1 for m in per_sample_metrics if not m["has_gold_hallucination"] and not m["has_pred_hallucination"])
    n_correct_halluc = sum(1 for m in per_sample_metrics if m["has_gold_hallucination"] and m["has_pred_hallucination"])

    overall_results = {
        "model": args.model_id,
        "eval_samples": n_total,
        "best_threshold_for_iou": best_t,
        "best_overall_iou": best_t_iou,
        "metrics": {
            "overall": {
                "iou_mean": float(np.mean(iou_scores)) if iou_scores else 0.0,
                "calibration_mean": float(np.mean(cal_scores)) if cal_scores else 0.0,
            },
            "hallucinated_samples": {
                "count": n_gold_halluc,
                "iou_mean": float(np.mean(halluc_iou)) if halluc_iou else 0.0,
            },
            "clean_samples": {
                "count": n_clean,
                "iou_mean": float(np.mean(clean_iou)) if clean_iou else 0.0,
            },
            "detection_stats": {
                "correct_clean": n_correct_clean,
                "total_clean": n_clean,
                "correct_halluc": n_correct_halluc,
                "total_halluc": n_gold_halluc,
                "detection_accuracy": (n_correct_clean + n_correct_halluc) / n_total if n_total > 0 else 0.0,
            },
        },
        "timing": {
            "total_seconds": round(time.time() - start_time, 2),
        },
    }

    # 6. Save Outputs
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "prompt", "image_name", "response_length",
            "gold_span_count", "pred_span_count",
            "gold_labels_json", "pred_labels_json", "iou", "calibration",
        ])
        for pred, m in zip(final_preds, per_sample_metrics):
            writer.writerow([
                pred["id"], pred["prompt"], pred["image_name"],
                len(pred["response"]), m["gold_span_count"], m["pred_span_count"],
                json.dumps(pred["gold_labels"], ensure_ascii=False),
                json.dumps(pred["pred_labels"], ensure_ascii=False),
                f"{m['iou']:.6f}",
                f"{m['calibration']:.6f}" if m["calibration"] is not None else "",
            ])

    with open(predictions_jsonl_path, "w", encoding="utf-8") as f:
        for pred in final_preds:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2, ensure_ascii=False)

    # 7. Print Standardized Table & Grid Search Summary
    det_stats = overall_results["metrics"]["detection_stats"]
    overall_iou = overall_results["metrics"]["overall"]["iou_mean"]
    halluc_iou_val = overall_results["metrics"]["hallucinated_samples"]["iou_mean"]
    clean_iou_val = overall_results["metrics"]["clean_samples"]["iou_mean"]
    det_acc = det_stats["detection_accuracy"]

    print("\n" + "=" * 70)
    print("  SHROOM-Visions Evaluation Summary — SpanCalib-VLM")
    print(f"  Model: {args.model_id}")
    print(f"  Eval Samples: {n_total} (Hallucinated: {n_gold_halluc}, Clean: {n_clean})")
    print(f"  Time: {overall_results['timing']['total_seconds']:.1f}s")
    print("=" * 70)

    print(f"\n🔍 Threshold Grid Search (Best threshold: {best_t:.2f} -> Overall IoU: {best_t_iou:.3f}):")
    print(tabulate(grid_results, headers=["Threshold", "Overall IoU", "Halluc IoU", "Clean IoU", "Det Acc", "Clean Correct", "Halluc Correct"], tablefmt="rounded_outline"))

    summary_table = [
        [f"Overall IoU (t={args.threshold:.2f})", f"{overall_iou:.3f}"],
        ["Peak Overall IoU", f"{best_t_iou:.3f} (at t={best_t:.2f})"],
        ["Hallucinated IoU", f"{halluc_iou_val:.3f}"],
        ["Clean IoU", f"{clean_iou_val:.3f}"],
        ["Detection Accuracy", f"{det_acc:.3f}"],
        ["Clean correct",
         f"{n_correct_clean}/{n_clean} ({100*n_correct_clean/n_clean:.1f}%)" if n_clean > 0 else "0/0"],
        ["Halluc correct",
         f"{n_correct_halluc}/{n_gold_halluc} ({100*n_correct_halluc/n_gold_halluc:.1f}%)" if n_gold_halluc > 0 else "0/0"],
    ]
    print(f"\n📊 Metrics for selected threshold ({args.threshold:.2f}):")
    print(tabulate(summary_table, headers=["Metric", "Value"], tablefmt="rounded_outline"))
    print(f"\n📁 Results saved to: {output_dir}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SpanCalib-VLM Model")
    parser.add_argument(
        "--data_file",
        default="shroom-visions-data/distrib/shroom-vision.train.en.labeled.jsonl",
    )
    parser.add_argument("--images_dir", default="shroom-vis-images")
    parser.add_argument("--model_id", default="xlm-roberta-base")
    parser.add_argument("--use_vision", action="store_true", help="Enable SigLIP-2 vision tower cross-attention fusion")
    parser.add_argument("--vision_model_id", default="google/siglip-base-patch16-224", help="SigLIP vision model ID")
    parser.add_argument("--checkpoint", default=None, help="Direct path to checkpoint file (.pt) or checkpoint directory")
    parser.add_argument("--checkpoint_dir", default="./checkpoints/spancalib_vlm")
    parser.add_argument("--output_dir", default="./outputs_spancalib_vlm")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
