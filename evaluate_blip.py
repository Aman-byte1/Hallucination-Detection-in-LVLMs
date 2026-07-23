#!/usr/bin/env python3
"""
SHROOM-Visions Hallucination Detection Evaluation — BLIP-VQA-Base
==================================================================
Evaluation script for BLIP-VQA-Base (base or finetuned).
Uses the same metrics (IoU + calibration) as evaluate.py for fair comparison.

Usage:
    python evaluate_blip.py                          # Evaluate base model
    python evaluate_blip.py --model_id ./checkpoints/blip-vqa-shroom-sft/final
    python evaluate_blip.py --max_samples 10         # Quick test
    python evaluate_blip.py --resume                 # Resume from checkpoint
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

# Bypass broken system torchaudio build in container if present
sys.modules["torchaudio"] = None

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr
from tabulate import tabulate
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL_ID = "Salesforce/blip-vqa-base"
DATA_DIR = Path(__file__).parent / "shroom-visions-data" / "distrib"
IMAGES_DIR = Path(__file__).parent / "shroom-vis-images"
OUTPUT_DIR = Path(__file__).parent / "outputs_blip"
TRAIN_FILE = DATA_DIR / "shroom-vision.train.en.labeled.jsonl"
EVAL_SPLIT_RATIO = 0.10
RANDOM_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Same question template as finetune_blip.py
BLIP_QUESTION_TEMPLATE = (
    "Does this response about the image contain factual errors? "
    "Respond with a JSON array of errors or []. "
    "Response to check: {response}"
)


# ============================================================================
# Metrics (same as evaluate.py)
# ============================================================================

def labels_to_char_binary(labels, response_length):
    arr = np.zeros(response_length, dtype=np.float64)
    for label in labels:
        start = max(0, label["start"])
        end = min(response_length, label["end"])
        arr[start:end] = 1.0
    return arr


def labels_to_char_probs(labels, response_length):
    arr = np.zeros(response_length, dtype=np.float64)
    for label in labels:
        start = max(0, label["start"])
        end = min(response_length, label["end"])
        prob = label.get("prob", 1.0)
        arr[start:end] = np.maximum(arr[start:end], prob)
    return arr


def compute_iou(gold_labels, pred_labels, response_length):
    gold_arr = labels_to_char_binary(gold_labels, response_length)
    pred_arr = labels_to_char_binary(pred_labels, response_length)
    intersection = np.sum(gold_arr * pred_arr)
    union = np.sum(np.maximum(gold_arr, pred_arr))
    return 1.0 if union == 0 else float(intersection / union)


def compute_calibration(gold_labels, pred_labels, response_length):
    gold_probs = labels_to_char_probs(gold_labels, response_length)
    pred_probs = labels_to_char_probs(pred_labels, response_length)
    if np.std(gold_probs) == 0 and np.std(pred_probs) == 0:
        return 1.0
    if np.std(gold_probs) == 0 or np.std(pred_probs) == 0:
        return 0.0
    corr, _ = pearsonr(gold_probs, pred_probs)
    return float(corr) if not np.isnan(corr) else 0.0


# ============================================================================
# Data & Image Loading
# ============================================================================

def load_data(filepath):
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} samples from {filepath.name}")
    return samples


def split_eval_data(samples, ratio=0.10, seed=42):
    rng = np.random.RandomState(seed)
    n_eval = max(1, int(len(samples) * ratio))
    indices = rng.choice(len(samples), size=n_eval, replace=False)
    eval_samples = [samples[i] for i in sorted(indices)]
    logger.info(f"Split: {n_eval} eval samples ({ratio*100:.0f}%)")
    return eval_samples


def find_image(image_name, base_dir):
    if not base_dir.exists():
        return None
    names_to_try = [image_name]
    decoded = urllib.parse.unquote(image_name)
    if decoded != image_name:
        names_to_try.append(decoded)
    for name in names_to_try:
        direct = base_dir / name
        if direct.exists():
            return direct
        try:
            matches = list(base_dir.glob(f"**/{name}"))
            if matches:
                return matches[0]
        except Exception:
            pass
    return None


# ============================================================================
# Output Parsing (adapted for BLIP's shorter outputs)
# ============================================================================

def clean_and_align(text):
    clean_chars, idx_map = [], []
    for i, c in enumerate(text):
        if c in ['*', '_', '`']:
            continue
        clean_chars.append(c)
        idx_map.append(i)
    return "".join(clean_chars), idx_map


def find_robust_span(span_text, response_text, used_positions):
    clean_resp, idx_map = clean_and_align(response_text)
    clean_span, _ = clean_and_align(span_text)
    words = clean_span.lower().split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(w) for w in words)
    search_start = 0
    while search_start < len(clean_resp):
        match = re.search(pattern, clean_resp.lower()[search_start:])
        if not match:
            break
        start_clean = search_start + match.start()
        end_clean = search_start + match.end() - 1
        if end_clean >= len(idx_map) or start_clean >= len(idx_map):
            break
        orig_start = idx_map[start_clean]
        orig_end = idx_map[end_clean] + 1
        pos_key = (orig_start, orig_end)
        if pos_key not in used_positions:
            return pos_key
        search_start = start_clean + 1
    return None


def resolve_labels(parsed_list, response_text):
    cleaned = []
    used_positions = set()
    for item in parsed_list:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "invention"))
        prob = 0.5
        try:
            prob = float(item.get("prob", item.get("confidence", 0.5)))
        except (ValueError, TypeError):
            pass
        prob = max(0.0, min(1.0, prob))

        if "text" in item and response_text:
            span_text = str(item["text"])
            if not span_text:
                continue
            pos_key = find_robust_span(span_text, response_text, used_positions)
            if pos_key is not None:
                used_positions.add(pos_key)
                cleaned.append({
                    "start": pos_key[0], "end": pos_key[1],
                    "label": label, "prob": prob,
                })
        elif "start" in item and "end" in item:
            try:
                start, end = int(item["start"]), int(item["end"])
                if 0 <= start < end:
                    cleaned.append({"start": start, "end": end, "label": label, "prob": prob})
            except (ValueError, TypeError):
                pass
    return cleaned


def parse_model_output(output_text, response_text=""):
    """Parse BLIP's output into structured labels.

    BLIP outputs can be:
    1. Valid JSON: [{"text": "blue", "label": "mischaracterization", "prob": 0.9}]
    2. Empty: [] or "no"
    3. Short text that we try to parse
    """
    import ast

    text = output_text.strip()

    # Handle common BLIP short answers
    if text.lower() in ("no", "none", "no errors", "no hallucinations", "correct", ""):
        return []

    # Remove markdown
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    parsed = None

    # Try JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        for attempt_str in [json_str, json_str.replace("'", '"')]:
            try:
                result = json.loads(attempt_str)
                if isinstance(result, list):
                    parsed = result
                    break
            except json.JSONDecodeError:
                pass
        if parsed is None:
            try:
                result = ast.literal_eval(json_str)
                if isinstance(result, list):
                    parsed = result
            except (ValueError, SyntaxError):
                pass

    # Try individual JSON objects
    if parsed is None:
        obj_matches = re.findall(r'\{[^{}]+\}', text)
        if obj_matches:
            parsed = []
            for obj_str in obj_matches:
                for s in [obj_str, obj_str.replace("'", '"')]:
                    try:
                        obj = json.loads(s)
                        if isinstance(obj, dict):
                            parsed.append(obj)
                            break
                    except json.JSONDecodeError:
                        pass

    if parsed is None:
        return []

    return resolve_labels(parsed, response_text)


# ============================================================================
# Model Loading & Inference
# ============================================================================

def load_model(model_id: str):
    """Load BLIP-VQA model for evaluation."""
    from transformers import BlipForQuestionAnswering, BlipProcessor

    logger.info(f"Loading model: {model_id}")

    model = BlipForQuestionAnswering.from_pretrained(model_id)
    processor = BlipProcessor.from_pretrained(model_id)

    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info(f"Model on GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Model on CPU")

    model.eval()
    logger.info("Model loaded successfully")
    return model, processor


def run_inference(model, processor, sample, max_new_tokens=128):
    """Run hallucination detection inference using BLIP-VQA."""
    image_name = sample.get("image_name", "")
    image = None
    if image_name:
        image_path = find_image(image_name, IMAGES_DIR)
        if image_path:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Error opening image {image_path}: {e}")

    if image is None:
        image = Image.new("RGB", (384, 384), (128, 128, 128))

    # Build question
    truncated_response = sample["response"][:300] if len(sample["response"]) > 300 else sample["response"]
    question = BLIP_QUESTION_TEMPLATE.format(response=truncated_response)

    try:
        # Process inputs
        inputs = processor(
            images=image,
            text=question,
            return_tensors="pt",
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        response = processor.decode(output_ids[0], skip_special_tokens=True)
        return response.strip()

    except Exception as e:
        logger.warning(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        return ""


# ============================================================================
# Checkpoint Management
# ============================================================================

def load_checkpoint(checkpoint_path):
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        logger.info(f"Resumed from checkpoint: {ckpt['processed']} samples")
        return ckpt
    return {"processed": 0, "predictions": [], "raw_outputs": []}


def save_checkpoint(checkpoint_path, ckpt):
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False)


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def evaluate(args):
    start_time = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "checkpoint.json"
    csv_path = OUTPUT_DIR / "predictions_en.csv"
    json_path = OUTPUT_DIR / "metrics_en.json"
    predictions_jsonl_path = OUTPUT_DIR / "predictions_en.jsonl"

    logger.info("=" * 60)
    logger.info("SHROOM-Visions Evaluation: BLIP-VQA-Base")
    logger.info("=" * 60)

    all_samples = load_data(TRAIN_FILE)
    eval_samples = split_eval_data(all_samples, ratio=EVAL_SPLIT_RATIO, seed=RANDOM_SEED)

    if not IMAGES_DIR.exists():
        logger.warning(f"Images folder {IMAGES_DIR} missing. Using blank images.")
    else:
        logger.info(f"Images folder found at {IMAGES_DIR}.")

    if args.max_samples and args.max_samples < len(eval_samples):
        eval_samples = eval_samples[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")

    model, processor = load_model(args.model_id)

    if args.resume:
        ckpt = load_checkpoint(checkpoint_path)
    else:
        ckpt = {"processed": 0, "predictions": [], "raw_outputs": []}

    logger.info(f"\nRunning inference on {len(eval_samples)} samples...")

    for idx in tqdm(range(ckpt["processed"], len(eval_samples)),
                    desc="Evaluating", initial=ckpt["processed"],
                    total=len(eval_samples)):
        sample = eval_samples[idx]

        try:
            raw_output = run_inference(
                model, processor, sample,
                max_new_tokens=args.max_new_tokens,
            )
            pred_labels = parse_model_output(raw_output, sample["response"])
            tqdm.write(f"Sample {sample['id']} - {len(pred_labels)} spans | Output: {raw_output!r:.200}")
        except Exception as e:
            tqdm.write(f"Error on sample {sample['id']}: {e}")
            raw_output = ""
            pred_labels = []

        ckpt["predictions"].append({
            "id": sample["id"],
            "pred_labels": pred_labels,
            "gold_labels": sample.get("labels", []),
            "response_length": len(sample["response"]),
            "response": sample["response"],
            "prompt": sample["prompt"],
            "image_name": sample.get("image_name", ""),
        })
        ckpt["raw_outputs"].append({
            "id": sample["id"],
            "raw_model_output": raw_output,
        })
        ckpt["processed"] = idx + 1

        if (idx + 1) % 25 == 0:
            save_checkpoint(checkpoint_path, ckpt)

    save_checkpoint(checkpoint_path, ckpt)

    # ── Compute metrics ──
    logger.info("\nComputing Metrics...")

    per_sample_metrics = []
    category_metrics = {
        "invention": {"iou_list": [], "cal_list": []},
        "mischaracterization": {"iou_list": [], "cal_list": []},
        "OCR": {"iou_list": [], "cal_list": []},
        "miscounting": {"iou_list": [], "cal_list": []},
    }

    for pred in ckpt["predictions"]:
        gold = pred["gold_labels"]
        predicted = pred["pred_labels"]
        resp_len = pred["response_length"]

        iou = compute_iou(gold, predicted, resp_len)
        cal = compute_calibration(gold, predicted, resp_len)

        per_sample_metrics.append({
            "id": pred["id"], "iou": iou, "calibration": cal,
            "gold_span_count": len(gold), "pred_span_count": len(predicted),
            "has_gold_hallucination": len(gold) > 0,
            "has_pred_hallucination": len(predicted) > 0,
            "response_length": resp_len,
        })

        if gold:
            for cat in set(l["label"] for l in gold):
                cat_gold = [l for l in gold if l["label"] == cat]
                cat_pred = [l for l in predicted if l["label"] == cat]
                if cat in category_metrics:
                    category_metrics[cat]["iou_list"].append(
                        compute_iou(cat_gold, cat_pred, resp_len)
                    )
                    c = compute_calibration(cat_gold, cat_pred, resp_len)
                    if c is not None:
                        category_metrics[cat]["cal_list"].append(c)

    # ── Aggregate ──
    iou_scores = [m["iou"] for m in per_sample_metrics]
    cal_scores = [m["calibration"] for m in per_sample_metrics if m["calibration"] is not None]

    n_total = len(per_sample_metrics)
    n_gold_halluc = sum(1 for m in per_sample_metrics if m["has_gold_hallucination"])
    n_pred_halluc = sum(1 for m in per_sample_metrics if m["has_pred_hallucination"])
    n_correct_clean = sum(1 for m in per_sample_metrics
                          if not m["has_gold_hallucination"] and not m["has_pred_hallucination"])
    n_correct_halluc = sum(1 for m in per_sample_metrics
                           if m["has_gold_hallucination"] and m["has_pred_hallucination"])

    overall_results = {
        "model": args.model_id,
        "model_family": "BLIP-VQA-Base",
        "language": "en",
        "eval_samples": n_total,
        "metrics": {
            "overall": {
                "iou_mean": float(np.mean(iou_scores)) if iou_scores else 0.0,
                "iou_std": float(np.std(iou_scores)) if iou_scores else 0.0,
                "iou_median": float(np.median(iou_scores)) if iou_scores else 0.0,
                "calibration_mean": float(np.mean(cal_scores)) if cal_scores else 0.0,
                "calibration_std": float(np.std(cal_scores)) if cal_scores else 0.0,
            },
            "detection_stats": {
                "gold_has_hallucination": n_gold_halluc,
                "pred_has_hallucination": n_pred_halluc,
                "correct_clean": n_correct_clean,
                "correct_halluc": n_correct_halluc,
                "detection_accuracy": (n_correct_clean + n_correct_halluc) / n_total if n_total > 0 else 0.0,
            },
            "per_category": {},
        },
        "timing": {
            "total_seconds": round(time.time() - start_time, 2),
            "samples_per_second": round(n_total / max(time.time() - start_time, 0.01), 4),
        },
    }

    for cat, data in category_metrics.items():
        if data["iou_list"]:
            overall_results["metrics"]["per_category"][cat] = {
                "sample_count": len(data["iou_list"]),
                "iou_mean": float(np.mean(data["iou_list"])),
            }

    # ── Save outputs ──
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "prompt", "image_name", "response_length",
            "gold_span_count", "pred_span_count",
            "gold_labels_json", "pred_labels_json",
            "iou", "calibration",
        ])
        for pred, metrics in zip(ckpt["predictions"], per_sample_metrics):
            writer.writerow([
                pred["id"], pred["prompt"], pred["image_name"],
                pred["response_length"], metrics["gold_span_count"],
                metrics["pred_span_count"],
                json.dumps(pred["gold_labels"], ensure_ascii=False),
                json.dumps(pred["pred_labels"], ensure_ascii=False),
                f"{metrics['iou']:.6f}",
                f"{metrics['calibration']:.6f}" if metrics["calibration"] is not None else "",
            ])

    with open(predictions_jsonl_path, "w", encoding="utf-8") as f:
        for pred in ckpt["predictions"]:
            f.write(json.dumps({
                "id": pred["id"], "pred_labels": pred["pred_labels"],
                "gold_labels": pred["gold_labels"], "response": pred["response"],
            }, ensure_ascii=False) + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2, ensure_ascii=False)

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("  SHROOM-Visions Evaluation Summary — BLIP-VQA-Base")
    print(f"  Model: {args.model_id}")
    print(f"  Eval Samples: {n_total}")
    print(f"  Time: {overall_results['timing']['total_seconds']:.1f}s")
    print("=" * 70)

    print("\n📊 Overall Metrics:")
    print(tabulate([
        ["IoU (mean ± std)",
         f"{overall_results['metrics']['overall']['iou_mean']:.4f} ± "
         f"{overall_results['metrics']['overall']['iou_std']:.4f}"],
        ["Calibration (mean)",
         f"{overall_results['metrics']['overall']['calibration_mean']:.4f}"],
        ["Detection Accuracy",
         f"{overall_results['metrics']['detection_stats']['detection_accuracy']:.4f}"],
    ], headers=["Metric", "Value"], tablefmt="rounded_outline"))

    if overall_results["metrics"]["per_category"]:
        cat_table = [[cat, d["sample_count"], f"{d['iou_mean']:.4f}"]
                      for cat, d in overall_results["metrics"]["per_category"].items()]
        print("\n🏷️  Per-Category:")
        print(tabulate(cat_table, headers=["Category", "N", "IoU"], tablefmt="rounded_outline"))

    print(f"\n📁 Results: {OUTPUT_DIR}")
    print("=" * 70)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return overall_results


def main():
    parser = argparse.ArgumentParser(description="SHROOM-Visions Eval: BLIP-VQA-Base")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    if not TRAIN_FILE.exists():
        logger.error(f"Data file not found: {TRAIN_FILE}")
        sys.exit(1)

    evaluate(args)


if __name__ == "__main__":
    main()
