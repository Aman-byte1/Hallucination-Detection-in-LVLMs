#!/usr/bin/env python3
"""
SHROOM-Visions Hallucination Detection Evaluation — MiniCPM-V-2
=================================================================
Evaluation script for MiniCPM-V-2 (base or finetuned).
Uses the same metrics (IoU + calibration) as evaluate.py for fair comparison.

Usage:
    python evaluate_minicpm.py                          # Evaluate base model
    python evaluate_minicpm.py --model_id ./checkpoints/minicpm-v2-shroom-sft/final
    python evaluate_minicpm.py --max_samples 10         # Quick test
    python evaluate_minicpm.py --resume                 # Resume from checkpoint
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
from scipy.stats import pearsonr
from tabulate import tabulate
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL_ID = "openbmb/MiniCPM-V-2"
DATA_DIR = Path(__file__).parent / "shroom-visions-data" / "distrib"
IMAGES_DIR = Path(__file__).parent / "shroom-vis-images"
OUTPUT_DIR = Path(__file__).parent / "outputs_minicpm"
TRAIN_FILE = DATA_DIR / "shroom-vision.train.en.labeled.jsonl"
EVAL_SPLIT_RATIO = 0.10
RANDOM_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Metrics (same as evaluate.py for fair comparison)
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


def compute_iou(gold_labels, pred_labels, response_length):
    gold_arr = labels_to_char_binary(gold_labels, response_length)
    pred_arr = labels_to_char_binary(pred_labels, response_length)
    intersection = np.sum(gold_arr * pred_arr)
    union = np.sum(np.maximum(gold_arr, pred_arr))
    if union == 0:
        return 1.0
    return float(intersection / union)


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
# System Prompt & Prompt Building (same as evaluate.py)
# ============================================================================

SYSTEM_PROMPT = """You are a hallucination detector. Check if text responses about images contain factual errors.

RULES:
- Output ONLY a JSON array: [{"text": "<wrong>", "label": "<type>", "prob": 0.9}] or []
- Quote 1-3 words MAXIMUM - the specific wrong part
- Correct labels: invention, mischaracterization, OCR, miscounting
- Be AGGRESSIVE - flag anything that seems wrong
- If the text says something exists and it doesn't, that's invention
- If the text says something is X color/shape/size but it's different, that's mischaracterization
- If the text says wrong number, that's miscounting
- If the text misquotes text from image, that's OCR
- Do NOT flag opinions ("beautiful", "nice") or hedging ("appears", "seems")

EXAMPLES (image → text → output):
- Image has red car, text says "blue car" → [{"text": "blue", "label": "mischaracterization", "prob": 0.9}]
- Image has 3 dogs, text says "five dogs" → [{"text": "five", "label": "miscounting", "prob": 0.9}]
- Image has no cat, text says "the cat sits" → [{"text": "cat", "label": "invention", "prob": 0.9}]
- Image has cat with white fur, text says "black fur" → [{"text": "black", "label": "mischaracterization", "prob": 0.9}]
- Image shows sign "STOP", text says "sign says GO" → [{"text": "GO", "label": "OCR", "prob": 0.9}]
- Image has 2 birds, text says "birds" (no count) → []
- Image has mushroom with cap, text says "has a cap" → []

Output:"""


def build_user_prompt(sample: dict) -> str:
    return (
        f"IMAGE QUESTION: {sample['prompt']}\n\n"
        f"RESPONSE TO CHECK: {sample['response']}\n\n"
        f"Find factual errors. Flag anything that seems wrong.\n"
        f"Quote 1-3 wrong words only. Output JSON array or []."
    )


# ============================================================================
# Data Loading & Splitting (same split logic as evaluate.py)
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
# Parsing (reused from evaluate.py)
# ============================================================================

def clean_and_align(text):
    clean_chars = []
    idx_map = []
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
    import ast
    text = output_text.strip()
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    parsed = None
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            result = json.loads(json_str)
            if isinstance(result, list):
                parsed = result
        except json.JSONDecodeError:
            pass
        if parsed is None:
            try:
                result = ast.literal_eval(json_str)
                if isinstance(result, list):
                    parsed = result
            except (ValueError, SyntaxError):
                pass
        if parsed is None:
            try:
                result = json.loads(json_str.replace("'", '"'))
                if isinstance(result, list):
                    parsed = result
            except json.JSONDecodeError:
                pass

    if parsed is None:
        obj_matches = re.findall(r'\{[^{}]+\}', text)
        if obj_matches:
            parsed = []
            for obj_str in obj_matches:
                try:
                    obj = json.loads(obj_str)
                    if isinstance(obj, dict):
                        parsed.append(obj)
                except json.JSONDecodeError:
                    try:
                        obj = json.loads(obj_str.replace("'", '"'))
                        if isinstance(obj, dict):
                            parsed.append(obj)
                    except json.JSONDecodeError:
                        pass

    if parsed is None:
        return []
    return resolve_labels(parsed, response_text)


# ============================================================================
# Model Loading & Inference
# ============================================================================

def load_model(model_id: str):
    """Load MiniCPM-V-2 for evaluation."""
    from transformers import AutoModel, AutoTokenizer

    logger.info(f"Loading model: {model_id}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
    logger.info(f"Using dtype: {dtype}")

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded successfully")
    return model, tokenizer


def run_inference(model, tokenizer, sample, max_new_tokens=512):
    """Run hallucination detection inference using MiniCPM-V-2's chat API."""
    from PIL import Image

    image_name = sample.get("image_name", "")
    image = None
    if image_name:
        image_path = find_image(image_name, IMAGES_DIR)
        if image_path:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Error opening image {image_path}: {e}")

    # Build message for MiniCPM-V-2
    user_prompt = f"{SYSTEM_PROMPT}\n{build_user_prompt(sample)}"

    msgs = [{"role": "user", "content": user_prompt}]

    try:
        # MiniCPM-V-2 uses model.chat() for inference
        response = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=False,  # Greedy decoding
            max_new_tokens=max_new_tokens,
        )

        if isinstance(response, tuple):
            response = response[0]

        return str(response).strip()

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
        logger.info(f"Resumed from checkpoint: {ckpt['processed']} samples processed")
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
    logger.info("SHROOM-Visions Evaluation: MiniCPM-V-2")
    logger.info("=" * 60)

    all_samples = load_data(TRAIN_FILE)
    eval_samples = split_eval_data(all_samples, ratio=EVAL_SPLIT_RATIO, seed=RANDOM_SEED)

    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.iterdir()):
        logger.warning(f"Images folder {IMAGES_DIR} is missing. Running TEXT-ONLY.")
    else:
        logger.info(f"Images folder found at {IMAGES_DIR}.")

    if args.max_samples and args.max_samples < len(eval_samples):
        eval_samples = eval_samples[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")

    model, tokenizer = load_model(args.model_id)

    if args.resume:
        ckpt = load_checkpoint(checkpoint_path)
    else:
        ckpt = {"processed": 0, "predictions": [], "raw_outputs": []}

    logger.info(f"\nRunning inference on {len(eval_samples)} samples...")
    logger.info(f"Starting from sample {ckpt['processed']}")

    for idx in tqdm(range(ckpt["processed"], len(eval_samples)),
                    desc="Evaluating", initial=ckpt["processed"],
                    total=len(eval_samples)):
        sample = eval_samples[idx]

        try:
            raw_output = run_inference(
                model, tokenizer, sample,
                max_new_tokens=args.max_new_tokens,
            )
            pred_labels = parse_model_output(raw_output, sample["response"])
            display = raw_output[:300] if len(raw_output) > 300 else raw_output
            tqdm.write(f"Sample {sample['id']} - {len(pred_labels)} spans | Output: {display!r}")
        except Exception as e:
            tqdm.write(f"Error on sample {sample['id']}: {e}")
            import traceback
            traceback.print_exc()
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
            logger.info(f"Checkpoint saved at sample {idx + 1}")

    save_checkpoint(checkpoint_path, ckpt)

    # ── Compute metrics ──
    logger.info("\n" + "=" * 60)
    logger.info("Computing Metrics")
    logger.info("=" * 60)

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

        has_gold = len(gold) > 0
        has_pred = len(predicted) > 0

        per_sample_metrics.append({
            "id": pred["id"], "iou": iou, "calibration": cal,
            "gold_span_count": len(gold), "pred_span_count": len(predicted),
            "has_gold_hallucination": has_gold, "has_pred_hallucination": has_pred,
            "response_length": resp_len,
        })

        if gold:
            gold_cats = set(lbl["label"] for lbl in gold)
            for cat in gold_cats:
                cat_gold = [l for l in gold if l["label"] == cat]
                cat_pred = [l for l in predicted if l["label"] == cat]
                cat_iou = compute_iou(cat_gold, cat_pred, resp_len)
                cat_cal = compute_calibration(cat_gold, cat_pred, resp_len)
                if cat in category_metrics:
                    category_metrics[cat]["iou_list"].append(cat_iou)
                    if cat_cal is not None:
                        category_metrics[cat]["cal_list"].append(cat_cal)

    # ── Aggregate ──
    iou_scores = [m["iou"] for m in per_sample_metrics]
    cal_scores = [m["calibration"] for m in per_sample_metrics if m["calibration"] is not None]
    halluc_iou = [m["iou"] for m in per_sample_metrics if m["has_gold_hallucination"]]
    halluc_cal = [m["calibration"] for m in per_sample_metrics
                  if m["has_gold_hallucination"] and m["calibration"] is not None]
    clean_iou = [m["iou"] for m in per_sample_metrics if not m["has_gold_hallucination"]]

    n_total = len(per_sample_metrics)
    n_gold_halluc = sum(1 for m in per_sample_metrics if m["has_gold_hallucination"])
    n_pred_halluc = sum(1 for m in per_sample_metrics if m["has_pred_hallucination"])
    n_correct_clean = sum(1 for m in per_sample_metrics
                          if not m["has_gold_hallucination"] and not m["has_pred_hallucination"])
    n_correct_halluc = sum(1 for m in per_sample_metrics
                           if m["has_gold_hallucination"] and m["has_pred_hallucination"])

    overall_results = {
        "model": args.model_id,
        "model_family": "MiniCPM-V-2",
        "language": "en",
        "eval_samples": n_total,
        "eval_split_ratio": EVAL_SPLIT_RATIO,
        "random_seed": RANDOM_SEED,
        "metrics": {
            "overall": {
                "iou_mean": float(np.mean(iou_scores)) if iou_scores else 0.0,
                "iou_std": float(np.std(iou_scores)) if iou_scores else 0.0,
                "iou_median": float(np.median(iou_scores)) if iou_scores else 0.0,
                "calibration_mean": float(np.mean(cal_scores)) if cal_scores else 0.0,
                "calibration_std": float(np.std(cal_scores)) if cal_scores else 0.0,
                "calibration_median": float(np.median(cal_scores)) if cal_scores else 0.0,
            },
            "hallucinated_samples": {
                "count": n_gold_halluc,
                "iou_mean": float(np.mean(halluc_iou)) if halluc_iou else 0.0,
                "calibration_mean": float(np.mean(halluc_cal)) if halluc_cal else 0.0,
            },
            "clean_samples": {
                "count": n_total - n_gold_halluc,
                "iou_mean": float(np.mean(clean_iou)) if clean_iou else 0.0,
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
            "samples_per_second": round(n_total / (time.time() - start_time), 4) if (time.time() - start_time) > 0 else 0,
        },
    }

    for cat, data in category_metrics.items():
        if data["iou_list"]:
            overall_results["metrics"]["per_category"][cat] = {
                "sample_count": len(data["iou_list"]),
                "iou_mean": float(np.mean(data["iou_list"])),
                "calibration_mean": float(np.mean(data["cal_list"])) if data["cal_list"] else None,
            }

    # ── Save outputs ──
    logger.info(f"\nSaving predictions CSV to {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "prompt", "image_name", "response_length",
            "gold_span_count", "pred_span_count",
            "gold_labels_json", "pred_labels_json",
            "iou", "calibration",
            "has_gold_hallucination", "has_pred_hallucination",
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
                metrics["has_gold_hallucination"], metrics["has_pred_hallucination"],
            ])

    logger.info(f"Saving predictions JSONL to {predictions_jsonl_path}")
    with open(predictions_jsonl_path, "w", encoding="utf-8") as f:
        for pred in ckpt["predictions"]:
            f.write(json.dumps({
                "id": pred["id"], "pred_labels": pred["pred_labels"],
                "gold_labels": pred["gold_labels"], "response": pred["response"],
                "prompt": pred["prompt"], "image_name": pred["image_name"],
            }, ensure_ascii=False) + "\n")

    logger.info(f"Saving metrics JSON to {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2, ensure_ascii=False)

    # ── Print summary ──
    print("\n")
    print("=" * 70)
    print("  SHROOM-Visions Evaluation Summary — MiniCPM-V-2")
    print(f"  Model: {args.model_id}")
    print(f"  Language: English | Eval Samples: {n_total}")
    print(f"  Time: {overall_results['timing']['total_seconds']:.1f}s "
          f"({overall_results['timing']['samples_per_second']:.2f} samples/s)")
    print("=" * 70)

    overall_table = [
        ["Overall IoU (mean ± std)",
         f"{overall_results['metrics']['overall']['iou_mean']:.4f} ± "
         f"{overall_results['metrics']['overall']['iou_std']:.4f}"],
        ["Overall IoU (median)",
         f"{overall_results['metrics']['overall']['iou_median']:.4f}"],
        ["Overall Calibration (mean ± std)",
         f"{overall_results['metrics']['overall']['calibration_mean']:.4f} ± "
         f"{overall_results['metrics']['overall']['calibration_std']:.4f}"],
    ]
    print("\n📊 Overall Metrics:")
    print(tabulate(overall_table, headers=["Metric", "Value"], tablefmt="rounded_outline"))

    detect_table = [
        ["Gold has hallucination", n_gold_halluc],
        ["Predicted has hallucination", n_pred_halluc],
        ["Detection Accuracy",
         f"{overall_results['metrics']['detection_stats']['detection_accuracy']:.4f}"],
    ]
    print("\n🎯 Detection Statistics:")
    print(tabulate(detect_table, headers=["Stat", "Value"], tablefmt="rounded_outline"))

    if overall_results["metrics"]["per_category"]:
        cat_table = []
        for cat, data in overall_results["metrics"]["per_category"].items():
            cat_table.append([cat, data["sample_count"], f"{data['iou_mean']:.4f}"])
        print("\n🏷️  Per-Category Metrics:")
        print(tabulate(cat_table, headers=["Category", "Samples", "IoU"], tablefmt="rounded_outline"))

    print("\n" + "=" * 70)
    print(f"  📁 Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint removed (evaluation complete)")

    return overall_results


def main():
    parser = argparse.ArgumentParser(
        description="SHROOM-Visions Evaluation: MiniCPM-V-2"
    )
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if not TRAIN_FILE.exists():
        logger.error(f"Data file not found: {TRAIN_FILE}")
        sys.exit(1)

    evaluate(args)


if __name__ == "__main__":
    main()
