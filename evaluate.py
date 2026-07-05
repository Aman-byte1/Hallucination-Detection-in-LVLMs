#!/usr/bin/env python3
"""
SHROOM-Visions Hallucination Detection Evaluation
===================================================
Single-script evaluation pipeline using Qwen3-VL-2B-Thinking model.

Evaluates hallucination span detection on a held-out 10% of the English
training data, computes IoU (span identification) and Pearson correlation
(confidence calibration), and outputs:
  - CSV file with per-sample predictions
  - JSON file with aggregated metric results

Usage:
    python evaluate.py                          # Run full evaluation
    python evaluate.py --max_samples 10         # Quick test with 10 samples
    python evaluate.py --resume                 # Resume from checkpoint
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from tabulate import tabulate
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Thinking"
DATA_DIR = Path(__file__).parent / "shroom-visions-data" / "distrib"
IMAGES_DIR = Path(__file__).parent / "shroom-vis-images"
OUTPUT_DIR = Path(__file__).parent / "outputs"
TRAIN_FILE = DATA_DIR / "shroom-vision.train.en.labeled.jsonl"
EVAL_SPLIT_RATIO = 0.10  # Use 10% of training data for evaluation
RANDOM_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Metrics: IoU (Span Identification) & Calibration (Pearson Correlation)
# ============================================================================

def labels_to_char_binary(labels: list[dict], response_length: int) -> np.ndarray:
    """Convert span labels to a binary character-level array.

    Each position is 1 if ANY label spans that character, else 0.
    """
    arr = np.zeros(response_length, dtype=np.float64)
    for label in labels:
        start = max(0, label["start"])
        end = min(response_length, label["end"])
        arr[start:end] = 1.0
    return arr


def labels_to_char_probs(labels: list[dict], response_length: int) -> np.ndarray:
    """Convert span labels to a character-level probability array.

    For overlapping spans, takes the maximum probability at each position.
    The `prob` field represents the empirical annotator agreement probability.
    """
    arr = np.zeros(response_length, dtype=np.float64)
    for label in labels:
        start = max(0, label["start"])
        end = min(response_length, label["end"])
        prob = label.get("prob", 1.0)
        arr[start:end] = np.maximum(arr[start:end], prob)
    return arr


def compute_iou(gold_labels: list[dict], pred_labels: list[dict],
                response_length: int) -> float:
    """Compute character-level Intersection-over-Union (IoU).

    If both gold and pred are empty, IoU = 1.0 (perfect agreement on
    "no hallucination").
    """
    gold_arr = labels_to_char_binary(gold_labels, response_length)
    pred_arr = labels_to_char_binary(pred_labels, response_length)

    intersection = np.sum(gold_arr * pred_arr)
    union = np.sum(np.maximum(gold_arr, pred_arr))

    if union == 0:
        return 1.0  # Both are empty → perfect agreement
    return float(intersection / union)


def compute_calibration(gold_labels: list[dict], pred_labels: list[dict],
                        response_length: int) -> float | None:
    """Compute Pearson correlation between gold and predicted char-level probs.

    Returns None if correlation cannot be computed (e.g., zero variance).
    """
    gold_probs = labels_to_char_probs(gold_labels, response_length)
    pred_probs = labels_to_char_probs(pred_labels, response_length)

    # Pearson requires variance in both arrays
    if np.std(gold_probs) == 0 and np.std(pred_probs) == 0:
        return 1.0  # Both constant → perfect calibration
    if np.std(gold_probs) == 0 or np.std(pred_probs) == 0:
        return 0.0  # One constant, one not → zero correlation

    corr, _ = pearsonr(gold_probs, pred_probs)
    return float(corr) if not np.isnan(corr) else 0.0


# ============================================================================
# Data Loading & Splitting
# ============================================================================

def load_data(filepath: Path) -> list[dict]:
    """Load JSONL data file."""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} samples from {filepath.name}")
    return samples


def split_eval_data(samples: list[dict], ratio: float = 0.10,
                    seed: int = 42) -> list[dict]:
    """Extract a random 10% subset for evaluation."""
    rng = np.random.RandomState(seed)
    n_eval = max(1, int(len(samples) * ratio))
    indices = rng.choice(len(samples), size=n_eval, replace=False)
    eval_samples = [samples[i] for i in sorted(indices)]
    logger.info(
        f"Split: {n_eval} eval samples ({ratio*100:.0f}%) from "
        f"{len(samples)} total"
    )
    return eval_samples


# ============================================================================
SYSTEM_PROMPT = """You are a hallucination detector for image descriptions.

Task: Given an image and a response about it, find text that is factually WRONG.
A hallucination is ONLY:
- invention: something not in the image at all
- mischaracterization: wrong color, shape, size, or material
- OCR: incorrectly read text
- miscounting: wrong count of objects

STRICT RULES:
- Output ONLY a JSON array. Nothing else.
- Be CONSERVATIVE. If unsure, output []
- Do NOT flag opinions, hedging, or reasonable interpretations
- Do NOT flag things that could be correct
- Quote ONLY the exact wrong word(s), maximum 5 words. NOT full sentences.

GOOD: [{"text": "red", "label": "mischaracterization", "prob": 0.9}]
GOOD: [{"text": "four wheels", "label": "mischaracterization", "prob": 0.9}]
BAD (too long): [{"text": "This mushroom does not have a traditional cap like many people picture", "label": "mischaracterization", "prob": 0.9}]

Output: """


def build_user_prompt(sample: dict) -> str:
    """Build the user message for hallucination detection."""
    prompt = sample["prompt"]
    response = sample["response"]

    return (
        f"Image question: {prompt}\n"
        f"Response: {response}\n\n"
        f"Find factually wrong text in the response based on the image. "
        f"Output wrong words as JSON or [] if correct."
    )


def parse_model_output(output_text: str, response_text: str = "") -> list[dict]:
    """Parse the model's JSON output into a list of label dicts.

    Handles common formatting issues: thinking tags, markdown code blocks, etc.
    The new prompt format returns 'text' fields instead of start/end indices,
    so we compute character positions from quoted text matches against the
    original response.
    """
    import ast
    text = output_text.strip()

    # Remove markdown code block wrappers
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    parsed = None

    # Try to find a JSON array in the text
    # Look for the outermost [...] pattern
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        json_str = match.group(0)

        # 1. Try standard json parsing
        try:
            result = json.loads(json_str)
            if isinstance(result, list):
                parsed = result
        except json.JSONDecodeError:
            pass

        # 2. Try ast.literal_eval for single quotes/python-like lists
        if parsed is None:
            try:
                result = ast.literal_eval(json_str)
                if isinstance(result, list):
                    parsed = result
            except (ValueError, SyntaxError):
                pass

        # 3. Try replacing single quotes with double quotes
        if parsed is None:
            try:
                fixed_str = json_str.replace("'", '"')
                result = json.loads(fixed_str)
                if isinstance(result, list):
                    parsed = result
            except json.JSONDecodeError:
                pass

    if parsed is None:
        # 4. Try to extract individual JSON objects even without array brackets
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

    # Fallback: if output is empty but thinking has answers
    if (not parsed or parsed == []) and '<think>' in output_text:
        think_part = output_text.split('</think>', 1)[0] if '</think>' in output_text else output_text
        think_part = think_part.replace('<think>', "")
        result = _extract_labels_from_thinking(think_part, response_text)
        if result:
            return result
    if parsed is None:
        return []

    return resolve_labels(parsed, response_text)


def clean_and_align(text: str) -> tuple[str, list[int]]:
    """Clean markdown formatting characters and track index alignment."""
    clean_chars = []
    idx_map = []
    for i, c in enumerate(text):
        if c in ['*', '_', '`']:
            continue
        clean_chars.append(c)
        idx_map.append(i)
    return "".join(clean_chars), idx_map


def find_robust_span(span_text: str, response_text: str, used_positions: set) -> tuple[int, int] | None:
    """Find a span's start/end index in original response text, ignoring markdown."""
    clean_resp, idx_map = clean_and_align(response_text)
    clean_span, _ = clean_and_align(span_text)

    # Split span into words to allow variable/flexible spacing/whitespace
    words = clean_span.lower().split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(w) for w in words)

    # Search in cleaned response text
    search_start = 0
    while search_start < len(clean_resp):
        match = re.search(pattern, clean_resp.lower()[search_start:])
        if not match:
            break

        start_clean = search_start + match.start()
        end_clean = search_start + match.end() - 1

        # Map back to original indices
        orig_start = idx_map[start_clean]
        orig_end = idx_map[end_clean] + 1

        pos_key = (orig_start, orig_end)
        if pos_key not in used_positions:
            return pos_key

        search_start = start_clean + 1

    return None


def resolve_labels(parsed_list: list, response_text: str) -> list[dict]:
    """Convert parsed model output to normalized labels with character indices.

    Handles two formats:
    1. New format: {"text": "...", "label": "...", "prob": ...}
       -> We locate 'text' in response_text to compute start/end.
    2. Legacy format: {"start": N, "end": M, "label": "...", "prob": ...}
       -> Used directly.
    """
    cleaned = []
    used_positions = set()  # Track used match positions to avoid duplicates

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

        # Case 1: Has 'text' field — resolve to start/end from response
        if "text" in item and response_text:
            span_text = str(item["text"])
            if not span_text:
                continue

            pos_key = find_robust_span(span_text, response_text, used_positions)
            if pos_key is not None:
                used_positions.add(pos_key)
                cleaned.append({
                    "start": pos_key[0],
                    "end": pos_key[1],
                    "label": label,
                    "prob": prob,
                })
            else:
                logger.debug(
                    f"Could not locate span text in response: "
                    f"{span_text!r:.60}"
                )

        # Case 2: Has 'start' and 'end' — legacy format
        elif "start" in item and "end" in item:
            try:
                start = int(item["start"])
                end = int(item["end"])
                if 0 <= start < end:
                    cleaned.append({
                        "start": start,
                        "end": end,
                        "label": label,
                        "prob": prob,
                    })
            except (ValueError, TypeError):
                pass

    return cleaned


# ============================================================================
# Model Inference
# ============================================================================

def find_image(image_name: str, base_dir: Path) -> Path | None:
    """Find the image path in the base_dir, recursively if necessary."""
    import urllib.parse
    
    if not base_dir.exists():
        return None
    
    # Try with both the raw name and the URL-decoded name
    names_to_try = [image_name]
    decoded_name = urllib.parse.unquote(image_name)
    if decoded_name != image_name:
        names_to_try.append(decoded_name)
        
    for name in names_to_try:
        # Try direct check first
        direct_path = base_dir / name
        if direct_path.exists():
            return direct_path
        
        # Try recursive check if not found directly
        try:
            matches = list(base_dir.glob(f"**/{name}"))
            if matches:
                return matches[0]
        except Exception:
            pass
            
    return None


def load_model(model_id: str):
    """Load a Qwen VL model and processor."""
    from transformers import AutoProcessor, AutoModelForImageTextToText

    logger.info(f"Loading model: {model_id}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Determine dtype
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        logger.info("Using bfloat16")
    elif torch.cuda.is_available():
        dtype = torch.float16
        logger.info("Using float16")
    else:
        dtype = torch.float32
        logger.info("Using float32 (CPU)")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    logger.info("Model loaded successfully")
    return model, processor



def _extract_labels_from_thinking(thinking_text, response_text):
    """Extract hallucination labels from thinking chain as fallback."""
    extracted = []
    for m in re.finditer(r'\{[^{}]+\}', thinking_text):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "text" in obj and "label" in obj:
                extracted.append(obj)
        except json.JSONDecodeError:
            try:
                fixed = m.group(0).replace("'", '"')
                obj = json.loads(fixed)
                if isinstance(obj, dict) and "text" in obj and "label" in obj:
                    extracted.append(obj)
            except json.JSONDecodeError:
                pass
    if extracted:
        return resolve_labels(extracted, response_text)
    return []


def run_inference(model, processor, sample: dict, max_new_tokens: int = 2048) -> str:
    """Run hallucination detection inference on a single sample.

    Uses greedy decoding with the instruct model to produce JSON directly.
    """
    from PIL import Image

    image_name = sample.get("image_name", "")
    image_path = None
    if image_name:
        image_path = find_image(image_name, IMAGES_DIR)

    image = None
    if image_path is not None:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error opening image {image_path}: {e}")

    if image is not None:
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": build_user_prompt(sample)},
        ]
    else:
        user_content = [
            {"type": "text", "text": build_user_prompt(sample)},
        ]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    try:
        from qwen_vl_utils import process_vision_info
        vision_outputs = process_vision_info(messages)
        if len(vision_outputs) == 3:
            image_inputs, video_inputs, _ = vision_outputs
        else:
            image_inputs, video_inputs = vision_outputs
    except ImportError:
        image_inputs, video_inputs = None, None

    kwargs = {
        "text": [text],
        "padding": True,
        "return_tensors": "pt"
    }
    if image_inputs is not None:
        kwargs["images"] = image_inputs
    if video_inputs is not None:
        kwargs["videos"] = video_inputs

    inputs = processor(**kwargs)

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


# ============================================================================
# Checkpoint Management
# ============================================================================

def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        logger.info(
            f"Resumed from checkpoint: {ckpt['processed']} samples processed"
        )
        return ckpt
    return {"processed": 0, "predictions": [], "raw_outputs": []}


def save_checkpoint(checkpoint_path: Path, ckpt: dict):
    """Save checkpoint to disk."""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False)


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def evaluate(args):
    """Run the full evaluation pipeline."""
    start_time = time.time()

    # ── Setup output directory ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "checkpoint.json"
    csv_path = OUTPUT_DIR / "predictions_en.csv"
    json_path = OUTPUT_DIR / "metrics_en.json"
    predictions_jsonl_path = OUTPUT_DIR / "predictions_en.jsonl"

    # ── Load and split data ──
    logger.info("=" * 60)
    logger.info("SHROOM-Visions Evaluation Pipeline")
    logger.info("=" * 60)

    all_samples = load_data(TRAIN_FILE)
    eval_samples = split_eval_data(all_samples, ratio=EVAL_SPLIT_RATIO, seed=RANDOM_SEED)

    # ── Check images presence ──
    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.iterdir()):
        logger.warning(f"Images folder {IMAGES_DIR} is missing or empty. The evaluation will run in TEXT-ONLY fallback mode.")
    else:
        logger.info(f"Images folder found at {IMAGES_DIR}. Running in multimodal vision-language mode.")

    if args.max_samples and args.max_samples < len(eval_samples):
        eval_samples = eval_samples[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples (--max_samples)")

    # ── Load model ──
    model, processor = load_model(args.model_id)

    # ── Load checkpoint if resuming ──
    if args.resume:
        ckpt = load_checkpoint(checkpoint_path)
    else:
        ckpt = {"processed": 0, "predictions": [], "raw_outputs": []}

    # ── Inference loop ──
    logger.info(f"\nRunning inference on {len(eval_samples)} samples...")
    logger.info(f"Starting from sample {ckpt['processed']}")

    for idx in tqdm(range(ckpt["processed"], len(eval_samples)),
                    desc="Evaluating", initial=ckpt["processed"],
                    total=len(eval_samples)):
        sample = eval_samples[idx]

        try:
            raw_output = run_inference(
                model, processor, sample,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.think,
            )
            pred_labels = parse_model_output(raw_output, sample["response"])
            # Log raw output — show head and tail to see if JSON is at the end
            clean_output = raw_output
            clean_output = clean_output.strip()
            if len(clean_output) > 300:
                display = clean_output[:150] + " ... " + clean_output[-150:]
            else:
                display = clean_output
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

        # Save checkpoint every 25 samples
        if (idx + 1) % 25 == 0:
            save_checkpoint(checkpoint_path, ckpt)
            logger.info(f"Checkpoint saved at sample {idx + 1}")

    # Final checkpoint save
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

        has_gold_halluc = len(gold) > 0
        has_pred_halluc = len(predicted) > 0

        sample_metrics = {
            "id": pred["id"],
            "iou": iou,
            "calibration": cal,
            "gold_span_count": len(gold),
            "pred_span_count": len(predicted),
            "has_gold_hallucination": has_gold_halluc,
            "has_pred_hallucination": has_pred_halluc,
            "response_length": resp_len,
        }
        per_sample_metrics.append(sample_metrics)

        # Track per-category performance
        if gold:
            gold_categories = set(lbl["label"] for lbl in gold)
            for cat in gold_categories:
                cat_gold = [l for l in gold if l["label"] == cat]
                cat_pred = [l for l in predicted if l["label"] == cat]
                cat_iou = compute_iou(cat_gold, cat_pred, resp_len)
                cat_cal = compute_calibration(cat_gold, cat_pred, resp_len)
                if cat in category_metrics:
                    category_metrics[cat]["iou_list"].append(cat_iou)
                    if cat_cal is not None:
                        category_metrics[cat]["cal_list"].append(cat_cal)

    # ── Aggregate metrics ──
    iou_scores = [m["iou"] for m in per_sample_metrics]
    cal_scores = [m["calibration"] for m in per_sample_metrics if m["calibration"] is not None]

    # Samples with hallucinations only
    halluc_iou = [m["iou"] for m in per_sample_metrics if m["has_gold_hallucination"]]
    halluc_cal = [m["calibration"] for m in per_sample_metrics
                  if m["has_gold_hallucination"] and m["calibration"] is not None]

    # No-hallucination samples
    clean_iou = [m["iou"] for m in per_sample_metrics if not m["has_gold_hallucination"]]

    # Detection stats
    n_total = len(per_sample_metrics)
    n_gold_halluc = sum(1 for m in per_sample_metrics if m["has_gold_hallucination"])
    n_pred_halluc = sum(1 for m in per_sample_metrics if m["has_pred_hallucination"])
    n_correct_clean = sum(
        1 for m in per_sample_metrics
        if not m["has_gold_hallucination"] and not m["has_pred_hallucination"]
    )
    n_correct_halluc = sum(
        1 for m in per_sample_metrics
        if m["has_gold_hallucination"] and m["has_pred_hallucination"]
    )

    overall_results = {
        "model": args.model_id,
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
                "iou_std": float(np.std(halluc_iou)) if halluc_iou else 0.0,
                "calibration_mean": float(np.mean(halluc_cal)) if halluc_cal else 0.0,
                "calibration_std": float(np.std(halluc_cal)) if halluc_cal else 0.0,
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

    # Per-category metrics
    for cat, data in category_metrics.items():
        if data["iou_list"]:
            overall_results["metrics"]["per_category"][cat] = {
                "sample_count": len(data["iou_list"]),
                "iou_mean": float(np.mean(data["iou_list"])),
                "iou_std": float(np.std(data["iou_list"])),
                "calibration_mean": float(np.mean(data["cal_list"])) if data["cal_list"] else None,
                "calibration_std": float(np.std(data["cal_list"])) if data["cal_list"] else None,
            }

    # ── Save outputs ──

    # 1. CSV with per-sample predictions
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
                pred["id"],
                pred["prompt"],
                pred["image_name"],
                pred["response_length"],
                metrics["gold_span_count"],
                metrics["pred_span_count"],
                json.dumps(pred["gold_labels"], ensure_ascii=False),
                json.dumps(pred["pred_labels"], ensure_ascii=False),
                f"{metrics['iou']:.6f}",
                f"{metrics['calibration']:.6f}" if metrics["calibration"] is not None else "",
                metrics["has_gold_hallucination"],
                metrics["has_pred_hallucination"],
            ])

    # 2. JSONL with full predictions (for potential submission / further analysis)
    logger.info(f"Saving predictions JSONL to {predictions_jsonl_path}")
    with open(predictions_jsonl_path, "w", encoding="utf-8") as f:
        for pred in ckpt["predictions"]:
            f.write(json.dumps({
                "id": pred["id"],
                "pred_labels": pred["pred_labels"],
                "gold_labels": pred["gold_labels"],
                "response": pred["response"],
                "prompt": pred["prompt"],
                "image_name": pred["image_name"],
            }, ensure_ascii=False) + "\n")

    # 3. JSON with aggregated metrics
    logger.info(f"Saving metrics JSON to {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2, ensure_ascii=False)

    # ── Print summary table ──
    print("\n")
    print("=" * 70)
    print("  SHROOM-Visions Evaluation Summary")
    print(f"  Model: {args.model_id}")
    print(f"  Language: English | Eval Samples: {n_total}")
    print(f"  Time: {overall_results['timing']['total_seconds']:.1f}s "
          f"({overall_results['timing']['samples_per_second']:.2f} samples/s)")
    print("=" * 70)

    # Overall metrics table
    overall_table = [
        ["Overall IoU (mean ± std)",
         f"{overall_results['metrics']['overall']['iou_mean']:.4f} ± "
         f"{overall_results['metrics']['overall']['iou_std']:.4f}"],
        ["Overall IoU (median)",
         f"{overall_results['metrics']['overall']['iou_median']:.4f}"],
        ["Overall Calibration (mean ± std)",
         f"{overall_results['metrics']['overall']['calibration_mean']:.4f} ± "
         f"{overall_results['metrics']['overall']['calibration_std']:.4f}"],
        ["Overall Calibration (median)",
         f"{overall_results['metrics']['overall']['calibration_median']:.4f}"],
    ]
    print("\n📊 Overall Metrics:")
    print(tabulate(overall_table, headers=["Metric", "Value"],
                   tablefmt="rounded_outline"))

    # Hallucinated vs clean breakdown
    breakdown_table = [
        ["With Hallucinations",
         n_gold_halluc,
         f"{overall_results['metrics']['hallucinated_samples']['iou_mean']:.4f}" if halluc_iou else "N/A",
         f"{overall_results['metrics']['hallucinated_samples']['calibration_mean']:.4f}" if halluc_cal else "N/A"],
        ["Clean (No Hallucination)",
         n_total - n_gold_halluc,
         f"{overall_results['metrics']['clean_samples']['iou_mean']:.4f}" if clean_iou else "N/A",
         "N/A"],
    ]
    print("\n📋 Breakdown by Hallucination Presence:")
    print(tabulate(breakdown_table,
                   headers=["Category", "Count", "IoU", "Calibration"],
                   tablefmt="rounded_outline"))

    # Detection accuracy
    detect_table = [
        ["Gold has hallucination", n_gold_halluc],
        ["Predicted has hallucination", n_pred_halluc],
        ["Correctly identified clean", n_correct_clean],
        ["Correctly identified hallucinated", n_correct_halluc],
        ["Detection Accuracy",
         f"{overall_results['metrics']['detection_stats']['detection_accuracy']:.4f}"],
    ]
    print("\n🎯 Detection Statistics:")
    print(tabulate(detect_table, headers=["Stat", "Value"],
                   tablefmt="rounded_outline"))

    # Per-category metrics
    if overall_results["metrics"]["per_category"]:
        cat_table = []
        for cat, data in overall_results["metrics"]["per_category"].items():
            cat_table.append([
                cat,
                data["sample_count"],
                f"{data['iou_mean']:.4f} ± {data['iou_std']:.4f}",
                f"{data['calibration_mean']:.4f}" if data["calibration_mean"] is not None else "N/A",
            ])
        print("\n🏷️  Per-Category Metrics:")
        print(tabulate(cat_table,
                       headers=["Category", "Samples", "IoU (mean ± std)", "Calibration"],
                       tablefmt="rounded_outline"))

    print("\n" + "=" * 70)
    print(f"  📁 Results saved to: {OUTPUT_DIR}")
    print(f"     ├── {csv_path.name}           (per-sample CSV)")
    print(f"     ├── {predictions_jsonl_path.name}  (full predictions JSONL)")
    print(f"     └── {json_path.name}            (aggregated metrics JSON)")
    print("=" * 70)

    # Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint removed (evaluation complete)")

    return overall_results


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SHROOM-Visions Hallucination Detection Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py                            # Full evaluation
  python evaluate.py --max_samples 10           # Quick test (10 samples)
  python evaluate.py --resume                   # Resume from checkpoint
  python evaluate.py --model_id Qwen/Qwen3-VL-2B-Thinking
        """,
    )
    parser.add_argument(
        "--model_id", type=str, default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=16384,
        help="Max tokens for model generation (default: 16384)",
    )
    parser.add_argument(
        "--no_think", action="store_true",
        help="Disable thinking mode (greedy output, no CoT reasoning chain).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint",
    )

    args = parser.parse_args()
    args.think = not args.no_think

    if args.think:
        logger.info("Thinking mode enabled.")
    else:
        logger.info("Thinking mode disabled (greedy output).")

    # Validate data file exists
    if not TRAIN_FILE.exists():
        logger.error(f"Data file not found: {TRAIN_FILE}")
        logger.error(
            "Make sure shroom-visions-data is extracted in the project root."
        )
        sys.exit(1)

    evaluate(args)


if __name__ == "__main__":
    main()
