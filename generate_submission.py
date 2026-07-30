#!/usr/bin/env python3
"""
Generate submission-ready JSONL predictions for SHROOM-Visions competition.
Runs the trained SpanCalib-VLM tagger on unlabeled test files.

Usage:
    python generate_submission.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from spancalib_vlm.dataset import find_image
from spancalib_vlm.model import SpanCalibVLM
from spancalib_vlm.evaluate import reconstruct_spans_from_tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ──
DATA_DIR = Path("shroom-visions-data/distrib")
IMAGES_DIR = Path("shroom-vis-images")
CHECKPOINT_DIR = Path("checkpoints/spancalib_vlm_multilingual")
OUTPUT_DIR = Path("submission")
MODEL_ID = "xlm-roberta-large"
VISION_MODEL_ID = "google/siglip-base-patch16-224"
LANGUAGES = ["en", "fr", "it", "zh"]

# Per-language optimal thresholds from grid search on val data
THRESHOLDS = {
    "en": 0.35,
    "fr": 0.35,
    "it": 0.35,
    "zh": 0.60,
}


def load_test_data(filepath: Path) -> list[dict]:
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def predict_single(
    sample: dict,
    model,
    tokenizer,
    image_processor,
    device: torch.device,
    threshold: float,
) -> dict:
    """Generate prediction for a single test sample."""
    prompt_text = sample.get("prompt", "")
    response_text = sample.get("response", "")

    if not response_text:
        return {"id": sample["id"], "labels": []}

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

    # Load image
    pixel_values = None
    if image_processor is not None:
        img_name = sample.get("image_name", "")
        img_path = find_image(img_name, IMAGES_DIR)
        if img_path and img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (255, 255, 255))
        else:
            img = Image.new("RGB", (224, 224), (255, 255, 255))
        pixel_values = image_processor(images=img, return_tensors="pt")["pixel_values"].to(device)

    # Compute response offset in formatted prompt
    resp_start = formatted_prompt.find(response_text)
    if resp_start == -1:
        resp_start = 0

    adjusted_offsets = []
    for s, e in offsets:
        if s >= resp_start and e > resp_start:
            adjusted_offsets.append((s - resp_start, e - resp_start))
        else:
            adjusted_offsets.append((0, 0))

    # Run inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        pred_probs = outputs["pred_probs"].squeeze(0).cpu().numpy()
        cat_logits = outputs["cat_logits"].squeeze(0).cpu().numpy()
        token_cats = cat_logits.argmax(axis=-1)

    # Reconstruct character-level spans
    pred_labels = reconstruct_spans_from_tokens(
        token_probs=pred_probs,
        token_cats=token_cats,
        offsets=adjusted_offsets,
        response_text=response_text,
        threshold=threshold,
    )

    return {"id": sample["id"], "labels": pred_labels}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Load image processor
    from transformers import SiglipImageProcessor
    image_processor = SiglipImageProcessor.from_pretrained(VISION_MODEL_ID)

    # Load model with trained weights
    model = SpanCalibVLM(model_id=MODEL_ID, use_vision=True)

    checkpoint_file = CHECKPOINT_DIR / "best_model.pt"
    if not checkpoint_file.exists():
        checkpoint_file = CHECKPOINT_DIR / "final_model.pt"

    if checkpoint_file.exists():
        logger.info(f"Loading checkpoint: {checkpoint_file}")
        model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    else:
        logger.error(f"No checkpoint found at {CHECKPOINT_DIR}")
        sys.exit(1)

    model.to(device)
    model.eval()

    # Generate predictions for each language
    total_predictions = 0

    for lang in LANGUAGES:
        test_file = DATA_DIR / f"shroom-vision.test.{lang}.unlabeled.jsonl"
        output_file = OUTPUT_DIR / f"{lang}.jsonl"
        threshold = THRESHOLDS.get(lang, 0.35)

        if not test_file.exists():
            logger.warning(f"Test file not found: {test_file}, skipping {lang}")
            continue

        samples = load_test_data(test_file)
        logger.info(f"[{lang.upper()}] {len(samples)} test samples, threshold={threshold}")

        predictions = []
        for sample in tqdm(samples, desc=f"Predicting {lang.upper()}"):
            pred = predict_single(sample, model, tokenizer, image_processor, device, threshold)
            predictions.append(pred)

        # Write submission JSONL
        with open(output_file, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

        n_with_spans = sum(1 for p in predictions if len(p["labels"]) > 0)
        logger.info(f"[{lang.upper()}] Wrote {len(predictions)} predictions to {output_file}")
        logger.info(f"[{lang.upper()}] {n_with_spans}/{len(predictions)} samples have hallucination spans")
        total_predictions += len(predictions)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Total: {total_predictions} predictions across {len(LANGUAGES)} languages")
    logger.info(f"  Submission files: {OUTPUT_DIR}/")
    logger.info(f"{'='*60}")

    # Run format checker if available
    format_checker = Path("participant_kit/participant_kit/format_checker.py")
    if format_checker.exists():
        import subprocess
        submission_files = [str(OUTPUT_DIR / f"{lang}.jsonl") for lang in LANGUAGES if (OUTPUT_DIR / f"{lang}.jsonl").exists()]
        if submission_files:
            logger.info("\nRunning format checker...")
            result = subprocess.run(
                ["python", str(format_checker)] + submission_files,
                capture_output=True, text=True,
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            if result.returncode == 0:
                logger.info("✓ Format check PASSED!")
            else:
                logger.warning("✗ Format check had issues — review output above")
    else:
        logger.info("Format checker not found, skipping validation")


if __name__ == "__main__":
    main()
