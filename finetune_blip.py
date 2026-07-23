#!/usr/bin/env python3
"""
Finetune BLIP-VQA-Base on SHROOM-Visions Hallucination Detection
================================================================
Full fine-tune of Salesforce/blip-vqa-base (0.4B) for hallucination
span detection. The model learns to generate JSON arrays of
hallucinated spans when given an image + question about a VLM response.

BLIP-VQA-Base is an encoder-decoder VQA model. We fine-tune its
decoder to generate structured JSON output (spans are 1-3 words,
so the decoder length is manageable).

Usage:
    python finetune_blip.py --max_samples 10     # Quick test
    python finetune_blip.py --push_to_hub --hub_token hf_xxx
"""

import argparse
import json
import logging
import os
import sys
import urllib.parse
from pathlib import Path

# Bypass broken system torchaudio build in container if present
sys.modules["torchaudio"] = None

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    BlipForQuestionAnswering,
    BlipProcessor,
    TrainingArguments,
    Trainer,
)

# ============================================================================
# Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Simplified prompt for BLIP (shorter is better for encoder-decoder models)
BLIP_QUESTION_TEMPLATE = (
    "Does this response about the image contain factual errors? "
    "Respond with a JSON array of errors or []. "
    "Response to check: {response}"
)


# ============================================================================
# Data Utilities
# ============================================================================

def find_image(image_name: str, images_dir: Path) -> Path | None:
    """Find the image file, handling URL-encoded names and nested dirs."""
    if not images_dir.exists():
        return None

    names_to_try = [image_name]
    decoded = urllib.parse.unquote(image_name)
    if decoded != image_name:
        names_to_try.append(decoded)

    for name in names_to_try:
        direct = images_dir / name
        if direct.exists():
            return direct
        try:
            matches = list(images_dir.glob(f"**/{name}"))
            if matches:
                return matches[0]
        except Exception:
            pass
    return None


def labels_to_text_spans(labels: list[dict], response: str) -> list[dict]:
    """Convert character-level SHROOM labels to text span format."""
    spans = []
    for label in labels:
        start = max(0, label["start"])
        end = min(len(response), label["end"])
        text = response[start:end].strip()
        if not text:
            continue
        spans.append({
            "text": text,
            "label": label["label"],
            "prob": round(label["prob"], 2),
        })
    return spans


# ============================================================================
# BLIP VQA Dataset
# ============================================================================

class BLIPSHROOMDataset(TorchDataset):
    """Prepare SHROOM samples for BLIP-VQA-Base training.

    Converts hallucination detection into a VQA format:
    - Question: "Does this response contain factual errors? Response: {text}"
    - Answer (target): JSON array like [{"text":"blue","label":"mischaracterization","prob":0.9}]
    """

    def __init__(self, samples, processor, images_dir,
                 max_question_length=512, max_answer_length=128):
        self.samples = samples
        self.processor = processor
        self.images_dir = Path(images_dir)
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length

    def __len__(self):
        return len(self.samples)

    def _load_image(self, image_name):
        """Load image with fallback to blank."""
        if image_name:
            image_path = find_image(image_name, self.images_dir)
            if image_path:
                try:
                    return Image.open(image_path).convert("RGB")
                except Exception as e:
                    logger.debug(f"Error loading image: {e}")

        return Image.new("RGB", (384, 384), (128, 128, 128))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        response = sample["response"]
        labels_data = sample.get("labels", [])

        # Build target
        spans = labels_to_text_spans(labels_data, response)
        target = json.dumps(spans, ensure_ascii=False) if spans else "[]"

        # Build question — truncate response if too long for BLIP
        # BLIP works best with shorter inputs
        truncated_response = response[:300] if len(response) > 300 else response
        question = BLIP_QUESTION_TEMPLATE.format(response=truncated_response)

        # Load image
        image = self._load_image(sample.get("image_name", ""))

        # Process with BlipProcessor
        encoding = self.processor(
            images=image,
            text=question,
            padding="max_length",
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt",
        )

        # Process answer/labels
        labels = self.processor.tokenizer(
            target,
            padding="max_length",
            truncation=True,
            max_length=self.max_answer_length,
            return_tensors="pt",
        )

        # Replace pad tokens with -100 for loss masking
        label_ids = labels["input_ids"].clone()
        label_ids[label_ids == self.processor.tokenizer.pad_token_id] = -100

        # Squeeze batch dimension
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = label_ids.squeeze(0)

        return item


# ============================================================================
# Data Loading & Splitting
# ============================================================================

def load_and_split_data(
    data_file: str,
    images_dir: str,
    processor,
    eval_ratio: float = 0.10,
    seed: int = 42,
    max_samples: int | None = None,
    max_question_length: int = 512,
    max_answer_length: int = 128,
) -> tuple:
    """Load SHROOM data, convert to BLIP VQA format, split train/eval."""
    # Load raw JSONL
    raw_samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_samples.append(json.loads(line))

    logger.info(f"Loaded {len(raw_samples)} raw samples from {data_file}")

    if max_samples is not None:
        raw_samples = raw_samples[:max_samples]
        logger.info(f"Limited to {max_samples} samples")

    # Split — SAME seed/ratio as finetune.py
    rng = np.random.RandomState(seed)
    n_eval = max(1, int(len(raw_samples) * eval_ratio))
    eval_indices = set(rng.choice(len(raw_samples), size=n_eval, replace=False))

    train_samples = [s for i, s in enumerate(raw_samples) if i not in eval_indices]
    eval_samples = [s for i, s in enumerate(raw_samples) if i in eval_indices]

    logger.info(f"Split: {len(train_samples)} train + {len(eval_samples)} eval")

    # Check image availability
    images_path = Path(images_dir)
    n_with_image = 0
    for s in raw_samples[:100]:
        if s.get("image_name") and find_image(s["image_name"], images_path):
            n_with_image += 1
    logger.info(
        f"Image availability (first 100): {n_with_image}/100 "
        f"({'with images' if n_with_image > 50 else 'mostly text-only'})"
    )

    train_dataset = BLIPSHROOMDataset(
        train_samples, processor, images_dir,
        max_question_length, max_answer_length,
    )
    eval_dataset = BLIPSHROOMDataset(
        eval_samples, processor, images_dir,
        max_question_length, max_answer_length,
    )

    return train_dataset, eval_dataset


# ============================================================================
# Model Loading & Training
# ============================================================================

def load_model(model_id: str):
    """Load BLIP-VQA-Base for full fine-tuning."""
    logger.info(f"Loading model: {model_id}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU Memory: {mem_gb:.1f} GB")

    model = BlipForQuestionAnswering.from_pretrained(model_id)
    processor = BlipProcessor.from_pretrained(model_id)

    # Full fine-tune (BLIP-VQA-Base is only ~0.4B params)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Parameters: {total:,} total, {trainable:,} trainable "
        f"({100 * trainable / total:.1f}%)"
    )

    return model, processor


def train(model, processor, train_dataset, eval_dataset, args):
    """Run full fine-tuning with HF Trainer."""
    logger.info(
        f"Starting training: {len(train_dataset)} train, "
        f"{len(eval_dataset)} eval"
    )
    logger.info(
        f"Effective batch size: "
        f"{args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}"
    )

    steps_per_epoch = max(
        1, len(train_dataset) // (args.batch_size * args.grad_accum)
    )
    eval_steps = max(1, steps_per_epoch // 4)
    save_steps = eval_steps * 2

    logger.info(
        f"Steps/epoch: {steps_per_epoch}, eval every {eval_steps}, "
        f"save every {save_steps}"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        optim="adamw_torch",
        seed=args.seed,
        dataloader_pin_memory=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    resume = args.resume_from_checkpoint if args.resume_from_checkpoint else None
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")

    trainer.train(resume_from_checkpoint=resume)
    logger.info("Training complete!")
    return trainer


def save_and_upload(model, processor, args):
    """Save the finetuned model and optionally push to HuggingFace."""
    save_dir = os.path.join(args.output_dir, "final")
    logger.info(f"Saving model to {save_dir}")

    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    logger.info(f"Model saved to {save_dir}")

    if args.push_to_hub:
        logger.info(f"Pushing to HuggingFace: {args.hub_model_id}")
        model.push_to_hub(
            args.hub_model_id,
            token=args.hub_token,
            private=False,
        )
        processor.push_to_hub(
            args.hub_model_id,
            token=args.hub_token,
        )
        logger.info(f"✓ Model uploaded to https://huggingface.co/{args.hub_model_id}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune BLIP-VQA-Base on SHROOM hallucination detection"
    )

    # Data
    parser.add_argument(
        "--data_file",
        default="shroom-visions-data/distrib/shroom-vision.train.en.labeled.jsonl",
    )
    parser.add_argument("--images_dir", default="shroom-vis-images")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.10)

    # Model
    parser.add_argument(
        "--model_id", default="Salesforce/blip-vqa-base",
        help="HuggingFace model ID",
    )

    # Sequence lengths
    parser.add_argument("--max_question_length", type=int, default=512,
                        help="Max tokens for question input")
    parser.add_argument("--max_answer_length", type=int, default=128,
                        help="Max tokens for answer/target output")

    # Training
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument(
        "--output_dir", default="./checkpoints/blip-vqa-shroom-sft",
    )
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument(
        "--hub_model_id", default="amanuelbyte/BLIP-VQA-Base-SHROOM-SFT",
    )
    parser.add_argument("--hub_token", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  SHROOM-Visions SFT: BLIP-VQA-Base")
    logger.info("=" * 60)
    logger.info(f"Model:        {args.model_id}")
    logger.info(f"Data:         {args.data_file}")
    logger.info(f"Images:       {args.images_dir}")
    logger.info(f"Epochs:       {args.num_epochs}")
    logger.info(f"Batch:        {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    logger.info(f"LR:           {args.lr}")
    logger.info(f"Question len: {args.max_question_length}")
    logger.info(f"Answer len:   {args.max_answer_length}")
    logger.info(f"Output:       {args.output_dir}")
    if args.push_to_hub:
        logger.info(f"Hub:          {args.hub_model_id}")

    # ---- Step 1: Load model ----
    logger.info("\n[1/4] Loading model...")
    model, processor = load_model(args.model_id)

    # ---- Step 2: Load and prepare data ----
    logger.info("\n[2/4] Loading and preparing data...")
    train_dataset, eval_dataset = load_and_split_data(
        data_file=args.data_file,
        images_dir=args.images_dir,
        processor=processor,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
        max_question_length=args.max_question_length,
        max_answer_length=args.max_answer_length,
    )

    # ---- Step 3: Train ----
    logger.info("\n[3/4] Training...")
    trainer = train(model, processor, train_dataset, eval_dataset, args)

    # ---- Step 4: Save & Upload ----
    logger.info("\n[4/4] Saving and uploading...")
    save_and_upload(model, processor, args)

    logger.info("\n" + "=" * 60)
    logger.info("  ✓ BLIP-VQA-Base Finetuning complete!")
    logger.info("=" * 60)

    # Print final metrics
    if trainer.state.log_history:
        final_train_loss = None
        final_eval_loss = None
        for entry in reversed(trainer.state.log_history):
            if "loss" in entry and final_train_loss is None:
                final_train_loss = entry["loss"]
            if "eval_loss" in entry and final_eval_loss is None:
                final_eval_loss = entry["eval_loss"]
            if final_train_loss and final_eval_loss:
                break
        if final_train_loss:
            logger.info(f"Final train loss: {final_train_loss:.4f}")
        if final_eval_loss:
            logger.info(f"Final eval loss:  {final_eval_loss:.4f}")


if __name__ == "__main__":
    main()
