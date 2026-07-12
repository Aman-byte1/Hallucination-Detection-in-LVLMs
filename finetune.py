#!/usr/bin/env python3
"""
Finetune Qwen3.5-4B on SHROOM-Visions Hallucination Detection
==============================================================
Uses Unsloth for 2x faster training + 50% less VRAM on A40 GPU.

The model learns to:
  1. Look at an image
  2. Read a prompt and VLM response about that image  
  3. Output a JSON array of hallucinated text spans with categories
     e.g. [{"text": "blue", "label": "mischaracterization", "prob": 0.9}]
     or [] if no hallucinations found

This is the SFT (Supervised Fine-Tuning) stage. RL comes later.

Usage:
    # Quick test (10 samples)
    python finetune.py --max_samples 10

    # Full training
    python finetune.py --push_to_hub --hub_token hf_xxx

    # Resume from checkpoint
    python finetune.py --resume_from_checkpoint ./checkpoints/qwen35-4b-shroom-sft/checkpoint-xxx
"""

import argparse
import json
import logging
import os
import urllib.parse
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset

# ============================================================================
# Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Same system prompt as evaluate.py — the model learns this during SFT
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
        # Direct path
        direct = images_dir / name
        if direct.exists():
            return direct
        # Recursive search
        try:
            matches = list(images_dir.glob(f"**/{name}"))
            if matches:
                return matches[0]
        except Exception:
            pass

    return None


def labels_to_text_spans(labels: list[dict], response: str) -> list[dict]:
    """Convert character-level SHROOM labels to text span format.

    Input:  {"start": 148, "end": 154, "label": "mischaracterization", "prob": 0.33}
    Output: {"text": "cables", "label": "mischaracterization", "prob": 0.33}
    """
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


def build_user_prompt(sample: dict) -> str:
    """Build the user message text (same format as evaluate.py)."""
    return (
        f"IMAGE QUESTION: {sample['prompt']}\n\n"
        f"RESPONSE TO CHECK: {sample['response']}\n\n"
        f"Find factual errors. Flag anything that seems wrong.\n"
        f"Quote 1-3 wrong words only. Output JSON array or []."
    )


def convert_sample(sample: dict, images_dir: Path) -> dict | None:
    """Convert a single SHROOM sample to Unsloth vision chat format.

    Returns a dict with "messages" key in the format expected by
    UnslothVisionDataCollator.
    """
    response = sample["response"]
    labels = sample.get("labels", [])

    # Build ground-truth target (what the model should output)
    spans = labels_to_text_spans(labels, response)
    target = json.dumps(spans, ensure_ascii=False) if spans else "[]"

    # Build user prompt
    user_text = build_user_prompt(sample)

    # Find image
    image_name = sample.get("image_name", "")
    image_path = find_image(image_name, images_dir) if image_name else None

    # Build multimodal user content
    if image_path is not None:
        user_content = [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": user_text},
        ]
    else:
        # Text-only fallback (no image found)
        user_content = [
            {"type": "text", "text": user_text},
        ]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]

    return {"messages": messages}


def load_and_split_data(
    data_file: str,
    images_dir: str,
    eval_ratio: float = 0.10,
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[Dataset, Dataset]:
    """Load SHROOM data, convert to chat format, split train/eval."""
    images_path = Path(images_dir)

    # Load raw JSONL
    raw_samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_samples.append(json.loads(line))

    logger.info(f"Loaded {len(raw_samples)} raw samples from {data_file}")

    # Optionally limit samples (for testing)
    if max_samples is not None:
        raw_samples = raw_samples[:max_samples]
        logger.info(f"Limited to {max_samples} samples for testing")

    # Split indices
    rng = np.random.RandomState(seed)
    n_eval = max(1, int(len(raw_samples) * eval_ratio))
    eval_indices = set(rng.choice(len(raw_samples), size=n_eval, replace=False))

    # Convert samples
    train_data, eval_data = [], []
    n_with_image = 0
    n_without_image = 0

    for i, sample in enumerate(raw_samples):
        converted = convert_sample(sample, images_path)
        if converted is None:
            continue

        # Track image availability
        has_image = any(
            item.get("type") == "image"
            for item in converted["messages"][1]["content"]
        )
        if has_image:
            n_with_image += 1
        else:
            n_without_image += 1

        if i in eval_indices:
            eval_data.append(converted)
        else:
            train_data.append(converted)

    logger.info(
        f"Converted {len(train_data)} train + {len(eval_data)} eval samples "
        f"({n_with_image} with images, {n_without_image} without)"
    )

    return Dataset.from_list(train_data), Dataset.from_list(eval_data)


# ============================================================================
# Model Loading & Training
# ============================================================================

def load_model(model_id: str, max_seq_length: int, lora_rank: int):
    """Load Qwen3.5-4B with Unsloth and apply LoRA adapters."""
    from unsloth import FastVisionModel

    logger.info(f"Loading model: {model_id}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info(f"GPU Memory: {mem_gb:.1f} GB")

    # Load base model in BF16 (NOT 4-bit — Unsloth docs say QLoRA not
    # recommended for Qwen3.5 due to quantization differences)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
    )

    logger.info("Applying LoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,      # Finetune vision encoder
        finetune_language_layers=True,     # Finetune language layers
        finetune_attention_modules=True,   # Finetune attention
        finetune_mlp_modules=True,         # Finetune MLP
        r=lora_rank,
        lora_alpha=lora_rank,              # alpha == r is recommended
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    # Print trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    return model, tokenizer


def train(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    args: argparse.Namespace,
):
    """Run SFT training with Unsloth."""
    from trl import SFTTrainer, SFTConfig
    from unsloth import UnslothVisionDataCollator

    logger.info(
        f"Starting training: {len(train_dataset)} train, "
        f"{len(eval_dataset)} eval samples"
    )
    logger.info(
        f"Effective batch size: "
        f"{args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}"
    )

    # Calculate logging/eval steps based on dataset size
    steps_per_epoch = max(
        1, len(train_dataset) // (args.batch_size * args.grad_accum)
    )
    eval_steps = max(1, steps_per_epoch // 4)   # Eval ~4 times per epoch
    save_steps = max(1, steps_per_epoch // 2)    # Save ~2 times per epoch
    logger.info(
        f"Steps/epoch: {steps_per_epoch}, "
        f"eval every {eval_steps} steps, save every {save_steps} steps"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.05,
            num_train_epochs=args.num_epochs,
            learning_rate=args.lr,
            bf16=True,
            fp16=False,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,
            output_dir=args.output_dir,
            optim="adamw_8bit",
            seed=args.seed,
            max_seq_length=args.max_seq_length,
            dataset_num_proc=4,
            dataloader_pin_memory=True,
            report_to="none",       # Set to "wandb" if you want W&B logging
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            remove_unused_columns=False,
        ),
    )

    # Resume from checkpoint if specified
    resume = None
    if args.resume_from_checkpoint:
        resume = args.resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {resume}")

    trainer.train(resume_from_checkpoint=resume)

    logger.info("Training complete!")
    return trainer


def save_and_upload(model, tokenizer, args: argparse.Namespace):
    """Save the finetuned model and optionally push to HuggingFace."""

    # Always save locally first
    local_merged = os.path.join(args.output_dir, "merged")
    logger.info(f"Saving merged model to {local_merged}")
    model.save_pretrained_merged(
        local_merged,
        tokenizer,
        save_method="merged_16bit",
    )
    logger.info(f"Merged model saved to {local_merged}")

    # Also save LoRA adapters separately (smaller, for RL stage)
    lora_dir = os.path.join(args.output_dir, "lora_adapters")
    logger.info(f"Saving LoRA adapters to {lora_dir}")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    # Push to HuggingFace Hub
    if args.push_to_hub:
        token = args.hub_token
        hub_id = args.hub_model_id
        logger.info(f"Pushing merged model to HuggingFace: {hub_id}")
        model.push_to_hub_merged(
            hub_id,
            tokenizer,
            save_method="merged_16bit",
            token=token,
            private=False,
        )
        logger.info(f"✓ Model uploaded to https://huggingface.co/{hub_id}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune Qwen3.5-4B on SHROOM hallucination detection"
    )

    # Data
    parser.add_argument(
        "--data_file",
        default="shroom-visions-data/distrib/shroom-vision.train.en.labeled.jsonl",
        help="Path to labeled JSONL training data",
    )
    parser.add_argument(
        "--images_dir",
        default="shroom-vis-images",
        help="Directory containing the SHROOM images",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples (for quick testing)",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.10,
        help="Fraction of data to use for evaluation",
    )

    # Model
    parser.add_argument(
        "--model_id",
        default="unsloth/Qwen3.5-4B",
        help="HuggingFace model ID (use unsloth/ prefix for faster loading)",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (higher = more capacity, more VRAM)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for training",
    )

    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument(
        "--output_dir",
        default="./checkpoints/qwen35-4b-shroom-sft",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push finetuned model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        default="amanuelbyte/Qwen3.5-4B-SHROOM-SFT",
        help="HuggingFace model ID for upload",
    )
    parser.add_argument(
        "--hub_token",
        default=None,
        help="HuggingFace API token",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to checkpoint to resume training from",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  SHROOM-Visions SFT Finetuning")
    logger.info("=" * 60)
    logger.info(f"Model:      {args.model_id}")
    logger.info(f"Data:       {args.data_file}")
    logger.info(f"Images:     {args.images_dir}")
    logger.info(f"Epochs:     {args.num_epochs}")
    logger.info(f"Batch:      {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    logger.info(f"LR:         {args.lr}")
    logger.info(f"LoRA rank:  {args.lora_rank}")
    logger.info(f"Output:     {args.output_dir}")
    if args.push_to_hub:
        logger.info(f"Hub:        {args.hub_model_id}")

    # ---- Step 1: Load and prepare data ----
    logger.info("\n[1/4] Loading and preparing data...")
    train_dataset, eval_dataset = load_and_split_data(
        data_file=args.data_file,
        images_dir=args.images_dir,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    # ---- Step 2: Load model ----
    logger.info("\n[2/4] Loading model with Unsloth...")
    model, tokenizer = load_model(
        model_id=args.model_id,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
    )

    # ---- Step 3: Train ----
    logger.info("\n[3/4] Training...")
    trainer = train(model, tokenizer, train_dataset, eval_dataset, args)

    # ---- Step 4: Save & Upload ----
    logger.info("\n[4/4] Saving and uploading...")
    save_and_upload(model, tokenizer, args)

    logger.info("\n" + "=" * 60)
    logger.info("  ✓ Finetuning complete!")
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
