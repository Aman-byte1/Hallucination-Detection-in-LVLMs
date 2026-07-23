#!/usr/bin/env python3
"""
Finetune MiniCPM-V-4.6 on SHROOM-Visions Hallucination Detection
=================================================================
Uses LoRA via peft for parameter-efficient finetuning on A40 GPU.
MiniCPM-V-4.6 is a ~1B multimodal model (SigLIP2-400M + Qwen3.5-0.8B).

The model learns to:
  1. Look at an image
  2. Read a prompt and VLM response about that image
  3. Output a JSON array of hallucinated text spans with categories

Usage:
    # Quick test (10 samples)
    python finetune_minicpm.py --max_samples 10

    # Full training
    python finetune_minicpm.py --push_to_hub --hub_token hf_xxx

    # Resume from checkpoint
    python finetune_minicpm.py --resume_from_checkpoint ./checkpoints/minicpm-v4.6-shroom-sft/checkpoint-xxx
"""

import argparse
import copy
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

# ============================================================================
# Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Same system prompt as evaluate.py / finetune.py for consistency
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


def build_user_prompt(sample: dict) -> str:
    """Build the user message text (same format as evaluate.py)."""
    return (
        f"IMAGE QUESTION: {sample['prompt']}\n\n"
        f"RESPONSE TO CHECK: {sample['response']}\n\n"
        f"Find factual errors. Flag anything that seems wrong.\n"
        f"Quote 1-3 wrong words only. Output JSON array or []."
    )


# ============================================================================
# MiniCPM-V-4.6 Dataset
# ============================================================================

class MiniCPMV46SHROOMDataset(TorchDataset):
    """Prepare SHROOM samples for MiniCPM-V-4.6 training.

    MiniCPM-V-4.6 uses AutoProcessor with apply_chat_template.
    This dataset prepares conversations and tokenizes them for training.
    """

    def __init__(self, samples, processor, images_dir, max_length=2048):
        self.samples = samples
        self.processor = processor
        self.images_dir = Path(images_dir)
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def _load_image(self, image_name):
        """Load and preprocess an image, with fallback to blank image."""
        if image_name:
            image_path = find_image(image_name, self.images_dir)
            if image_path:
                try:
                    img = Image.open(image_path).convert("RGB")
                    return img
                except Exception as e:
                    logger.debug(f"Error loading image {image_path}: {e}")

        # Fallback: create a small blank image
        return Image.new("RGB", (224, 224), (128, 128, 128))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        response = sample["response"]
        labels_data = sample.get("labels", [])

        # Build target (what the model should output)
        spans = labels_to_text_spans(labels_data, response)
        target = json.dumps(spans, ensure_ascii=False) if spans else "[]"

        # Build user prompt
        user_text = build_user_prompt(sample)

        # Load image
        image = self._load_image(sample.get("image_name", ""))

        # Build messages in chat format
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
            {"role": "assistant", "content": target},
        ]

        # Tokenize using processor.apply_chat_template
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                max_slice_nums=1,  # Keep small for training efficiency
            )
        except Exception as e:
            logger.warning(f"Error tokenizing sample {idx}: {e}, using fallback")
            # Fallback: tokenize text-only
            full_text = f"{SYSTEM_PROMPT}\n{user_text}\n{target}"
            tokenized = self.processor.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            # Mask first 80% as approximate prompt
            mask_len = int(len(labels) * 0.8)
            labels[:mask_len] = -100
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        # Extract tensors (squeeze batch dim from apply_chat_template)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Truncate to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        # Create labels: mask everything before assistant response
        labels = input_ids.clone()

        # Find where the target/assistant response starts
        target_tokens = self.processor.tokenizer(
            target, add_special_tokens=False
        )["input_ids"]

        target_start = None
        target_len = len(target_tokens)
        ids_list = input_ids.tolist()
        for i in range(len(ids_list) - target_len + 1):
            if ids_list[i:i + target_len] == target_tokens:
                target_start = i
                break

        if target_start is not None:
            labels[:target_start] = -100
        else:
            # Fallback: mask first 80%
            mask_len = int(len(labels) * 0.8)
            labels[:mask_len] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Include all extra vision keys from inputs (pixel_values, image_sizes, tgt_sizes, etc.)
        for key, val in inputs.items():
            if key not in ["input_ids", "attention_mask"]:
                if isinstance(val, torch.Tensor):
                    result[key] = val.squeeze(0) if (val.dim() > 1 and val.shape[0] == 1) else val
                else:
                    result[key] = val

        return result


class MiniCPMV46DataCollator:
    """Collate MiniCPM-V-4.6 training samples into batches.

    Handles padding of variable-length sequences.
    """

    def __init__(self, processor, max_length=2048):
        self.processor = processor
        self.max_length = max_length
        self.pad_token_id = (
            processor.tokenizer.pad_token_id
            or processor.tokenizer.eos_token_id
        )

    def __call__(self, batch):
        max_len = min(
            max(len(item["input_ids"]) for item in batch),
            self.max_length,
        )

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for item in batch:
            ids = item["input_ids"][:max_len]
            mask = item["attention_mask"][:max_len]
            labs = item["labels"][:max_len]

            # Pad to max_len
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = torch.cat([
                    ids,
                    torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                ])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
                labs = torch.cat([
                    labs, torch.full((pad_len,), -100, dtype=torch.long),
                ])

            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            labels_list.append(labs)

        result = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }

        # Helper to stack/convert vision tensors (pixel_values, image_sizes, tgt_sizes, etc.)
        def pad_and_stack(vals):
            if not vals:
                return None
            first = vals[0]
            if isinstance(first, torch.Tensor):
                try:
                    return torch.stack(vals)
                except RuntimeError:
                    # Pad variable shapes to max size across batch
                    max_dims = [max(t.shape[i] for t in vals) for i in range(first.dim())]
                    padded = []
                    for t in vals:
                        pad_amounts = []
                        for i in reversed(range(t.dim())):
                            pad_amounts.extend([0, max_dims[i] - t.shape[i]])
                        padded.append(torch.nn.functional.pad(t, pad_amounts))
                    return torch.stack(padded)
            elif isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (int, float)):
                return torch.tensor(vals, dtype=torch.long)
            elif isinstance(first, (int, float)):
                return torch.tensor(vals)
            else:
                return vals

        # Collate all extra vision keys
        extra_keys = set(batch[0].keys()) - {"input_ids", "attention_mask", "labels"}
        for key in extra_keys:
            key_vals = [item[key] for item in batch if key in item]
            if len(key_vals) == len(batch):
                result[key] = pad_and_stack(key_vals)

        return result


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
    max_length: int = 2048,
) -> tuple:
    """Load SHROOM data, convert to MiniCPM-V-4.6 format, split train/eval."""
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
        logger.info(f"Limited to {max_samples} samples for testing")

    # Split indices — SAME seed/ratio as finetune.py for consistency
    rng = np.random.RandomState(seed)
    n_eval = max(1, int(len(raw_samples) * eval_ratio))
    eval_indices = set(rng.choice(len(raw_samples), size=n_eval, replace=False))

    train_samples = [s for i, s in enumerate(raw_samples) if i not in eval_indices]
    eval_samples = [s for i, s in enumerate(raw_samples) if i in eval_indices]

    logger.info(f"Split: {len(train_samples)} train + {len(eval_samples)} eval")

    # Create datasets
    train_dataset = MiniCPMV46SHROOMDataset(
        train_samples, processor, images_dir, max_length
    )
    eval_dataset = MiniCPMV46SHROOMDataset(
        eval_samples, processor, images_dir, max_length
    )

    return train_dataset, eval_dataset


# ============================================================================
# Model Loading & Training
# ============================================================================

def load_model(model_id: str, lora_rank: int):
    """Load MiniCPM-V-4.6 and apply LoRA adapters."""
    from peft import LoraConfig, get_peft_model

    logger.info(f"Loading model: {model_id}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU Memory: {mem_gb:.1f} GB")

    # Determine dtype
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
    logger.info(f"Using dtype: {dtype}")

    # Load model using the native transformers API for MiniCPM-V-4.6
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    # Ensure pad token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Freeze vision encoder
    frozen_count = 0
    for name, param in model.named_parameters():
        # Freeze vision-related modules (keep LLM trainable via LoRA)
        if any(vk in name.lower() for vk in [
            "vision", "vpm", "visual", "resampler", "siglip",
            "image_encoder", "img_processor",
        ]):
            param.requires_grad = False
            frozen_count += 1

    if frozen_count > 0:
        logger.info(f"Froze {frozen_count} vision encoder parameters")

    # Apply LoRA to LLM layers
    logger.info("Applying LoRA adapters to LLM layers...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    return model, processor


def train(model, processor, train_dataset, eval_dataset, args):
    """Run SFT training."""
    from transformers import TrainingArguments, Trainer

    logger.info(
        f"Starting training: {len(train_dataset)} train, "
        f"{len(eval_dataset)} eval samples"
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
        f"Steps/epoch: {steps_per_epoch}, "
        f"eval every {eval_steps} steps, save every {save_steps} steps"
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

    collator = MiniCPMV46DataCollator(processor, max_length=args.max_seq_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    resume = args.resume_from_checkpoint if args.resume_from_checkpoint else None
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")

    trainer.train(resume_from_checkpoint=resume)
    logger.info("Training complete!")
    return trainer


def save_and_upload(model, processor, args):
    """Save the finetuned model and optionally push to HuggingFace."""
    # Save locally
    save_dir = os.path.join(args.output_dir, "final")
    logger.info(f"Saving model to {save_dir}")

    # Save LoRA adapters
    lora_dir = os.path.join(args.output_dir, "lora_adapters")
    model.save_pretrained(lora_dir)
    processor.save_pretrained(lora_dir)
    logger.info(f"LoRA adapters saved to {lora_dir}")

    # Merge LoRA and save full model
    try:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(save_dir, safe_serialization=True)
        processor.save_pretrained(save_dir)
        logger.info(f"Merged model saved to {save_dir}")
    except Exception as e:
        logger.warning(f"Could not merge LoRA: {e}. Saving adapter only.")
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)

    # Push to Hub
    if args.push_to_hub:
        logger.info(f"Pushing to HuggingFace: {args.hub_model_id}")
        try:
            model.push_to_hub(
                args.hub_model_id,
                token=args.hub_token,
                private=False,
            )
            processor.push_to_hub(
                args.hub_model_id,
                token=args.hub_token,
            )
            logger.info(
                f"✓ Model uploaded to https://huggingface.co/{args.hub_model_id}"
            )
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")
            logger.info("You can manually push later from the saved checkpoint.")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune MiniCPM-V-4.6 on SHROOM hallucination detection"
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
        "--model_id", default="openbmb/MiniCPM-V-4.6",
        help="HuggingFace model ID",
    )
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=2048)

    # Training
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument(
        "--output_dir", default="./checkpoints/minicpm-v4.6-shroom-sft",
    )
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument(
        "--hub_model_id", default="amanuelbyte/MiniCPM-V-4.6-SHROOM-SFT",
    )
    parser.add_argument("--hub_token", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  SHROOM-Visions SFT: MiniCPM-V-4.6")
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

    # ---- Step 1: Load model ----
    logger.info("\n[1/4] Loading model...")
    model, processor = load_model(
        model_id=args.model_id,
        lora_rank=args.lora_rank,
    )

    # ---- Step 2: Load and prepare data ----
    logger.info("\n[2/4] Loading and preparing data...")
    train_dataset, eval_dataset = load_and_split_data(
        data_file=args.data_file,
        images_dir=args.images_dir,
        processor=processor,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
        max_length=args.max_seq_length,
    )

    # ---- Step 3: Train ----
    logger.info("\n[3/4] Training...")
    trainer = train(model, processor, train_dataset, eval_dataset, args)

    # ---- Step 4: Save & Upload ----
    logger.info("\n[4/4] Saving and uploading...")
    save_and_upload(model, processor, args)

    logger.info("\n" + "=" * 60)
    logger.info("  ✓ MiniCPM-V-4.6 Finetuning complete!")
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
