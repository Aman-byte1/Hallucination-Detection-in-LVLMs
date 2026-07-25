"""
SpanCalib-VLM Training Script
==============================
Trains SpanCalibVLM using Focal Loss + Soft-Dice Loss + Pearson Correlation Loss.

Usage:
    # Quick test (10 samples)
    python spancalib_vlm/train.py --max_samples 10

    # Full training
    python spancalib_vlm/train.py --epochs 3 --batch_size 8 --lr 3e-5
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

# Add parent directory to path to enable clean package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spancalib_vlm.dataset import SpanCalibDataset, DataCollatorForSpanCalib
from spancalib_vlm.model import SpanCalibVLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data(filepath: Path) -> list[dict]:
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} samples from {filepath.name}")
    return samples


def split_data(samples: list[dict], ratio: float = 0.10, seed: int = 42):
    rng = np.random.RandomState(seed)
    n_eval = max(1, int(len(samples) * ratio))
    indices = rng.choice(len(samples), size=n_eval, replace=False)
    eval_indices = set(indices)
    train_samples = [samples[i] for i in range(len(samples)) if i not in eval_indices]
    eval_samples = [samples[i] for i in sorted(indices)]
    logger.info(f"Split: {len(train_samples)} train samples, {len(eval_samples)} eval samples ({ratio*100:.0f}%)")
    return train_samples, eval_samples


def evaluate_val(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    all_pred_probs = []
    all_target_probs = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device) if batch["pixel_values"] is not None else None
            resp_mask = batch["response_token_mask"].to(device)
            binary_labels = batch["binary_labels"].to(device)
            prob_labels = batch["prob_labels"].to(device)
            cat_labels = batch["category_labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                response_token_mask=resp_mask,
                binary_labels=binary_labels,
                prob_labels=prob_labels,
                category_labels=cat_labels,
            )

            total_loss += outputs["loss"].item()
            probs = outputs["pred_probs"].detach().cpu()
            resp_m = resp_mask.detach().cpu()
            targets = prob_labels.detach().cpu()

            for p, m, t in zip(probs, resp_m, targets):
                p_masked = p[m].numpy()
                t_masked = t[m].numpy()
                if len(p_masked) > 0:
                    all_pred_probs.extend(p_masked)
                    all_target_probs.extend(t_masked)

    avg_loss = total_loss / max(1, len(val_loader))

    # Pearson correlation calculation
    if len(all_pred_probs) > 1 and np.std(all_pred_probs) > 0 and np.std(all_target_probs) > 0:
        corr = float(np.corrcoef(all_pred_probs, all_target_probs)[0, 1])
    else:
        corr = 0.0

    return avg_loss, corr


def parse_args():
    parser = argparse.ArgumentParser(description="Train SpanCalib-VLM Model")
    parser.add_argument(
        "--data_file",
        default="shroom-visions-data/distrib/shroom-vision.train.en.labeled.jsonl",
    )
    parser.add_argument("--images_dir", default="shroom-vis-images")
    parser.add_argument("--model_id", default="xlm-roberta-base")
    parser.add_argument("--use_vision", action="store_true", help="Enable SigLIP-2 vision tower cross-attention fusion")
    parser.add_argument("--vision_model_id", default="google/siglip-base-patch16-224", help="SigLIP vision model ID")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="./checkpoints/spancalib_vlm")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  SpanCalib-VLM Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Model ID:   {args.model_id}")
    logger.info(f"Vision:     {args.use_vision} ({args.vision_model_id if args.use_vision else 'None'})")
    logger.info(f"Epochs:     {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    logger.info(f"LR:         {args.lr}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    data_path = Path(args.data_file)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    all_samples = load_data(data_path)
    train_samples, val_samples = split_data(all_samples, ratio=0.10, seed=args.seed)

    if args.max_samples:
        train_samples = train_samples[:args.max_samples]
        val_samples = val_samples[:max(1, args.max_samples // 10)]
        logger.info(f"Limited to {len(train_samples)} train / {len(val_samples)} val samples (--max_samples)")

    # 2. Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    images_dir = Path(args.images_dir)

    train_dataset = SpanCalibDataset(train_samples, tokenizer, images_dir, use_vision=args.use_vision, vision_model_id=args.vision_model_id)
    val_dataset = SpanCalibDataset(val_samples, tokenizer, images_dir, use_vision=args.use_vision, vision_model_id=args.vision_model_id)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
    collator = DataCollatorForSpanCalib(pad_token_id=pad_id)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # 3. Model & Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = SpanCalibVLM(model_id=args.model_id, use_vision=args.use_vision)
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    best_corr = -1.0

    # 5. Training Loop
    logger.info("\nStarting Training...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device) if batch["pixel_values"] is not None else None
            resp_mask = batch["response_token_mask"].to(device)
            binary_labels = batch["binary_labels"].to(device)
            prob_labels = batch["prob_labels"].to(device)
            cat_labels = batch["category_labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                response_token_mask=resp_mask,
                binary_labels=binary_labels,
                prob_labels=prob_labels,
                category_labels=cat_labels,
            )

            loss = outputs["loss"] / args.grad_accum
            loss.backward()
            epoch_loss += outputs["loss"].item()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{outputs['loss'].item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, val_corr = evaluate_val(model, val_loader, device)

        logger.info(
            f"Epoch {epoch} Complete | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Pearson Corr: {val_corr:.4f}"
        )

        # Save Best Checkpoint
        if val_corr > best_corr:
            best_corr = val_corr
            best_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"✓ Saved new best checkpoint to {best_path} (Pearson Corr: {best_corr:.4f})")

    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"\nTraining Complete! Total time: {time.time() - start_time:.1f}s")
    logger.info(f"Best Validation Pearson Correlation: {best_corr:.4f}")


if __name__ == "__main__":
    main()
