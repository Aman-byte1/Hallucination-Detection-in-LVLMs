#!/usr/bin/env python3
"""
GRPO Vision RL Training for SHROOM-Visions Hallucination Detection
==================================================================
Uses Unsloth + GRPO to reinforce the SFT-finetuned Qwen3.5-4B model
with reward functions tailored to hallucination detection.

Starting model: amanuelbyte/Qwen3.5-4B-SHROOM-SFT
Output model:   amanuelbyte/Qwen3.5-4B-SHROOM-GRPO

Reward functions:
  1. format_reward:         Valid JSON array output? (+2.0)
  2. detection_reward:      Correct hallucination presence detection? (+3.0)
  3. span_iou_reward:       Character-level IoU between pred and gold spans (+4.0)
  4. label_accuracy_reward: Correct category labels used? (+1.0)

Usage:
    python rl_grpo.py                          # Full training (500 steps)
    python rl_grpo.py --max_samples 50         # Quick test with fewer samples
    python rl_grpo.py --max_steps 100          # Limit training steps
"""

import argparse
import ast
import json
import logging
import os
import re
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

# Same system prompt used in finetune.py and evaluate.py
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

VALID_LABELS = {"invention", "mischaracterization", "OCR", "miscounting"}


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


def build_user_prompt(sample: dict) -> str:
    """Build the user message text (same format as evaluate.py)."""
    return (
        f"IMAGE QUESTION: {sample['prompt']}\n\n"
        f"RESPONSE TO CHECK: {sample['response']}\n\n"
        f"Find factual errors. Flag anything that seems wrong.\n"
        f"Quote 1-3 wrong words only. Output JSON array or []."
    )


def prepare_grpo_dataset(
    data_file: str,
    images_dir: str,
    eval_ratio: float = 0.10,
    seed: int = 42,
    max_samples: int | None = None,
) -> Dataset:
    """Load SHROOM data and prepare for GRPO training.

    Returns a Dataset with:
      - prompt: list of messages (system + user, NO assistant)
      - gold_labels_json: JSON string of gold labels
      - response_text: the original VLM response being checked
      - has_hallucination: whether gold labels exist
    """
    images_path = Path(images_dir)

    # Load raw JSONL
    raw_samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_samples.append(json.loads(line))

    logger.info(f"Loaded {len(raw_samples)} raw samples from {data_file}")

    # Split — use the SAME seed/ratio as finetune.py so we exclude the eval set
    rng = np.random.RandomState(seed)
    n_eval = max(1, int(len(raw_samples) * eval_ratio))
    eval_indices = set(rng.choice(len(raw_samples), size=n_eval, replace=False))

    train_samples = [s for i, s in enumerate(raw_samples) if i not in eval_indices]
    logger.info(
        f"Using {len(train_samples)} training samples "
        f"(excluded {n_eval} eval samples)"
    )

    if max_samples is not None:
        train_samples = train_samples[:max_samples]
        logger.info(f"Limited to {max_samples} samples")

    # Convert to GRPO format
    grpo_data = []
    n_with_image = 0

    for sample in train_samples:
        image_name = sample.get("image_name", "")
        image_path = find_image(image_name, images_path) if image_name else None

        user_text = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(sample)}"

        # Build multimodal user content
        if image_path is not None:
            user_content = [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_text},
            ]
            n_with_image += 1
        else:
            user_content = [
                {"type": "text", "text": user_text},
            ]

        # Prompt = user role only (system prompt is prepended)
        prompt = [
            {"role": "user", "content": user_content},
        ]

        labels = sample.get("labels", [])

        grpo_data.append({
            "prompt": prompt,
            "gold_labels_json": json.dumps(labels, ensure_ascii=False),
            "response_text": sample["response"],
            "has_hallucination": len(labels) > 0,
        })

    logger.info(
        f"Prepared {len(grpo_data)} GRPO samples ({n_with_image} with images)"
    )
    return Dataset.from_list(grpo_data)


# ============================================================================
# Parsing Helpers (shared by reward functions)
# ============================================================================

def _parse_json_output(text: str) -> list[dict] | None:
    """Try to parse a JSON array from model output. Returns None on failure."""
    text = text.strip()

    # Remove thinking tags
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    # Remove markdown code block wrappers
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Find JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None

    json_str = match.group(0)

    # Try json.loads
    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try ast.literal_eval (handles single quotes)
    try:
        result = ast.literal_eval(json_str)
        if isinstance(result, list):
            return result
    except (ValueError, SyntaxError):
        pass

    # Try replacing single → double quotes
    try:
        result = json.loads(json_str.replace("'", '"'))
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    return None


def _find_span_in_text(
    span_text: str, response_text: str
) -> tuple[int, int] | None:
    """Find a span's character position in the response text."""
    if not span_text or not response_text:
        return None
    words = span_text.lower().split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(w) for w in words)
    m = re.search(pattern, response_text.lower())
    if m:
        return (m.start(), m.end())
    return None


def _compute_char_iou(
    gold_labels: list[dict], pred_items: list[dict], response_text: str
) -> float:
    """Compute character-level IoU between gold and predicted spans."""
    resp_len = len(response_text)
    if resp_len == 0:
        return 1.0 if not gold_labels and not pred_items else 0.0

    # Gold binary array
    gold_arr = np.zeros(resp_len, dtype=np.float64)
    for label in gold_labels:
        start = max(0, label.get("start", 0))
        end = min(resp_len, label.get("end", 0))
        gold_arr[start:end] = 1.0

    # Pred binary array — locate text spans in the response
    pred_arr = np.zeros(resp_len, dtype=np.float64)
    for item in pred_items:
        if not isinstance(item, dict):
            continue
        if "text" in item:
            pos = _find_span_in_text(str(item["text"]), response_text)
            if pos:
                pred_arr[pos[0] : pos[1]] = 1.0
        elif "start" in item and "end" in item:
            try:
                s, e = int(item["start"]), int(item["end"])
                if 0 <= s < e <= resp_len:
                    pred_arr[s:e] = 1.0
            except (ValueError, TypeError):
                pass

    intersection = np.sum(gold_arr * pred_arr)
    union = np.sum(np.maximum(gold_arr, pred_arr))

    if union == 0:
        return 1.0  # Both empty → perfect agreement
    return float(intersection / union)


# ============================================================================
# Reward Functions
# ============================================================================

def _extract_completion_text(completion) -> str:
    """Extract text from a TRL completion (may be str or list of message dicts)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # TRL passes completions as [{"role": "assistant", "content": "..."}]
        for msg in completion:
            if isinstance(msg, dict) and "content" in msg:
                content = msg["content"]
                if isinstance(content, str):
                    return content
                # content could be a list of dicts [{"type": "text", "text": "..."}]
                if isinstance(content, list):
                    return " ".join(
                        item.get("text", "") for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
        # Fallback: join all string elements
        return " ".join(str(x) for x in completion)
    return str(completion)

def format_reward(completions, **kwargs) -> list[float]:
    """Reward for outputting valid JSON array format.

    +2.0  valid JSON array with correct fields
    +1.0  parseable but missing fields
    -1.0  unparseable
    -2.0  extra penalty for excessive gibberish
    """
    scores = []
    for completion in completions:
        text = _extract_completion_text(completion)
        parsed = _parse_json_output(text)

        if parsed is not None:
            all_valid = all(
                isinstance(item, dict) and "text" in item and "label" in item
                for item in parsed
            ) if parsed else True  # [] is valid

            score = 2.0 if all_valid else 1.0
        else:
            score = -1.0

        # Penalise gibberish / repetitive output
        if len(text) > 0:
            cleaned = text.replace("addCriterion", "").replace("\n", "")
            if (len(text) - len(cleaned)) / len(text) >= 0.3:
                score -= 2.0
            if len(text) > 1000:
                score -= 0.5

        scores.append(score)
    return scores


def detection_reward(completions, gold_labels_json, **kwargs) -> list[float]:
    """Reward for correctly detecting hallucination presence.

    +3.0  correct (both present or both absent)
    -1.0  wrong
    """
    scores = []
    for completion, gold_json in zip(completions, gold_labels_json):
        text = _extract_completion_text(completion)
        gold = json.loads(gold_json)
        gold_has = len(gold) > 0

        parsed = _parse_json_output(text)
        if parsed is None:
            scores.append(-1.0)
            continue

        pred_has = len(parsed) > 0
        scores.append(3.0 if gold_has == pred_has else -1.0)

    return scores


def span_iou_reward(
    completions, gold_labels_json, response_text, **kwargs
) -> list[float]:
    """Reward proportional to character-level IoU.

    Maps IoU [0, 1] → score [-1.0, +4.0].
    """
    scores = []
    for completion, gold_json, resp in zip(
        completions, gold_labels_json, response_text
    ):
        text = _extract_completion_text(completion)
        gold = json.loads(gold_json)
        parsed = _parse_json_output(text)

        if parsed is None:
            scores.append(-1.0)
            continue

        iou = _compute_char_iou(gold, parsed, resp)
        score = -1.0 + 5.0 * iou  # IoU 0→-1, IoU 1→+4
        scores.append(score)

    return scores


def label_accuracy_reward(
    completions, gold_labels_json, **kwargs
) -> list[float]:
    """Reward for using correct hallucination category labels.

    +1.0  predicted categories match gold exactly
    partial credit for overlap
    -0.5  no overlap / invalid
    """
    scores = []
    for completion, gold_json in zip(completions, gold_labels_json):
        text = _extract_completion_text(completion)
        gold = json.loads(gold_json)
        parsed = _parse_json_output(text)

        if parsed is None:
            scores.append(-0.5)
            continue

        gold_cats = set(
            l.get("label", "") for l in gold if isinstance(l, dict)
        )
        pred_cats = set(
            item.get("label", "")
            for item in parsed
            if isinstance(item, dict) and item.get("label", "") in VALID_LABELS
        )

        if not gold_cats and not pred_cats:
            score = 0.5  # Both empty — correct
        elif gold_cats == pred_cats:
            score = 1.0  # Perfect match
        elif gold_cats & pred_cats:
            score = len(gold_cats & pred_cats) / len(gold_cats | pred_cats)
        else:
            score = -0.5  # No overlap at all

        # Penalise invalid labels
        invalid = pred_cats - VALID_LABELS
        if invalid:
            score -= 0.25 * len(invalid)

        scores.append(score)
    return scores


# ============================================================================
# Model Loading & Training
# ============================================================================

def load_model(model_id: str, max_seq_length: int, lora_rank: int):
    """Load the SFT model with Unsloth and apply fresh LoRA for GRPO."""
    from unsloth import FastVisionModel

    logger.info(f"Loading SFT model: {model_id}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU Memory: {mem_gb:.1f} GB")

    # bf16 LoRA — Unsloth docs say QLoRA not recommended for Qwen3.5
    # fast_inference=False — Qwen3.5 not yet supported by vLLM
    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        fast_inference=False,
    )

    # ── Disable Qwen3.5 thinking mode ──
    # Qwen3.5's chat template generates <think>...</think> blocks by default.
    # These consume all completion tokens, leaving no room for JSON output.
    # Prepending `enable_thinking = false` forces direct answer generation.
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        if 'enable_thinking' in tokenizer.chat_template:
            tokenizer.chat_template = (
                "{%- set enable_thinking = false -%}\n"
                + tokenizer.chat_template
            )
            logger.info("Disabled thinking mode in chat template")

    logger.info("Applying LoRA adapters for GRPO...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_rank,
        lora_alpha=lora_rank * 2,  # *2 speeds up training per Unsloth docs
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_tokens"],
        use_gradient_checkpointing="unsloth",
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    return model, tokenizer


def train_grpo(model, tokenizer, dataset: Dataset, args):
    """Run GRPO training with hallucination-detection reward functions."""
    from trl import GRPOTrainer, GRPOConfig

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",

        # ── GRPO-specific ──
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=0.7,

        # Dr-GRPO loss (improved GRPO variant)
        loss_type="dr_grpo",

        # Clipping
        epsilon=0.2,
        max_grad_norm=0.1,

        # ── Logging & checkpointing ──
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        log_completions=True,

        # ── Duration ──
        max_steps=args.max_steps if args.max_steps else -1,
        num_train_epochs=args.num_epochs if not args.max_steps else 1,

        bf16=True,
        fp16=False,
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[
            format_reward,
            detection_reward,
            span_iou_reward,
            label_accuracy_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    logger.info("Starting GRPO training...")
    logger.info(f"  Dataset size:        {len(dataset)}")
    logger.info(f"  Generations/prompt:  {args.num_generations}")
    logger.info(f"  Max steps:           {args.max_steps or 'all'}")
    logger.info(f"  Grad accumulation:   {args.grad_accum}")
    logger.info(f"  Learning rate:       {args.lr}")
    logger.info(f"  Max completion len:  {args.max_completion_length}")

    trainer.train()
    logger.info("GRPO training complete!")
    return trainer


def save_and_push(model, tokenizer, args):
    """Save the GRPO-trained model and optionally push to HuggingFace."""
    local_dir = os.path.join(args.output_dir, "merged")
    logger.info(f"Saving merged model to {local_dir}")
    model.save_pretrained_merged(
        local_dir,
        tokenizer,
        save_method="merged_16bit",
    )
    logger.info(f"Merged model saved to {local_dir}")

    if args.push_to_hub:
        logger.info(f"Pushing to HuggingFace: {args.hub_model_id}")
        model.push_to_hub_merged(
            args.hub_model_id,
            tokenizer,
            save_method="merged_16bit",
            token=args.hub_token,
            private=False,
        )
        logger.info(
            f"✓ Model uploaded to https://huggingface.co/{args.hub_model_id}"
        )


# ============================================================================
# Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO Vision RL for SHROOM Hallucination Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rl_grpo.py                                    # Full 500-step training
  python rl_grpo.py --max_samples 50 --max_steps 20    # Quick test
  python rl_grpo.py --push_to_hub --hub_token hf_xxx   # Train + push
        """,
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
        "--model_id",
        default="amanuelbyte/Qwen3.5-4B-SHROOM-SFT",
        help="HuggingFace ID of the SFT model to start from",
    )
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=2048)

    # GRPO training
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument(
        "--output_dir", default="./checkpoints/qwen35-4b-shroom-grpo"
    )
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument(
        "--hub_model_id", default="amanuelbyte/Qwen3.5-4B-SHROOM-GRPO"
    )
    parser.add_argument("--hub_token", default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  SHROOM-Visions GRPO Vision RL Training")
    logger.info("=" * 60)
    logger.info(f"SFT Model:       {args.model_id}")
    logger.info(f"Output:          {args.output_dir}")
    logger.info(f"Hub:             {args.hub_model_id}")
    logger.info(f"Generations:     {args.num_generations}")
    logger.info(f"Max steps:       {args.max_steps}")
    logger.info(f"Completion len:  {args.max_completion_length}")
    logger.info(f"Grad accum:      {args.grad_accum}")
    logger.info(f"LR:              {args.lr}")

    # ── Step 1: Prepare dataset ──
    logger.info("\n[1/3] Preparing GRPO dataset...")
    dataset = prepare_grpo_dataset(
        data_file=args.data_file,
        images_dir=args.images_dir,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    # ── Step 2: Load model ──
    logger.info("\n[2/3] Loading model with Unsloth...")
    model, tokenizer = load_model(
        model_id=args.model_id,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
    )

    # ── Step 3: GRPO training ──
    logger.info("\n[3/3] GRPO training...")
    trainer = train_grpo(model, tokenizer, dataset, args)

    # ── Save & push ──
    logger.info("\nSaving and uploading...")
    save_and_push(model, tokenizer, args)

    logger.info("\n" + "=" * 60)
    logger.info("  ✓ GRPO Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
