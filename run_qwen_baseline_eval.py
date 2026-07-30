
#!/usr/bin/env python3
"""
Base Qwen-VL Zero-Shot Baseline Evaluation Script
=================================================
Evaluates un-finetuned Qwen-VL on the 10% eval split across EN, FR, IT, and ZH,
and computes Mean IoU and Spearman Correlation to compare against SFT + SpanCalib-VLM.
"""

import json
import re
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

DATA_DIR = Path("shroom-visions-data/distrib")
IMAGES_DIR = Path("shroom-vis-images")
BASE_MODEL_ID = "Qwen/Qwen3.5-4B"
LANGUAGES = ["en", "fr", "it", "zh"]
SEED = 42


def find_image(image_name: str, images_dir: Path) -> Path | None:
    if not images_dir.exists():
        return None
    direct = images_dir / image_name
    if direct.exists():
        return direct
    matches = list(images_dir.glob(f"**/{image_name}"))
    return matches[0] if matches else None


def load_dataset(data_dir: Path, lang: str):
    path = data_dir / f"shroom-vision.train.{lang}.labeled.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_data(samples, eval_ratio=0.10, seed=42):
    rng = np.random.RandomState(seed)
    n_eval = max(1, int(len(samples) * eval_ratio))
    eval_indices = set(rng.choice(len(samples), size=n_eval, replace=False))
    return [s for i, s in enumerate(samples) if i in eval_indices]


def parse_spans_from_text(text: str, response_len: int):
    spans = []
    # Try finding JSON block
    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "start" in item and "end" in item:
                        start = int(item["start"])
                        end = min(int(item["end"]), response_len)
                        prob = float(item.get("prob", 1.0))
                        label = str(item.get("label", "invention"))
                        if 0 <= start < end:
                            spans.append({"start": start, "end": end, "prob": prob, "label": label})
        except Exception:
            pass
    return spans


def compute_sample_scores(gold_sample, pred_spans):
    resp = gold_sample["response"]
    n = len(resp)
    if n == 0:
        return 1.0, 1.0

    gold_vec = [0.0] * n
    for s in gold_sample.get("labels", []):
        for i in range(s["start"], min(s["end"], n)):
            gold_vec[i] = 1.0

    pred_vec = [0.0] * n
    for s in pred_spans:
        for i in range(s["start"], min(s["end"], n)):
            pred_vec[i] = s.get("prob", 1.0)

    g_set = set(i for i, v in enumerate(gold_vec) if v > 0)
    p_set = set(i for i, v in enumerate(pred_vec) if v > 0)
    iou = len(g_set & p_set) / len(g_set | p_set) if (g_set | p_set) else 1.0

    g_unique = set(gold_vec)
    p_unique = set(pred_vec)
    if len(g_unique) == 1 or len(p_unique) == 1:
        cor = 1.0 if g_unique == p_unique else 0.0
    else:
        cor = spearmanr(gold_vec, pred_vec).correlation
        if math.isnan(cor):
            cor = 0.0

    return iou, cor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading Base Model: {BASE_MODEL_ID}...")

    try:
        from unsloth import FastVisionModel
        model, processor = FastVisionModel.from_pretrained(
            BASE_MODEL_ID,
            load_in_4bit=False,
            load_in_16bit=True,
        )
        FastVisionModel.for_inference(model)
    except Exception as e:
        print(f"Unsloth load fallback: {e}")
        from transformers import AutoProcessor, AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        model.eval()

    summary_rows = []

    for lang in LANGUAGES:
        print(f"\n============================================================")
        print(f"  Evaluating Base Qwen-VL Zero-Shot: {lang.upper()}")
        print(f"============================================================")

        all_samples = load_dataset(DATA_DIR, lang)
        eval_samples = split_data(all_samples, eval_ratio=0.10, seed=SEED)
        print(f"Eval samples: {len(eval_samples)}")

        ious, cors = [], []

        for sample in tqdm(eval_samples, desc=f"Qwen Zero-Shot {lang.upper()}"):
            prompt = sample.get("prompt", "")
            response = sample.get("response", "")
            img_name = sample.get("image_name", "")
            img_path = find_image(img_name, IMAGES_DIR)

            if not img_path or not img_path.exists():
                img = Image.new("RGB", (224, 224), (255, 255, 255))
            else:
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224), (255, 255, 255))

            user_prompt = (
                f"Image Question: {prompt}\n"
                f"Generated Response: {response}\n\n"
                "Task: Identify any hallucinated spans (inventions, mischaracterizations, OCR errors, miscounting) in the response.\n"
                "Return a JSON list of spans with character start, end, label, and probability score, e.g. "
                '[{"start": 10, "end": 25, "label": "invention", "prob": 0.9}]. If no hallucination, return [].'
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]

            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_input], images=[img], padding=True, return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            pred_spans = parse_spans_from_text(output_text, len(response))
            iou, cor = compute_sample_scores(sample, pred_spans)
            ious.append(iou)
            cors.append(cor)

        mean_iou = float(np.mean(ious))
        mean_cor = float(np.mean(cors))

        print(f"  [{lang.upper()}] Base Qwen-VL Zero-Shot -> Mean IoU: {mean_iou:.4f} | Spearman Cor: {mean_cor:.4f}")
        summary_rows.append({
            "language": lang.upper(),
            "model": "Qwen2.5-VL-3B (Base Zero-Shot)",
            "eval_samples": len(eval_samples),
            "mean_iou": round(mean_iou, 4),
            "spearman_cor": round(mean_cor, 4),
        })

    print("\n============================================================")
    print("  SUMMARY: Base Qwen-VL Zero-Shot Baseline Performance")
    print("============================================================")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
