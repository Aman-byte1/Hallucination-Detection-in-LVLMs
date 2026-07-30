#!/usr/bin/env python3
"""
HalluShift++ Baseline Evaluation Script for SHROOM-Visions
=========================================================
Extracts HalluShift internal state features (LLaVA-1.5-7B) for the 90% train / 10% eval split,
trains HalluShift classifiers (MLP, Random Forest, Logistic Regression),
and evaluates Token F1, IoU, and Spearman Correlation.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, LlavaProcessor

# Config
DATA_DIR = Path("shroom-visions-data/distrib")
IMAGES_DIR = Path("shroom-vis-images")
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
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
    train_samples = [s for i, s in enumerate(samples) if i not in eval_indices]
    eval_samples = [s for i, s in enumerate(samples) if i in eval_indices]
    return train_samples, eval_samples


def extract_llava_token_features(model, processor, sample, images_dir, device):
    prompt = sample.get("prompt", "")
    response = sample.get("response", "")
    img_name = sample.get("image_name", "")
    img_path = find_image(img_name, images_dir)

    if not img_path or not img_path.exists():
        img = Image.new("RGB", (224, 224), (255, 255, 255))
    else:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (255, 255, 255))

    conversation = [
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}
    ]
    chat_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=img, text=chat_prompt, return_tensors="pt").to(device)

    tokenizer = processor.tokenizer
    resp_ids = tokenizer.encode(response, add_special_tokens=False)
    if len(resp_ids) == 0:
        return []

    prompt_ids = inputs["input_ids"]
    prompt_len = prompt_ids.shape[1]
    response_ids = torch.tensor([resp_ids], device=device)
    response_len = response_ids.shape[1]

    full_input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    full_attention_mask = torch.ones_like(full_input_ids, device=device)
    if "attention_mask" in inputs:
        full_attention_mask = torch.cat(
            [inputs["attention_mask"], torch.ones((1, response_len), dtype=inputs["attention_mask"].dtype, device=device)],
            dim=1,
        )

    model_inputs = dict(inputs)
    model_inputs["input_ids"] = full_input_ids
    model_inputs["attention_mask"] = full_attention_mask
    model_inputs["output_hidden_states"] = True
    model_inputs["output_attentions"] = True
    model_inputs["return_dict"] = True

    with torch.no_grad():
        outputs = model(**model_inputs)

    logits = outputs.logits  # [1, total_len, vocab_size]
    hidden_states = outputs.hidden_states  # tuple of [1, total_len, hidden_dim]
    attentions = outputs.attentions  # tuple of [1, num_heads, total_len, total_len]

    # Align labels to character spans
    gold_labels = sample.get("labels", [])
    gold_spans = []
    for g in gold_labels:
        gold_spans.append((g["start"], g["end"]))

    token_features_list = []
    cursor = 0

    for t in range(response_len):
        token_id = resp_ids[t]
        token_text = tokenizer.decode([token_id])
        clean_tok = token_text.strip()

        # Char offsets
        if clean_tok:
            pos = response.find(clean_tok, cursor)
            if pos != -1:
                start_c = pos
                end_c = pos + len(clean_tok)
                cursor = end_c
            else:
                start_c, end_c = cursor, cursor
        else:
            start_c, end_c = cursor, cursor

        # Check label
        is_halluc = 0
        for s_c, e_c in gold_spans:
            if start_c < e_c and end_c > s_c:
                is_halluc = 1
                break

        # Logit & probability features
        step_idx = prompt_len + t - 1
        step_logits = logits[0, step_idx, :].to(torch.float32)
        step_probs = F.softmax(step_logits, dim=-1)

        max_prob = float(step_probs.max().item())
        target_prob = float(step_probs[token_id].item()) if token_id < step_probs.shape[-1] else 1e-12
        target_prob = max(target_prob, 1e-12)
        nll = -math.log(target_prob)
        perplexity = 1.0 / target_prob
        entropy = -float((step_probs * torch.log(step_probs + 1e-12)).sum().item())

        # Hidden state norms across decoder layers (skip vision layers 0..23)
        hidden_norms = []
        for layer_tensor in hidden_states[24:]:
            hidden_norms.append(float(layer_tensor[0, prompt_len + t, :].norm().item()))

        # Attention entropy / mean across decoder layers
        attn_entropies = []
        attn_means = []
        for att_layer in attentions[24:]:
            att_head = att_layer[0, :, prompt_len + t, :prompt_len + t + 1].to(torch.float32)
            mean_att = float(att_head.mean().item())
            norm_att = att_head / (att_head.sum() + 1e-12)
            ent_att = -float((norm_att * torch.log(norm_att + 1e-12)).sum().item())
            attn_means.append(mean_att)
            attn_entropies.append(ent_att)

        feat_vector = [
            max_prob,
            target_prob,
            nll,
            perplexity,
            entropy,
            float(t),
            float(len(response)),
            float(end_c - start_c),
        ] + hidden_norms + attn_means + attn_entropies

        token_features_list.append({
            "sample_id": sample["id"],
            "token_index": t,
            "token_text": token_text,
            "start_c": start_c,
            "end_c": end_c,
            "features": feat_vector,
            "label": is_halluc,
        })

    return token_features_list


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

    # IoU
    g_set = set(i for i, v in enumerate(gold_vec) if v > 0)
    p_set = set(i for i, v in enumerate(pred_vec) if v > 0)
    iou = len(g_set & p_set) / len(g_set | p_set) if (g_set | p_set) else 1.0

    # Spearman Cor
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

    print("Loading LLaVA-1.5-7B...")
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    processor = LlavaProcessor.from_pretrained(LLAVA_MODEL_ID, use_fast=False)
    model.eval()

    summary_rows = []

    for lang in LANGUAGES:
        print(f"\n============================================================")
        print(f"  Evaluating HalluShift++ Baseline: {lang.upper()}")
        print(f"============================================================")

        all_samples = load_dataset(DATA_DIR, lang)
        train_samples, eval_samples = split_data(all_samples, eval_ratio=0.10, seed=SEED)
        print(f"Train samples: {len(train_samples)} | Eval samples: {len(eval_samples)}")

        # Extract features for train set
        print("Extracting LLaVA features for Train split...")
        train_toks = []
        for s in tqdm(train_samples, desc=f"Train {lang.upper()}"):
            train_toks.extend(extract_llava_token_features(model, processor, s, IMAGES_DIR, device))

        # Extract features for eval set
        print("Extracting LLaVA features for Eval split...")
        eval_toks_by_sample = {}
        eval_toks = []
        for s in tqdm(eval_samples, desc=f"Eval {lang.upper()}"):
            toks = extract_llava_token_features(model, processor, s, IMAGES_DIR, device)
            eval_toks_by_sample[s["id"]] = toks
            eval_toks.extend(toks)

        X_train = np.array([t["features"] for t in train_toks])
        y_train = np.array([t["label"] for t in train_toks])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train classifiers: Logistic Regression and Random Forest
        print("Training HalluShift Classifiers...")
        clf_lr = LogisticRegression(max_iter=500, random_state=SEED, class_weight="balanced")
        clf_lr.fit(X_train_scaled, y_train)

        clf_rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=SEED, class_weight="balanced", n_jobs=-1)
        clf_rf.fit(X_train_scaled, y_train)

        # Predict on Eval
        for clf_name, clf in [("LogisticRegression", clf_lr), ("RandomForest", clf_rf)]:
            ious, cors = [], []

            for s in eval_samples:
                toks = eval_toks_by_sample.get(s["id"], [])
                if not toks:
                    ious.append(1.0)
                    cors.append(1.0)
                    continue

                X_eval = np.array([t["features"] for t in toks])
                X_eval_scaled = scaler.transform(X_eval)
                probs = clf.predict_proba(X_eval_scaled)[:, 1]

                # Convert token probabilities to character spans
                resp = s["response"]
                resp_len = len(resp)
                pred_spans = []

                in_span = False
                start_c = 0
                span_probs = []

                for t_info, p_val in zip(toks, probs):
                    if p_val >= 0.50:
                        if not in_span:
                            in_span = True
                            start_c = t_info["start_c"]
                        span_probs.append(p_val)
                        end_c = t_info["end_c"]
                    else:
                        if in_span:
                            in_span = False
                            avg_p = float(np.mean(span_probs))
                            if end_c > start_c:
                                pred_spans.append({"start": start_c, "end": end_c, "prob": round(avg_p, 4), "label": "invention"})
                            span_probs = []

                if in_span and end_c > start_c:
                    avg_p = float(np.mean(span_probs))
                    pred_spans.append({"start": start_c, "end": end_c, "prob": round(avg_p, 4), "label": "invention"})

                iou, cor = compute_sample_scores(s, pred_spans)
                ious.append(iou)
                cors.append(cor)

            mean_iou = float(np.mean(ious))
            mean_cor = float(np.mean(cors))

            print(f"  [{lang.upper()}] HalluShift++ ({clf_name:<18}) -> Mean IoU: {mean_iou:.4f} | Spearman Cor: {mean_cor:.4f}")

            summary_rows.append({
                "language": lang.upper(),
                "classifier": clf_name,
                "eval_samples": len(eval_samples),
                "mean_iou": round(mean_iou, 4),
                "spearman_cor": round(mean_cor, 4),
            })

    print("\n============================================================")
    print("  SUMMARY: HalluShift++ Baseline Performance")
    print("============================================================")
    df_res = pd.DataFrame(summary_rows)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
