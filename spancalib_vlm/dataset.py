"""
SpanCalib-VLM Dataset Module
=============================
Handles tokenization and precise character-to-token offset mapping
for token-level hallucination classification and calibration.
"""

import json
import logging
import re
import urllib.parse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image

logger = logging.getLogger(__name__)

CATEGORY_MAP = {
    "none": 0,
    "invention": 1,
    "mischaracterization": 2,
    "OCR": 3,
    "miscounting": 4,
}

REVERSE_CATEGORY_MAP = {v: k for k, v in CATEGORY_MAP.items()}


def find_image(image_name: str, images_dir: Path) -> Path | None:
    """Find image file, handling URL decoding and nested subdirectories."""
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


def map_char_spans_to_tokens(
    labels: list[dict],
    response_text: str,
    offset_mapping: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map character-level SHROOM gold labels to token-level targets.

    Args:
        labels: list of gold dicts [{'start': int, 'end': int, 'label': str, 'prob': float}]
        response_text: full response string
        offset_mapping: list of (start_char, end_char) for response tokens

    Returns:
        binary_mask: np.ndarray [seq_len] (1 if token inside hallucination, else 0)
        prob_targets: np.ndarray [seq_len] (annotator probability for token, else 0.0)
        category_targets: np.ndarray [seq_len] (category class index 0..4)
    """
    n_tokens = len(offset_mapping)
    binary_mask = np.zeros(n_tokens, dtype=np.float32)
    prob_targets = np.zeros(n_tokens, dtype=np.float32)
    category_targets = np.zeros(n_tokens, dtype=np.int64)

    if not labels:
        return binary_mask, prob_targets, category_targets

    # Build character level maps
    resp_len = len(response_text)
    char_binary = np.zeros(resp_len, dtype=np.float32)
    char_prob = np.zeros(resp_len, dtype=np.float32)
    char_cat = np.zeros(resp_len, dtype=np.int64)

    for gold in labels:
        s = max(0, gold["start"])
        e = min(resp_len, gold["end"])
        p = float(gold.get("prob", 1.0))
        cat_name = gold.get("label", "invention")
        cat_idx = CATEGORY_MAP.get(cat_name, 1)

        char_binary[s:e] = 1.0
        char_prob[s:e] = np.maximum(char_prob[s:e], p)
        char_cat[s:e] = cat_idx

    # Map character coverage to tokens
    for i, (t_start, t_end) in enumerate(offset_mapping):
        if t_start >= t_end or t_start >= resp_len:
            continue
        t_end_clamped = min(resp_len, t_end)
        tok_slice_binary = char_binary[t_start:t_end_clamped]
        
        # Token is considered hallucinated if > 30% of its characters fall in a span
        if len(tok_slice_binary) > 0 and np.mean(tok_slice_binary) >= 0.3:
            binary_mask[i] = 1.0
            prob_targets[i] = np.max(char_prob[t_start:t_end_clamped])
            category_targets[i] = char_cat[t_start] if char_cat[t_start] > 0 else 1

    return binary_mask, prob_targets, category_targets


class SpanCalibDataset(TorchDataset):
    """Dataset for token-level SHROOM hallucination detection and calibration."""

    def __init__(
        self,
        samples: list[dict],
        tokenizer_or_processor,
        images_dir: Path,
        max_length: int = 512,
    ):
        self.samples = samples
        self.tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
        self.processor = tokenizer_or_processor
        self.images_dir = Path(images_dir)
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        response_text = sample["response"]
        prompt_text = sample["prompt"]
        labels = sample.get("labels", [])

        # Format input text prompt for token classification
        formatted_prompt = f"Question: {prompt_text}\nResponse: {response_text}"

        # Tokenize with offsets
        encoded = self.tokenizer(
            formatted_prompt,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        offsets = encoded["offset_mapping"].squeeze(0).tolist()

        # Locate response_text start offset in formatted_prompt
        resp_start_in_prompt = formatted_prompt.find(response_text)
        if resp_start_in_prompt == -1:
            resp_start_in_prompt = 0

        # Adjust offsets relative to response_text
        adjusted_offsets = []
        response_token_mask = np.zeros(len(offsets), dtype=bool)

        for i, (s, e) in enumerate(offsets):
            if s >= resp_start_in_prompt and e > resp_start_in_prompt:
                rel_s = s - resp_start_in_prompt
                rel_e = e - resp_start_in_prompt
                adjusted_offsets.append((rel_s, rel_e))
                response_token_mask[i] = True
            else:
                adjusted_offsets.append((0, 0))

        # Compute ground truth token targets
        bin_targets, prob_targets, cat_targets = map_char_spans_to_tokens(
            labels=labels,
            response_text=response_text,
            offset_mapping=adjusted_offsets,
        )

        return {
            "id": sample["id"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_token_mask": torch.tensor(response_token_mask, dtype=torch.bool),
            "binary_labels": torch.tensor(bin_targets, dtype=torch.float32),
            "prob_labels": torch.tensor(prob_targets, dtype=torch.float32),
            "category_labels": torch.tensor(cat_targets, dtype=torch.long),
            "gold_labels": labels,
            "response_text": response_text,
        }


class DataCollatorForSpanCalib:
    """Collates and dynamically pads SpanCalib batch elements to equal sequence length."""

    def __init__(self, pad_token_id: int = 1):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(x["input_ids"]) for x in batch)

        batch_input_ids = []
        batch_attention_mask = []
        batch_resp_mask = []
        batch_bin_labels = []
        batch_prob_labels = []
        batch_cat_labels = []

        ids_list = []
        gold_labels_list = []
        resp_texts = []

        for item in batch:
            l = len(item["input_ids"])
            pad_len = max_len - l

            # Pad tensors
            padded_ids = torch.cat([item["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            padded_att = torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            padded_resp = torch.cat([item["response_token_mask"], torch.zeros(pad_len, dtype=torch.bool)])
            padded_bin = torch.cat([item["binary_labels"], torch.zeros(pad_len, dtype=torch.float32)])
            padded_prob = torch.cat([item["prob_labels"], torch.zeros(pad_len, dtype=torch.float32)])
            padded_cat = torch.cat([item["category_labels"], torch.zeros(pad_len, dtype=torch.long)])

            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(padded_att)
            batch_resp_mask.append(padded_resp)
            batch_bin_labels.append(padded_bin)
            batch_prob_labels.append(padded_prob)
            batch_cat_labels.append(padded_cat)

            ids_list.append(item["id"])
            gold_labels_list.append(item.get("gold_labels", []))
            resp_texts.append(item.get("response_text", ""))

        return {
            "id": ids_list,
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "attention_mask": torch.stack(batch_attention_mask, dim=0),
            "response_token_mask": torch.stack(batch_resp_mask, dim=0),
            "binary_labels": torch.stack(batch_bin_labels, dim=0),
            "prob_labels": torch.stack(batch_prob_labels, dim=0),
            "category_labels": torch.stack(batch_cat_labels, dim=0),
            "gold_labels": gold_labels_list,
            "response_text": resp_texts,
        }
