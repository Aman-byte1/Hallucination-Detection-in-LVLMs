# SHROOM-Visions: Hallucination Detection in Vision-Language Models

Evaluation pipeline for the [SHROOM-Visions 2026](https://helsinki-nlp.github.io/shroom/2026) shared task using the **Qwen3-VL-2B-Thinking** model.

## Task

Detect hallucinated spans in VLM-generated responses and classify them into four categories:
- **Invention**: Fabricated information with no basis
- **Mischaracterization**: Inaccurate description of something real
- **OCR Problems**: Incorrect text reading from images
- **Miscounting**: Wrong object/item counts

## Metrics

- **Span Identification (IoU)**: Character-level Intersection-over-Union between predicted and gold hallucination spans
- **Confidence Calibration**: Pearson correlation between predicted per-character hallucination probability and gold empirical probability from multi-annotator data

## Quick Start

```bash
# Clone the repo
git clone <your-repo-url>
cd Shroom

# Setup environment (on GPU server)
chmod +x setup.sh
./setup.sh

# Activate environment
source venv/bin/activate

# Quick test (10 samples)
python evaluate.py --max_samples 10

# Full evaluation
python evaluate.py

# Resume if interrupted
python evaluate.py --resume
```

## Outputs

After evaluation, results are saved to `outputs/`:

| File | Description |
|------|-------------|
| `predictions_en.csv` | Per-sample predictions with IoU and calibration scores |
| `predictions_en.jsonl` | Full predictions in JSONL format |
| `metrics_en.json` | Aggregated metrics (overall, per-category, detection stats) |

## Project Structure

```
├── evaluate.py          # Main evaluation script (all-in-one)
├── explore.py           # Data exploration script
├── setup.sh             # Environment setup for GPU server
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── shroom-visions-data/ # Dataset (not tracked by git)
    └── distrib/
        ├── shroom-vision.train.en.labeled.jsonl
        └── shroom-vision.test.en.unlabeled.jsonl
```

## Model

[Qwen3-VL-2B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking) — A 2B parameter vision-language model with enhanced reasoning capabilities.

## Hardware

Designed for NVIDIA A40 (48GB VRAM). The model uses ~4GB in FP16/BF16.
