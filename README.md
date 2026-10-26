# SHROOM-Visions 2026: Multi-lingual Span Calibration & Hallucination Detection in Vision-Language Models

This repository contains the complete research and engineering codebase for hallucination detection and span confidence calibration in Vision-Language Models (VLMs), developed for the [SHROOM-Visions 2026 Shared Task](https://helsinki-nlp.github.io/shroom/2026).

---

## 📌 Project Overview

Vision-Language Models frequently suffer from hallucinations—generating text that misrepresents, invents, or misinterprets visual content. This project provides an end-to-end multi-lingual pipeline (**English, French, Italian, Chinese**) designed to:
1. **Locate Hallucinated Spans**: Character-level span identification across 4 hallucination categories:
   - **Invention**: Fabricating entities/details not present in the image.
   - **Mischaracterization**: Inaccurately describing visible visual attributes.
   - **OCR Problems**: Incorrect text extraction from images.
   - **Miscounting**: Error in quantity or numerical counting.
2. **Calibrate Confidence**: Predicting per-character hallucination probabilities aligned with empirical multi-annotator distributions.

---

## 🛠️ Architecture & Key Components

### 1. `spancalib_vlm/` Package Module
Custom framework for end-to-end span detection and calibration:
- `dataset.py`: Multi-lingual PyTorch dataset loader with character-level span target alignment and category mapping.
- `model.py`: Model architecture wrapper supporting token/character classification heads.
- `train.py`: Supervised fine-tuning loop with multi-task loss (IoU loss + Calibration loss).
- `evaluate.py`: Evaluation suite computing Character IoU and Pearson Calibration correlation.
- `ensemble.py`: Multi-model output fusion and post-processing.
- `postprocess.py`: Span boundary smoothing and probability thresholding.

### 2. Baseline Evaluations
- `run_qwen_baseline_eval.py`: Zero-shot evaluation using **Qwen3.5-4B** / **Qwen3-VL-2B-Thinking**.
- `run_hallushift_eval.py`: Feature extraction and probing classifier (MLP/RandomForest/LogisticRegression) on LLaVA-1.5-7B internal activations.
- `evaluate_blip.py` & `evaluate_minicpm.py`: Evaluation scripts tailored for BLIP-VQA and MiniCPM-V models.

### 3. Supervised Fine-Tuning (SFT) & RL Pipelines
- `run_sft_multilingual.sh`: Multi-lingual SFT training across English, French, Italian, and Chinese data.
- `finetune.py`, `finetune_blip.py`, `finetune_minicpm.py`: Model-specific SFT training routines.
- `rl_grpo.py` & `run_rl.sh`: Reinforcement Learning with **Group Relative Policy Optimization (GRPO)** for span scoring alignment.

### 4. Prediction & Submission Generation
- `run_tagger_and_predict.sh`: Execution pipeline to train taggers and infer predictions on unlabelled test sets.
- `generate_submission.py`: Validates and packages test set predictions into benchmark-compliant JSONL format.

### 5. Research Paper (`paper/`)
- `shroom_spancalib_vlm.tex`: Complete LaTeX source for the paper.
- `references.bib`: Citation database.
- `shroom_spancalib_vlm.pdf`: Pre-compiled paper manuscript.

---

## 📁 Repository Structure

```
├── README.md                           # Project documentation
├── requirements.txt                    # Core dependencies
├── requirements_finetune.txt           # Fine-tuning dependencies
├── requirements_vqa.txt                # VQA model dependencies
├── setup.sh                            # GPU environment initialization
├── .gitignore                          # Git ignore rules
│
├── spancalib_vlm/                      # Span Calibration VLM package
│   ├── __init__.py
│   ├── dataset.py                      # Dataset parsing & category mapping
│   ├── model.py                        # Model architecture & heads
│   ├── train.py                        # Multi-task SFT training
│   ├── evaluate.py                     # Evaluation metrics (IoU, Pearson r)
│   ├── ensemble.py                     # Multi-model ensembling
│   ├── postprocess.py                  # Span extraction postprocessing
│   └── run.sh                          # Execution entrypoint
│
├── paper/                              # Paper manuscript source & PDF
│   ├── shroom_spancalib_vlm.tex        # Paper LaTeX source
│   ├── references.bib                  # BibTeX bibliography
│   ├── acl.sty & acl_natbib.bst        # ACL styling files
│   └── shroom_spancalib_vlm.pdf        # Compiled PDF manuscript
│
├── Baseline & Evaluation Scripts
│   ├── run_qwen_baseline_eval.py       # Qwen3.5 baseline evaluator
│   ├── run_hallushift_eval.py          # HalluShift++ LLaVA probing evaluator
│   ├── evaluate.py                     # Main evaluation script
│   ├── evaluate_blip.py                # BLIP evaluation script
│   └── evaluate_minicpm.py             # MiniCPM evaluation script
│
├── Training & RL Scripts
│   ├── run_sft_multilingual.sh         # Multilingual SFT launcher
│   ├── run_sft.sh & run_blip_sft.sh    # SFT execution shell scripts
│   ├── finetune.py                     # Core fine-tuning routine
│   ├── finetune_blip.py                # BLIP fine-tuning
│   ├── finetune_minicpm.py             # MiniCPM fine-tuning
│   ├── rl_grpo.py                      # GRPO Reinforcement Learning
│   └── run_rl.sh                       # RL launcher script
│
└── Submission Scripts
    ├── run_tagger_and_predict.sh       # Tagger train & predict pipeline
    └── generate_submission.py          # Submission generator
```

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/Aman-byte1/Hallucination-Detection-in-LVLMs.git
cd Hallucination-Detection-in-LVLMs

# Set up virtual environment and install requirements
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Multi-Lingual SFT Training
```bash
bash run_sft_multilingual.sh
```

### 3. Evaluate Baseline Models
```bash
# Evaluate Qwen3.5 Baseline on English test set
python run_qwen_baseline_eval.py --lang en

# Run HalluShift++ Probing Evaluator
python run_hallushift_eval.py
```

### 4. Generate Test Predictions & Submissions
```bash
bash run_tagger_and_predict.sh
python generate_submission.py
```

---

## 📜 Citation & Paper

If you use this codebase or reference our work, please cite the paper included in https://arxiv.org/abs/2608.29974:
```
