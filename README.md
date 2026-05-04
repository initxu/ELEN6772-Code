# Beyond Classification Accuracy: Investigating the Potential of Large Language Models for Explainable Network Intrusion Detection

Source code and experimental artifacts for the ELEN6772 course paper of the same
title. This package contains everything needed to reproduce the numbers in the
paper's tables and figure.

- Authors: Lilin Xu (lx2331), Yuang Fan (yf2676)
- Datasets: NSL-KDD (5-class), UNSW-NB15 (10-class)
- Methods compared: traditional ML, deep learning, and LLM (zero-shot,
  few-shot, LoRA fine-tuned, plus an explainability experiment)

## Repository layout

```
code/
├── README.md                  this file
├── requirements.txt           Python dependencies
├── data/                      datasets go here (see below)
├── src/                       all experiment code
└── results/                   raw JSON outputs + LoRA adapter checkpoints
```

The compiled LaTeX tables and PDF figure are kept with the paper sources, not
here. The scripts in `src/` that build them write to `tables/` and `figures/`
directories alongside this README, which are created on demand.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10 / CUDA 12.x on two NVIDIA RTX PRO 6000 Blackwell GPUs.
LLM fine-tuning of Qwen2.5-32B / DeepSeek-7B requires ≥48 GB VRAM; smaller
models fit on a single 24 GB GPU.

## Datasets

NSL-KDD is fetched automatically from
[`defcom17/NSL_KDD`](https://github.com/defcom17/NSL_KDD) the first time
`src/data_loader.py` is imported. UNSW-NB15 must be downloaded manually from
the [UNSW dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
and placed under `data/`:

```
data/
├── KDDTrain+.txt              auto-downloaded
├── KDDTest+.txt               auto-downloaded
├── UNSW_NB15_training-set.csv  manual
└── UNSW_NB15_testing-set.csv   manual
```

## Reproducing the paper

The paper's tables and figure draw from these JSONs:

| Paper artifact | Built by | Reads from |
|---|---|---|
| Table 1 (NSL-KDD results) | hand-curated from JSONs | `results/nsl_kdd/{ml,cnn,dl_new,llm}_results.json` |
| Table 2 (UNSW-NB15 results) | hand-curated from JSONs | `results/unsw_nb15/{ml,cnn,dl_new,llm}_results.json` |
| Table 3 (LoRA FT comparison) | `src/generate_paper_figures.py` | `results/multi_model_ft_*.json`, `results/ft_qwen25_32b_*.json`, `results/ft_deepseek_7b_*.json` |
| Table 4 (explainability summary) | `src/generate_explainability_table.py` | `results/explainability_*.json` |
| Table 5 (explainability examples) | `src/generate_explainability_table.py` | `results/explainability_finetuned_nsl_kdd.json` |
| Figure 1 (family comparison) | `src/generate_paper_figures.py` | `results/family_comparison_*.json`, `results/mistral_results.json`, `results/multi_model_ft_*.json` |

You can regenerate the auto-built tables and the figure straight from the
shipped JSONs (no training required):

```bash
python src/generate_paper_figures.py        # → ft_comparison.tex, family_comparison.{pdf,png}
python src/generate_explainability_table.py # → explain_{summary,examples}.tex
```

### Re-running experiments from scratch

End-to-end runner for ML + 1D-CNN + LLM (ZS/FS/LoRA-Qwen-3B) on both datasets:

```bash
python src/run_all_v2.py
```

Extra DL models (MLP, BiLSTM, 1D-ResNet) for the DL rows of Tables 1–2:

```bash
python src/run_dl_models.py
```

LoRA fine-tuning sweep across Qwen2.5-1.5B/3B/7B and Mistral-7B (Table 3 rows):

```bash
python src/llm_finetune_multi.py
```

Qwen2.5-32B and DeepSeek-7B fine-tuning (the heavier rows in Table 3):

```bash
python src/llm_finetune_remaining.py --model qwen32b
python src/llm_finetune_remaining.py --model deepseek7b
```

Zero-shot family comparison (Qwen2.5-7B, Llama-3.1-8B, DeepSeek-7B-Chat) and
the matched Mistral-7B zero-shot run that feeds Figure 1:

```bash
python src/llm_multi_model.py
python src/llm_mistral.py
```

Explainability experiment (Section 4.6 / Tables 4–5):

```bash
# Fine-tuned Qwen2.5-7B with explanation prompt
python src/llm_explainability.py --dataset nsl_kdd   --mode finetuned --adapter_name lora_ft_qwen25_7b_nsl_kdd
python src/llm_explainability.py --dataset unsw_nb15 --mode finetuned --adapter_name lora_ft_qwen25_7b_unsw_nb15

# Zero-shot baseline
python src/llm_explainability.py --dataset nsl_kdd --mode zeroshot
```

## Source files

| File | Purpose |
|---|---|
| `data_loader.py` | NSL-KDD load / preprocess / text serialization |
| `data_loader_unsw.py` | UNSW-NB15 load / preprocess / text serialization |
| `traditional_ml.py` | Logistic Regression, SVM (RBF), Random Forest, XGBoost |
| `cnn_model.py` | 1D-CNN (the DL row in Tables 1–2) |
| `dl_models.py` | MLP, BiLSTM, 1D-ResNet |
| `run_dl_models.py` | Driver that runs the three extra DL models |
| `llm_all_methods.py` | Shared LLM utilities: prompt templates, parse, ZS / FS / LoRA |
| `llm_multi_model.py` | Zero-shot sweep across Qwen / Llama / DeepSeek |
| `llm_mistral.py` | Mistral-7B zero-shot (used in Figure 1) |
| `llm_finetune_multi.py` | LoRA fine-tuning sweep across Qwen sizes + Mistral |
| `llm_finetune_remaining.py` | LoRA / QLoRA for Qwen2.5-32B and DeepSeek-7B |
| `llm_explainability.py` | Explanation-eliciting prompts on stratified samples |
| `generate_explainability_table.py` | Rebuilds Tables 4–5 from `explainability_*.json` |
| `generate_paper_figures.py` | Rebuilds Table 3 and Figure 1 |
| `run_all_v2.py` | End-to-end runner: ML + CNN + LLM (ZS/FS/FT) on both datasets |

## Results layout

```
results/
├── nsl_kdd/  unsw_nb15/        per-dataset ML / CNN / extra-DL / LLM JSONs
├── multi_model_ft_*.json       Table 3 rows (Qwen 1.5B/3B/7B + Mistral 7B)
├── ft_qwen25_32b_*.json        Table 3 row for Qwen2.5-32B
├── ft_deepseek_7b_*.json       Table 3 row for DeepSeek-7B
├── multi_model_zs_*.json       Zero-shot sweep across Qwen sizes
├── family_comparison_*.json    Zero-shot Qwen / Llama / DeepSeek (Figure 1)
├── mistral_results.json        Zero-shot Mistral-7B (Figure 1)
├── explainability_*.json       Section 4.6 raw outputs (Tables 4–5)
└── lora_*/                     LoRA adapter checkpoints for each FT run
```

The `lora_*` folders are PEFT-format LoRA adapters; load them on top of the
matching base model via `peft.PeftModel.from_pretrained(base_model, adapter_dir)`.
The naming is `lora_ft_<model>_<dataset>` except for the Qwen2.5-3B adapters,
which are stored under the legacy `lora_<dataset>` paths to preserve
backward-compatible references in the runner scripts.
