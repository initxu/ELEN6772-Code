"""Run zero-shot LLM classification across multiple models and sizes."""

import os
import sys
import json
import time
import gc
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.llm_all_methods import (
    load_model, classify_batch, compute_metrics, parse_label,
    build_category_description, get_stratified_sample,
    format_nslkdd_record, format_unsw_record,
)
from src.data_loader import load_nslkdd, COLUMNS, ATTACK_MAP, DATA_DIR
from src.data_loader_unsw import load_unsw_nb15

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# Models to test: (hf_id, short_name, size_category)
MODELS_SIZE_COMPARISON = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B", "1.5B"),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B", "3B"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B", "7B"),
    ("Qwen/Qwen2.5-14B-Instruct", "Qwen2.5-14B", "14B"),
]

MODELS_FAMILY_COMPARISON = [
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B", "Qwen"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B", "Llama"),
    ("deepseek-ai/deepseek-llm-7b-chat", "DeepSeek-7B", "DeepSeek"),
]


def run_zero_shot_with_model(model_name, short_name, raw_test_df, y_test,
                              label_names, format_fn, dataset_name, sample_indices):
    """Run zero-shot classification with a specific model."""
    cat_desc = build_category_description(dataset_name)
    prompt_template = f"""Classify this network connection record into exactly one category.

{cat_desc}

Reply with ONLY the category label, nothing else.

Input: {{record}}
Output:"""

    try:
        model, tokenizer = load_model(model_name)
    except Exception as e:
        print(f"  ERROR loading {model_name}: {e}")
        return None

    device = next(model.parameters()).device
    records = [format_fn(raw_test_df.iloc[idx]) for idx in sample_indices]
    y_sub = y_test[sample_indices]

    print(f"\n  Zero-shot ({short_name}): classifying {len(records)} samples...")
    preds, times = classify_batch(model, tokenizer, records, prompt_template, label_names, device)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    y_pred = np.array([label_names.index(p) if p in label_names else 0 for p in preds])
    result = compute_metrics(f"ZS-{short_name}", y_sub, y_pred, label_names, times, 0.0)
    return result


def run_comparison(models_list, comparison_name, datasets_config):
    """Run zero-shot comparison across multiple models on all datasets."""
    all_results = {}

    for ds_name, ds_config in datasets_config.items():
        print(f"\n{'='*60}")
        print(f"{comparison_name} — {ds_name}")
        print(f"{'='*60}")

        ds_results = []
        for model_id, short_name, category in models_list:
            print(f"\n--- {short_name} ({category}) ---")
            result = run_zero_shot_with_model(
                model_id, short_name,
                ds_config["raw_test"], ds_config["y_test"],
                ds_config["label_names"], ds_config["format_fn"],
                ds_name, ds_config["sample_indices"],
            )
            if result:
                result["model_id"] = model_id
                result["category"] = category
                ds_results.append(result)

                # Save incrementally
                out_path = os.path.join(RESULTS_DIR, f"{comparison_name.lower().replace(' ', '_')}_{ds_name.lower().replace('-', '_')}.json")
                with open(out_path, "w") as f:
                    json.dump(ds_results, f, indent=2)

        all_results[ds_name] = ds_results

    return all_results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load datasets
    print("Loading NSL-KDD...")
    X_train_nsl, X_test_nsl, y_train_nsl, y_test_nsl, _, nsl_labels, _, raw_test_nsl = load_nslkdd()
    nsl_indices = get_stratified_sample(y_test_nsl, nsl_labels, 500)

    print("Loading UNSW-NB15...")
    X_train_unsw, X_test_unsw, y_train_unsw, y_test_unsw, _, unsw_labels, _, raw_test_unsw = load_unsw_nb15()
    unsw_indices = get_stratified_sample(y_test_unsw, unsw_labels, 500)

    datasets_config = {
        "NSL-KDD": {
            "raw_test": raw_test_nsl,
            "y_test": y_test_nsl,
            "label_names": nsl_labels,
            "format_fn": format_nslkdd_record,
            "sample_indices": nsl_indices,
        },
        "UNSW-NB15": {
            "raw_test": raw_test_unsw,
            "y_test": y_test_unsw,
            "label_names": unsw_labels,
            "format_fn": format_unsw_record,
            "sample_indices": unsw_indices,
        },
    }

    # 1. Model size comparison (Qwen2.5 family)
    print("\n" + "#" * 70)
    print("# MODEL SIZE COMPARISON (Qwen2.5)")
    print("#" * 70)
    size_results = run_comparison(MODELS_SIZE_COMPARISON, "size_comparison", datasets_config)

    # 2. Model family comparison (~7B)
    print("\n" + "#" * 70)
    print("# MODEL FAMILY COMPARISON (~7B)")
    print("#" * 70)
    family_results = run_comparison(MODELS_FAMILY_COMPARISON, "family_comparison", datasets_config)

    # Save combined
    combined = {"size_comparison": size_results, "family_comparison": family_results}
    with open(os.path.join(RESULTS_DIR, "llm_multi_model_results.json"), "w") as f:
        json.dump(combined, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL MULTI-MODEL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
