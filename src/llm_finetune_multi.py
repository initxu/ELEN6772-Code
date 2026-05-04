"""Multi-model LoRA fine-tuning and evaluation for NIDS comparison."""

import os
import sys
import json
import time
import gc
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.llm_all_methods import (
    load_model, classify_batch, compute_metrics,
    build_category_description, get_stratified_sample,
    format_nslkdd_record, format_unsw_record,
    finetune_lora,
)
from src.data_loader import load_nslkdd, COLUMNS, ATTACK_MAP, DATA_DIR
from src.data_loader_unsw import load_unsw_nb15

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# Models to fine-tune: (hf_id, short_name, family, size_label)
FINETUNE_MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B", "Qwen", "1.5B"),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B", "Qwen", "3B"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B", "Qwen", "7B"),
    ("Qwen/Qwen2.5-32B-Instruct", "Qwen2.5-32B", "Qwen", "32B"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B", "Llama", "8B"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B", "Mistral", "7B"),
]


def get_adapter_dir(short_name, dataset_name):
    """Get the output directory for a LoRA adapter."""
    tag = short_name.lower().replace(".", "").replace("-", "_")
    ds_tag = dataset_name.lower().replace("-", "_")
    # Reuse existing dir for Qwen 3B
    if short_name == "Qwen2.5-3B":
        return os.path.join(RESULTS_DIR, f"lora_{ds_tag}")
    return os.path.join(RESULTS_DIR, f"lora_ft_{tag}_{ds_tag}")


def run_zero_shot_eval(model_id, short_name, family, size_label,
                       raw_test_df, y_test, label_names,
                       format_fn, dataset_name, sample_indices):
    """Run zero-shot evaluation for a model (needed for models without existing ZS results)."""
    from src.llm_all_methods import run_zero_shot

    print(f"\n{'='*60}")
    print(f"ZERO-SHOT: {short_name} ({family}, {size_label}) on {dataset_name}")
    print(f"{'='*60}")

    try:
        model, tokenizer = load_model(model_id)
    except Exception as e:
        print(f"  ERROR loading {short_name}: {e}")
        return None

    device = next(model.parameters()).device
    result = run_zero_shot(model, tokenizer, raw_test_df, y_test, label_names,
                           format_fn, dataset_name, sample_indices, device)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    result["model"] = f"ZS-{short_name}"
    result["model_id"] = model_id
    result["family"] = family
    result["size_label"] = size_label

    return result


def finetune_and_eval(model_id, short_name, family, size_label,
                      raw_train_df, raw_test_df, y_test, label_names,
                      format_fn, dataset_name, category_col, sample_indices):
    """Fine-tune a model with LoRA and evaluate it."""
    adapter_dir = get_adapter_dir(short_name, dataset_name)

    print(f"\n{'='*60}")
    print(f"FINE-TUNE: {short_name} ({family}, {size_label}) on {dataset_name}")
    print(f"Adapter dir: {adapter_dir}")
    print(f"{'='*60}")

    # Fine-tune
    t0 = time.time()
    try:
        adapter_path = finetune_lora(
            raw_train_df, label_names, format_fn, category_col,
            model_name=model_id, dataset_name=dataset_name,
            n_train=5000, output_dir=adapter_dir,
        )
    except Exception as e:
        print(f"  ERROR fine-tuning {short_name}: {e}")
        import traceback
        traceback.print_exc()
        return None
    ft_train_time = time.time() - t0

    # Evaluate
    cat_desc = build_category_description(dataset_name)
    prompt_template = f"""Classify this network connection.
{cat_desc}
Input: {{record}}
Output:"""

    try:
        model, tokenizer = load_model(model_id, adapter_path)
    except Exception as e:
        print(f"  ERROR loading fine-tuned {short_name}: {e}")
        return None

    device = next(model.parameters()).device
    records = [format_fn(raw_test_df.iloc[idx]) for idx in sample_indices]
    y_sub = y_test[sample_indices]

    print(f"\n  Evaluating FT-{short_name}: {len(records)} samples...")
    preds, times = classify_batch(model, tokenizer, records, prompt_template, label_names, device)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    y_pred = np.array([label_names.index(p) if p in label_names else 0 for p in preds])
    result = compute_metrics(f"FT-{short_name}", y_sub, y_pred, label_names, times, ft_train_time)
    result["model_id"] = model_id
    result["family"] = family
    result["size_label"] = size_label
    result["adapter_dir"] = adapter_dir

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-model LoRA fine-tuning for NIDS")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Short names of models to run (e.g., Qwen2.5-1.5B Llama-3.1-8B). Default: all")
    parser.add_argument("--datasets", nargs="*", default=["NSL-KDD", "UNSW-NB15"],
                        help="Datasets to use. Default: NSL-KDD UNSW-NB15")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate existing adapters, skip training")
    parser.add_argument("--run-zs", action="store_true",
                        help="Also run zero-shot evaluation before fine-tuning")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Filter models
    if args.models:
        models = [m for m in FINETUNE_MODELS if m[1] in args.models]
        if not models:
            print(f"No matching models found for: {args.models}")
            print(f"Available: {[m[1] for m in FINETUNE_MODELS]}")
            return
    else:
        models = FINETUNE_MODELS

    # Load datasets
    datasets_config = {}

    if "NSL-KDD" in args.datasets:
        print("Loading NSL-KDD...")
        X_train_nsl, X_test_nsl, y_train_nsl, y_test_nsl, _, nsl_labels, _, raw_test_nsl = load_nslkdd()
        nsl_indices = get_stratified_sample(y_test_nsl, nsl_labels, 500)

        raw_train_nsl = pd.read_csv(os.path.join(DATA_DIR, "KDDTrain+.txt"), header=None, names=COLUMNS)
        raw_train_nsl["category"] = raw_train_nsl["label"].map(ATTACK_MAP).fillna("unknown")
        raw_train_nsl = raw_train_nsl[raw_train_nsl["category"] != "unknown"]

        datasets_config["NSL-KDD"] = {
            "raw_train": raw_train_nsl,
            "raw_test": raw_test_nsl,
            "y_test": y_test_nsl,
            "label_names": nsl_labels,
            "format_fn": format_nslkdd_record,
            "category_col": "category",
            "sample_indices": nsl_indices,
        }

    if "UNSW-NB15" in args.datasets:
        print("Loading UNSW-NB15...")
        X_train_unsw, X_test_unsw, y_train_unsw, y_test_unsw, _, unsw_labels, _, raw_test_unsw = load_unsw_nb15()
        unsw_indices = get_stratified_sample(y_test_unsw, unsw_labels, 500)

        raw_train_unsw = pd.read_csv(os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv"))
        raw_train_unsw["attack_cat"] = raw_train_unsw["attack_cat"].fillna("Normal").str.strip()
        raw_train_unsw.loc[raw_train_unsw["attack_cat"] == "", "attack_cat"] = "Normal"
        name_fixes = {"Backdoors": "Backdoor", " Fuzzers": "Fuzzers", " Reconnaissance": "Reconnaissance"}
        raw_train_unsw["attack_cat"] = raw_train_unsw["attack_cat"].replace(name_fixes)

        datasets_config["UNSW-NB15"] = {
            "raw_train": raw_train_unsw,
            "raw_test": raw_test_unsw,
            "y_test": y_test_unsw,
            "label_names": unsw_labels,
            "format_fn": format_unsw_record,
            "category_col": "attack_cat",
            "sample_indices": unsw_indices,
        }

    # Run experiments
    all_results = {}
    start = time.time()

    for ds_name, ds_cfg in datasets_config.items():
        print(f"\n{'#'*70}")
        print(f"# DATASET: {ds_name}")
        print(f"{'#'*70}")

        ds_results = []
        zs_results = []
        for model_id, short_name, family, size_label in models:
            # Optional zero-shot evaluation
            if args.run_zs:
                zs_result = run_zero_shot_eval(
                    model_id, short_name, family, size_label,
                    ds_cfg["raw_test"], ds_cfg["y_test"],
                    ds_cfg["label_names"], ds_cfg["format_fn"],
                    ds_name, ds_cfg["sample_indices"],
                )
                if zs_result:
                    zs_results.append(zs_result)
                    zs_path = os.path.join(RESULTS_DIR, f"multi_model_zs_{ds_name.lower().replace('-', '_')}.json")
                    with open(zs_path, "w") as f:
                        json.dump(zs_results, f, indent=2)

            # Fine-tune and evaluate
            result = finetune_and_eval(
                model_id, short_name, family, size_label,
                ds_cfg["raw_train"], ds_cfg["raw_test"],
                ds_cfg["y_test"], ds_cfg["label_names"],
                ds_cfg["format_fn"], ds_name, ds_cfg["category_col"],
                ds_cfg["sample_indices"],
            )
            if result:
                ds_results.append(result)

                # Save incrementally
                out_path = os.path.join(RESULTS_DIR, f"multi_model_ft_{ds_name.lower().replace('-', '_')}.json")
                with open(out_path, "w") as f:
                    json.dump(ds_results, f, indent=2)

        all_results[ds_name] = ds_results

    # Save combined results
    with open(os.path.join(RESULTS_DIR, "multi_model_ft_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - start

    # Print summary
    print(f"\n{'='*70}")
    print(f"MULTI-MODEL FINE-TUNING COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")
    print(f"\nResults saved to: {RESULTS_DIR}/multi_model_ft_results.json")

    print("\nSummary:")
    print(f"{'Model':<25} {'Dataset':<12} {'F1 (macro)':<12} {'Accuracy':<12} {'Train (s)':<12}")
    print("-" * 73)
    for ds_name, ds_results in all_results.items():
        for r in ds_results:
            print(f"{r['model']:<25} {ds_name:<12} {r['f1_macro']:<12.4f} {r['accuracy']:<12.4f} {r['train_time_s']:<12.1f}")


if __name__ == "__main__":
    main()
