"""Unified experiment runner for both datasets with all methods."""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import load_nslkdd, COLUMNS, ATTACK_MAP, DATA_DIR
from src.data_loader_unsw import load_unsw_nb15
from src.traditional_ml import run_all_ml
from src.cnn_model import train_cnn
from src.llm_all_methods import (
    run_all_llm_methods, format_nslkdd_record, format_unsw_record
)

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def run_dataset(dataset_name, load_fn, raw_train_loader, format_fn, category_col,
                label_names_override=None):
    """Run all experiments on a single dataset."""
    dataset_dir = os.path.join(RESULTS_DIR, dataset_name.lower().replace("-", "_"))
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# DATASET: {dataset_name}")
    print(f"{'#'*70}")

    # Load data
    X_train, X_test, y_train, y_test, feat, label_names, scaler, raw_test = load_fn()

    # Load raw train for LLM
    raw_train = raw_train_loader()

    all_results = []

    # Stage 1: Traditional ML
    print(f"\n{'='*60}")
    print(f"TRADITIONAL ML — {dataset_name}")
    print(f"{'='*60}")
    ml_results = run_all_ml(X_train, X_test, y_train, y_test, label_names)
    all_results.extend(ml_results)
    with open(os.path.join(dataset_dir, "ml_results.json"), "w") as f:
        json.dump(ml_results, f, indent=2)

    # Stage 2: CNN
    print(f"\n{'='*60}")
    print(f"CNN — {dataset_name}")
    print(f"{'='*60}")
    cnn_result = train_cnn(X_train, X_test, y_train, y_test, label_names)
    all_results.append(cnn_result)
    with open(os.path.join(dataset_dir, "cnn_results.json"), "w") as f:
        json.dump(cnn_result, f, indent=2)

    # Stage 3: LLM (zero-shot, few-shot, fine-tuned)
    print(f"\n{'='*60}")
    print(f"LLM METHODS — {dataset_name}")
    print(f"{'='*60}")
    llm_results = run_all_llm_methods(
        raw_train, raw_test, y_test, label_names,
        format_fn=format_fn,
        dataset_name=dataset_name,
        category_col=category_col,
        base_model="Qwen/Qwen2.5-7B-Instruct",
        ft_model="Qwen/Qwen2.5-3B-Instruct",
        n_samples=500,
        n_few_shot=3,
        n_train_ft=5000,
    )
    all_results.extend(llm_results)
    with open(os.path.join(dataset_dir, "llm_results.json"), "w") as f:
        json.dump(llm_results, f, indent=2)

    # Save all results
    with open(os.path.join(dataset_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results, label_names


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    start = time.time()
    combined_results = {}

    # ===== NSL-KDD =====
    def load_nsl_raw_train():
        raw_train = pd.read_csv(os.path.join(DATA_DIR, "KDDTrain+.txt"), header=None, names=COLUMNS)
        raw_train["category"] = raw_train["label"].map(ATTACK_MAP).fillna("unknown")
        raw_train = raw_train[raw_train["category"] != "unknown"]
        return raw_train

    nsl_results, nsl_labels = run_dataset(
        "NSL-KDD", load_nslkdd, load_nsl_raw_train,
        format_nslkdd_record, "category"
    )
    combined_results["NSL-KDD"] = nsl_results

    # ===== UNSW-NB15 =====
    def load_unsw_raw_train():
        raw_train = pd.read_csv(os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv"))
        raw_train["attack_cat"] = raw_train["attack_cat"].fillna("Normal").str.strip()
        raw_train.loc[raw_train["attack_cat"] == "", "attack_cat"] = "Normal"
        name_fixes = {"Backdoors": "Backdoor", " Fuzzers": "Fuzzers", " Reconnaissance": "Reconnaissance"}
        raw_train["attack_cat"] = raw_train["attack_cat"].replace(name_fixes)
        return raw_train

    unsw_results, unsw_labels = run_dataset(
        "UNSW-NB15", load_unsw_nb15, load_unsw_raw_train,
        format_unsw_record, "attack_cat"
    )
    combined_results["UNSW-NB15"] = unsw_results

    # Save combined results
    with open(os.path.join(RESULTS_DIR, "combined_results.json"), "w") as f:
        json.dump(combined_results, f, indent=2)

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
