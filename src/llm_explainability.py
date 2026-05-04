"""LLM Explainability Experiment: Generate classification + explanation outputs.

Runs fine-tuned and zero-shot LLMs with explanation-eliciting prompts on
stratified samples from NSL-KDD and UNSW-NB15, collecting both labels and
natural-language reasoning traces.
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_nslkdd, CATEGORY_LABELS as NSL_LABELS
from data_loader_unsw import load_unsw_nb15

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

from llm_all_methods import (
    format_nslkdd_record, format_unsw_record,
    build_category_description, load_model, parse_label,
)


def build_explain_prompt(record_text, dataset_name):
    """Build a prompt that asks for BOTH a label and an explanation."""
    cat_desc = build_category_description(dataset_name)
    return f"""You are a network security analyst. Analyze the following network connection record and:
1. Classify it into exactly one of the categories below.
2. Explain your reasoning in 1-2 sentences, referencing specific features from the record.

{cat_desc}

Network record: {record_text}

Respond in this format:
Category: <label>
Reasoning: <your explanation>"""


def classify_with_explanation(model, tokenizer, records, dataset_name, label_names, device="cuda"):
    """Classify records and collect full-text explanations."""
    results = []
    for i, record_text in enumerate(records):
        prompt = build_explain_prompt(record_text, dataset_name)
        messages = [
            {"role": "system", "content": "You are a network security analyst. Classify the network record and explain your reasoning."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Parse label from output
        parsed_label = extract_label_from_explanation(raw_output, label_names)

        results.append({
            "sample_idx": i,
            "raw_output": raw_output,
            "parsed_label": parsed_label,
            "inference_time_s": elapsed,
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(records)} samples...")

    return results


def extract_label_from_explanation(raw_output, label_names):
    """Extract label from explanation-style output."""
    lines = raw_output.strip().split("\n")

    # Try to find "Category: <label>" line
    for line in lines:
        if line.lower().startswith("category:"):
            candidate = line.split(":", 1)[1].strip().rstrip(".")
            label = parse_label(candidate, label_names)
            if label != label_names[0] or candidate.lower() == label_names[0].lower():
                return label

    # Fall back to general parse on full text
    return parse_label(raw_output, label_names)


def categorize_output(raw_output, parsed_label, true_label, label_names):
    """Categorize output quality into one of four types."""
    has_reasoning = False
    lower = raw_output.lower()

    # Check if output contains reasoning text (more than just a label)
    if "reasoning:" in lower or "because" in lower or "indicates" in lower \
       or "consistent with" in lower or "suggests" in lower \
       or len(raw_output.split()) > 10:
        has_reasoning = True

    correct = (parsed_label == true_label)

    if correct and has_reasoning:
        return "correct_with_reasoning"
    elif correct and not has_reasoning:
        return "correct_label_only"
    elif not correct and has_reasoning:
        return "incorrect_with_reasoning"
    else:
        return "incorrect_no_reasoning"


def get_stratified_sample(y_test, n_total=100, seed=42):
    """Get stratified sample indices, proportional to class distribution."""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y_test, return_counts=True)
    fracs = counts / counts.sum()
    n_per_class = np.maximum((fracs * n_total).astype(int), 2)  # at least 2 per class
    # Adjust to hit target
    while n_per_class.sum() > n_total:
        idx = np.argmax(n_per_class)
        n_per_class[idx] -= 1
    while n_per_class.sum() < n_total:
        idx = np.argmax(fracs)
        n_per_class[idx] += 1

    indices = []
    for cls, n in zip(classes, n_per_class):
        cls_indices = np.where(y_test == cls)[0]
        chosen = rng.choice(cls_indices, size=min(n, len(cls_indices)), replace=False)
        indices.extend(chosen.tolist())
    rng.shuffle(indices)
    return np.array(indices)


def run_experiment(args):
    """Run the explainability experiment."""
    print("=" * 60)
    print("LLM EXPLAINABILITY EXPERIMENT")
    print("=" * 60)

    # Load dataset
    if args.dataset == "nsl_kdd":
        X_train, X_test, y_train, y_test, feat_names, label_names, scaler, raw_test_df = load_nslkdd()
        raw_train_df = pd.read_csv(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "KDDTrain+.txt"),
            header=None
        )
        format_fn = format_nslkdd_record
        dataset_display = "NSL-KDD"
    else:
        X_train, X_test, y_train, y_test, feat_names, label_names, scaler, raw_test_df = load_unsw_nb15()
        format_fn = format_unsw_record
        dataset_display = "UNSW-NB15"

    print(f"\nDataset: {dataset_display}")
    print(f"Test set size: {len(y_test)}")
    print(f"Classes: {label_names}")

    # Get stratified sample
    sample_indices = get_stratified_sample(y_test, n_total=args.n_samples, seed=42)
    print(f"Sampled {len(sample_indices)} test records (stratified)")

    # Show class distribution in sample
    sampled_labels = y_test[sample_indices]
    for i, name in enumerate(label_names):
        count = (sampled_labels == i).sum()
        print(f"  {name}: {count}")

    # Prepare records
    records = [format_fn(raw_test_df.iloc[idx]) for idx in sample_indices]
    true_labels = [label_names[y_test[idx]] for idx in sample_indices]

    # Load model
    adapter_path = None
    if args.mode == "finetuned":
        adapter_dir = os.path.join(RESULTS_DIR, args.adapter_name)
        if os.path.exists(adapter_dir):
            adapter_path = adapter_dir
            print(f"Using LoRA adapter: {adapter_dir}")
        else:
            print(f"WARNING: Adapter not found at {adapter_dir}, running base model")

    model, tokenizer = load_model(args.model_name, adapter_path=adapter_path)
    device = next(model.parameters()).device

    # Run explanation inference
    print(f"\nRunning {args.mode} explanation inference on {len(records)} samples...")
    t_start = time.time()
    results = classify_with_explanation(model, tokenizer, records, dataset_display, label_names, device)
    total_time = time.time() - t_start
    print(f"Total inference time: {total_time:.1f}s ({total_time/len(records):.2f}s/sample)")

    # Analyze results
    parsed_labels = [r["parsed_label"] for r in results]
    categories = []
    for r, tl in zip(results, true_labels):
        cat = categorize_output(r["raw_output"], r["parsed_label"], tl, label_names)
        r["true_label"] = tl
        r["category"] = cat
        r["correct"] = (r["parsed_label"] == tl)
        categories.append(cat)

    # Compute metrics
    y_true_idx = [label_names.index(tl) for tl in true_labels]
    y_pred_idx = [label_names.index(pl) if pl in label_names else 0 for pl in parsed_labels]
    acc = accuracy_score(y_true_idx, y_pred_idx)
    f1 = f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)

    # Count categories
    from collections import Counter
    cat_counts = Counter(categories)

    print(f"\n{'='*60}")
    print(f"RESULTS ({args.mode}, {dataset_display})")
    print(f"{'='*60}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro F1: {f1:.3f}")
    print(f"\nOutput categories:")
    for cat_name in ["correct_with_reasoning", "correct_label_only",
                     "incorrect_with_reasoning", "incorrect_no_reasoning"]:
        n = cat_counts.get(cat_name, 0)
        pct = 100 * n / len(results)
        print(f"  {cat_name}: {n} ({pct:.1f}%)")

    has_reasoning = cat_counts.get("correct_with_reasoning", 0) + cat_counts.get("incorrect_with_reasoning", 0)
    print(f"\nSamples with reasoning: {has_reasoning}/{len(results)} ({100*has_reasoning/len(results):.1f}%)")
    print(f"Correct classifications: {sum(1 for r in results if r['correct'])}/{len(results)}")

    # Save full results
    output = {
        "experiment": "explainability",
        "dataset": args.dataset,
        "model": args.model_name,
        "mode": args.mode,
        "adapter": args.adapter_name if args.mode == "finetuned" else None,
        "n_samples": len(results),
        "accuracy": acc,
        "macro_f1": f1,
        "category_counts": dict(cat_counts),
        "total_inference_time_s": total_time,
        "avg_inference_time_s": total_time / len(results),
        "samples": results,
    }

    outfile = os.path.join(RESULTS_DIR, f"explainability_{args.mode}_{args.dataset}.json")
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")

    # Print representative examples
    print(f"\n{'='*60}")
    print("REPRESENTATIVE EXAMPLES")
    print(f"{'='*60}")
    for cat_name in ["correct_with_reasoning", "incorrect_with_reasoning",
                     "correct_label_only", "incorrect_no_reasoning"]:
        examples = [r for r in results if r["category"] == cat_name]
        if examples:
            ex = examples[0]
            print(f"\n--- {cat_name.upper()} ---")
            print(f"True: {ex['true_label']} | Predicted: {ex['parsed_label']}")
            print(f"Output: {ex['raw_output'][:300]}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Explainability Experiment")
    parser.add_argument("--dataset", choices=["nsl_kdd", "unsw_nb15"], default="nsl_kdd")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--mode", choices=["zeroshot", "finetuned"], default="finetuned")
    parser.add_argument("--adapter_name", default="lora_ft_qwen25_7b_nsl_kdd",
                        help="Name of LoRA adapter directory under results/")
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()
    run_experiment(args)
