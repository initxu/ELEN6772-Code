"""Fine-tune remaining models: Qwen2.5-32B (QLoRA 4-bit) and DeepSeek-7B."""

import os
import sys
import json
import time
import random
import gc
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.llm_all_methods import (
    classify_batch, compute_metrics,
    build_category_description, get_stratified_sample,
    format_nslkdd_record, format_unsw_record,
    parse_label,
)
from src.data_loader import load_nslkdd, COLUMNS, ATTACK_MAP, DATA_DIR
from src.data_loader_unsw import load_unsw_nb15

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def load_model_quantized(model_name, quantize_4bit=False, adapter_path=None):
    """Load model with optional 4-bit quantization for QLoRA."""
    print(f"Loading model: {model_name} (4-bit={quantize_4bit})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter from {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        # Don't merge_and_unload for quantized models
        if not quantize_4bit:
            model = model.merge_and_unload()

    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


def finetune_qlora(raw_train_df, label_names, format_fn, category_col,
                   model_name, dataset_name, n_train=5000, output_dir=None,
                   quantize_4bit=False, batch_size=2, grad_accum=8):
    """Fine-tune model with LoRA/QLoRA."""
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from transformers import TrainingArguments, Trainer
    from torch.utils.data import Dataset as TorchDataset

    if output_dir is None:
        raise ValueError("output_dir must be specified")

    # Check if already fine-tuned
    if os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        print(f"  LoRA adapter already exists at {output_dir}, skipping training")
        return output_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Fine-tuning {model_name} with {'QLoRA (4-bit)' if quantize_4bit else 'LoRA'} on {dataset_name}")
    print(f"  Training samples: {n_train}, batch_size: {batch_size}, grad_accum: {grad_accum}")

    # Sample training data (stratified)
    train_samples = []
    for cat in label_names:
        subset = raw_train_df[raw_train_df[category_col] == cat]
        n_cat = max(1, int(n_train * len(subset) / len(raw_train_df)))
        n_cat = min(n_cat, len(subset))
        sampled = subset.sample(n=n_cat, random_state=42)
        for _, row in sampled.iterrows():
            text = format_fn(row)
            train_samples.append({"input": text, "output": cat})

    random.seed(42)
    random.shuffle(train_samples)
    print(f"  Total training samples: {len(train_samples)}")

    cat_desc = build_category_description(dataset_name)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if quantize_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    class IDSDataset(TorchDataset):
        def __init__(self, samples, tokenizer, cat_desc, max_length=512):
            self.samples = samples
            self.tokenizer = tokenizer
            self.cat_desc = cat_desc
            self.max_length = max_length

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            prompt = f"Classify this network connection.\n{self.cat_desc}\nInput: {s['input']}\nOutput:"
            full_text = f"{prompt} {s['output']}"

            encoding = self.tokenizer(
                full_text, truncation=True, max_length=self.max_length,
                padding="max_length", return_tensors="pt"
            )
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()

            # Only compute loss on the answer part
            prompt_enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length)
            prompt_len = len(prompt_enc["input_ids"])

            labels = input_ids.clone()
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100

            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_ds = IDSDataset(train_samples, tokenizer, cat_desc)

    # Training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_strategy="no",
        warmup_steps=50,
        weight_decay=0.01,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=quantize_4bit,  # Enable for large models
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"  Fine-tuning completed in {train_time:.1f}s")

    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  LoRA adapter saved to {output_dir}")

    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


def finetune_and_eval_model(model_id, short_name, family, size_label,
                            raw_train_df, raw_test_df, y_test, label_names,
                            format_fn, dataset_name, category_col, sample_indices,
                            quantize_4bit=False, batch_size=4, grad_accum=4):
    """Fine-tune and evaluate a single model on a single dataset."""
    tag = short_name.lower().replace(".", "").replace("-", "_")
    ds_tag = dataset_name.lower().replace("-", "_")
    adapter_dir = os.path.join(RESULTS_DIR, f"lora_ft_{tag}_{ds_tag}")

    print(f"\n{'='*60}")
    print(f"FINE-TUNE: {short_name} ({family}, {size_label}) on {dataset_name}")
    print(f"Adapter dir: {adapter_dir}")
    print(f"4-bit QLoRA: {quantize_4bit}")
    print(f"{'='*60}")

    # Fine-tune
    t0 = time.time()
    try:
        adapter_path = finetune_qlora(
            raw_train_df, label_names, format_fn, category_col,
            model_name=model_id, dataset_name=dataset_name,
            n_train=5000, output_dir=adapter_dir,
            quantize_4bit=quantize_4bit,
            batch_size=batch_size, grad_accum=grad_accum,
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
        model, tokenizer = load_model_quantized(model_id, quantize_4bit=quantize_4bit,
                                                 adapter_path=adapter_path)
    except Exception as e:
        print(f"  ERROR loading fine-tuned {short_name}: {e}")
        import traceback
        traceback.print_exc()
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
    result["quantized_4bit"] = quantize_4bit

    return result


def run_zero_shot_eval(model_id, short_name, family, size_label,
                       raw_test_df, y_test, label_names,
                       format_fn, dataset_name, sample_indices):
    """Run zero-shot evaluation."""
    cat_desc = build_category_description(dataset_name)
    prompt_template = f"""Classify this network connection record into exactly one category.

{cat_desc}

Reply with ONLY the category label, nothing else.

Input: {{record}}
Output:"""

    try:
        model, tokenizer = load_model_quantized(model_id, quantize_4bit=False)
    except Exception as e:
        print(f"  ERROR loading {short_name}: {e}")
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
    result["model_id"] = model_id
    result["family"] = family
    result["size_label"] = size_label

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune remaining models (32B QLoRA + DeepSeek)")
    parser.add_argument("--model", required=True, choices=["qwen32b", "deepseek7b"],
                        help="Which model to fine-tune")
    parser.add_argument("--datasets", nargs="*", default=["NSL-KDD", "UNSW-NB15"])
    parser.add_argument("--run-zs", action="store_true", help="Also run zero-shot")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Model configs
    MODEL_CONFIGS = {
        "qwen32b": {
            "model_id": "Qwen/Qwen2.5-32B-Instruct",
            "short_name": "Qwen2.5-32B",
            "family": "Qwen",
            "size_label": "32B",
            "quantize_4bit": False,
            "batch_size": 4,
            "grad_accum": 4,
        },
        "deepseek7b": {
            "model_id": "deepseek-ai/deepseek-llm-7b-chat",
            "short_name": "DeepSeek-7B",
            "family": "DeepSeek",
            "size_label": "7B",
            "quantize_4bit": False,
            "batch_size": 4,
            "grad_accum": 4,
        },
    }

    cfg = MODEL_CONFIGS[args.model]

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
        print(f"# {cfg['short_name']} on {ds_name}")
        print(f"{'#'*70}")

        ds_results = []

        # Zero-shot (optional)
        if args.run_zs:
            zs_result = run_zero_shot_eval(
                cfg["model_id"], cfg["short_name"], cfg["family"], cfg["size_label"],
                ds_cfg["raw_test"], ds_cfg["y_test"],
                ds_cfg["label_names"], ds_cfg["format_fn"],
                ds_name, ds_cfg["sample_indices"],
            )
            if zs_result:
                ds_results.append(zs_result)

        # Fine-tune
        ft_result = finetune_and_eval_model(
            cfg["model_id"], cfg["short_name"], cfg["family"], cfg["size_label"],
            ds_cfg["raw_train"], ds_cfg["raw_test"],
            ds_cfg["y_test"], ds_cfg["label_names"],
            ds_cfg["format_fn"], ds_name, ds_cfg["category_col"],
            ds_cfg["sample_indices"],
            quantize_4bit=cfg["quantize_4bit"],
            batch_size=cfg["batch_size"],
            grad_accum=cfg["grad_accum"],
        )
        if ft_result:
            ds_results.append(ft_result)

        all_results[ds_name] = ds_results

        # Save incrementally
        tag = cfg["short_name"].lower().replace(".", "").replace("-", "_")
        out_path = os.path.join(RESULTS_DIR, f"ft_{tag}_{ds_name.lower().replace('-', '_')}.json")
        with open(out_path, "w") as f:
            json.dump(ds_results, f, indent=2)

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"{cfg['short_name']} EXPERIMENTS COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")

    for ds_name, ds_results in all_results.items():
        for r in ds_results:
            print(f"  {r['model']:<25} {ds_name:<12} F1={r['f1_macro']:.4f}  Acc={r['accuracy']:.4f}")


if __name__ == "__main__":
    main()
