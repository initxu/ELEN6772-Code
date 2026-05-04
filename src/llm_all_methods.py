"""Unified LLM classifier: zero-shot, few-shot, and fine-tuned LoRA."""

import time
import json
import random
import os
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def format_nslkdd_record(row):
    """Convert an NSL-KDD record to text."""
    parts = [
        f"duration={row['duration']}", f"protocol_type={row['protocol_type']}",
        f"service={row['service']}", f"flag={row['flag']}",
        f"src_bytes={row['src_bytes']}", f"dst_bytes={row['dst_bytes']}",
        f"land={row['land']}", f"wrong_fragment={row['wrong_fragment']}",
        f"urgent={row['urgent']}", f"hot={row['hot']}",
        f"num_failed_logins={row['num_failed_logins']}", f"logged_in={row['logged_in']}",
        f"num_compromised={row['num_compromised']}", f"root_shell={row['root_shell']}",
        f"su_attempted={row['su_attempted']}", f"num_root={row['num_root']}",
        f"num_file_creations={row['num_file_creations']}",
        f"count={row['count']}", f"srv_count={row['srv_count']}",
        f"serror_rate={row['serror_rate']}", f"srv_serror_rate={row['srv_serror_rate']}",
        f"rerror_rate={row['rerror_rate']}", f"same_srv_rate={row['same_srv_rate']}",
        f"diff_srv_rate={row['diff_srv_rate']}",
        f"dst_host_count={row['dst_host_count']}",
        f"dst_host_srv_count={row['dst_host_srv_count']}",
        f"dst_host_same_srv_rate={row['dst_host_same_srv_rate']}",
        f"dst_host_serror_rate={row['dst_host_serror_rate']}",
    ]
    return ", ".join(parts)


def format_unsw_record(row):
    """Convert an UNSW-NB15 record to text."""
    parts = [
        f"duration={row['dur']}", f"protocol={row['proto']}",
        f"service={row['service']}", f"state={row['state']}",
        f"src_packets={row['spkts']}", f"dst_packets={row['dpkts']}",
        f"src_bytes={row['sbytes']}", f"dst_bytes={row['dbytes']}",
        f"rate={row['rate']}", f"src_ttl={row['sttl']}", f"dst_ttl={row['dttl']}",
        f"src_load={row['sload']}", f"dst_load={row['dload']}",
        f"src_loss={row['sloss']}", f"dst_loss={row['dloss']}",
        f"src_interpacket_time={row['sinpkt']}", f"dst_interpacket_time={row['dinpkt']}",
        f"tcp_rtt={row['tcprtt']}", f"syn_ack_time={row['synack']}",
        f"src_mean_pkt_size={row['smean']}", f"dst_mean_pkt_size={row['dmean']}",
        f"ct_srv_src={row['ct_srv_src']}", f"ct_state_ttl={row['ct_state_ttl']}",
        f"ct_dst_ltm={row['ct_dst_ltm']}",
    ]
    return ", ".join(parts)


def build_category_description(dataset_name):
    """Return category descriptions for the prompt."""
    if dataset_name == "NSL-KDD":
        return """Categories:
- normal: Normal, legitimate network connection
- DoS: Denial of Service attack (e.g., SYN flood, smurf, neptune)
- Probe: Network surveillance or probing (e.g., port scan, IP sweep)
- R2L: Remote to Local attack (unauthorized remote access)
- U2R: User to Root attack (privilege escalation)"""
    else:  # UNSW-NB15
        return """Categories:
- Normal: Normal network traffic
- Generic: Generic attack traffic
- Exploits: Exploitation of vulnerabilities
- Fuzzers: Fuzzing attacks (random data injection)
- DoS: Denial of Service attack
- Reconnaissance: Network reconnaissance/scanning
- Analysis: Analysis-based attacks (port scan, spam, HTML attacks)
- Backdoor: Backdoor access attacks
- Shellcode: Shellcode injection attacks
- Worms: Self-propagating worm attacks"""


def build_few_shot_examples(raw_df, label_names, format_fn, category_col, n_per_class=3):
    """Build few-shot examples from training data."""
    examples = []
    for cat in label_names:
        subset = raw_df[raw_df[category_col] == cat]
        if len(subset) == 0:
            continue
        sampled = subset.sample(n=min(n_per_class, len(subset)), random_state=42)
        for _, row in sampled.iterrows():
            text = format_fn(row)
            examples.append(f"Input: {text}\nOutput: {cat}")
    return "\n\n".join(examples)


def load_model(model_name, adapter_path=None):
    """Load LLM model and tokenizer, optionally with LoRA adapter."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter from {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


def classify_batch(model, tokenizer, records, prompt_template, label_names, device="cuda"):
    """Classify records one at a time (batch=1 for autoregressive models)."""
    predictions = []
    times = []

    for i, record_text in enumerate(records):
        prompt = prompt_template.format(record=record_text)
        messages = [
            {"role": "system", "content": "You are a network security classifier. Output exactly one category label."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=5, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0
        times.append(elapsed)

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip().rstrip(".")

        # Parse response
        pred = parse_label(response, label_names)
        predictions.append(pred)

    return predictions, times


def parse_label(response, label_names):
    """Parse model response to extract a label."""
    response_clean = response.strip()
    # Exact match
    for label in label_names:
        if response_clean.lower() == label.lower():
            return label
    # Substring match
    for label in label_names:
        if label.lower() in response_clean.lower():
            return label
    # Default to first label (most common)
    return label_names[0]


def run_zero_shot(model, tokenizer, raw_test_df, y_test, label_names,
                  format_fn, dataset_name, sample_indices, device):
    """Zero-shot classification — no examples, just task description."""
    cat_desc = build_category_description(dataset_name)
    prompt_template = f"""Classify this network connection record into exactly one category.

{cat_desc}

Reply with ONLY the category label, nothing else.

Input: {{record}}
Output:"""

    records = [format_fn(raw_test_df.iloc[idx]) for idx in sample_indices]
    y_sub = y_test[sample_indices]

    print(f"\n  Zero-shot: classifying {len(records)} samples...")
    preds, times = classify_batch(model, tokenizer, records, prompt_template, label_names, device)

    y_pred = np.array([label_names.index(p) if p in label_names else 0 for p in preds])
    return compute_metrics("LLM Zero-shot", y_sub, y_pred, label_names, times, 0.0)


def run_few_shot(model, tokenizer, raw_train_df, raw_test_df, y_test, label_names,
                 format_fn, dataset_name, category_col, sample_indices, device, n_few_shot=3):
    """Few-shot classification — with examples in prompt."""
    cat_desc = build_category_description(dataset_name)
    few_shot_text = build_few_shot_examples(raw_train_df, label_names, format_fn, category_col, n_few_shot)

    prompt_template = f"""Classify this network connection record into exactly one category.

{cat_desc}

Examples:

{few_shot_text}

Now classify this record. Reply with ONLY the category label.

Input: {{record}}
Output:"""

    records = [format_fn(raw_test_df.iloc[idx]) for idx in sample_indices]
    y_sub = y_test[sample_indices]

    print(f"\n  Few-shot ({n_few_shot}-shot): classifying {len(records)} samples...")
    preds, times = classify_batch(model, tokenizer, records, prompt_template, label_names, device)

    y_pred = np.array([label_names.index(p) if p in label_names else 0 for p in preds])
    return compute_metrics("LLM Few-shot", y_sub, y_pred, label_names, times, 0.0)


def compute_metrics(model_name, y_true, y_pred, label_names, times, train_time):
    """Compute and return all evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))

    total_time = sum(times)
    avg_time = np.mean(times) if times else 0

    print(f"\n  {model_name} Results ({len(y_true)} samples):")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f} (macro)")
    print(f"    Recall:    {rec:.4f} (macro)")
    print(f"    F1-score:  {f1:.4f} (macro)")
    print(f"    Inference: {avg_time*1000:.2f}ms/sample")
    print(f"\n{classification_report(y_true, y_pred, target_names=label_names, zero_division=0)}")

    per_class = {}
    prec_pc = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(label_names))))
    rec_pc = recall_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(label_names))))
    f1_pc = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(label_names))))
    for i, name in enumerate(label_names):
        per_class[name] = {
            "precision": float(prec_pc[i]) if i < len(prec_pc) else 0.0,
            "recall": float(rec_pc[i]) if i < len(rec_pc) else 0.0,
            "f1": float(f1_pc[i]) if i < len(f1_pc) else 0.0,
        }

    return {
        "model": model_name,
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "train_time_s": float(train_time),
        "inference_time_s": float(total_time),
        "inference_ms_per_sample": float(avg_time * 1000),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "n_samples": len(y_true),
    }


def get_stratified_sample(y_test, label_names, n_samples=500):
    """Get stratified sample indices from test set."""
    random.seed(42)
    np.random.seed(42)
    sample_indices = []
    for cat_idx in range(len(label_names)):
        cat_indices = np.where(y_test == cat_idx)[0]
        n_cat = max(1, int(n_samples * len(cat_indices) / len(y_test)))
        n_cat = min(n_cat, len(cat_indices))
        sampled = np.random.choice(cat_indices, size=n_cat, replace=False)
        sample_indices.extend(sampled)
    return sorted(sample_indices)[:n_samples]


def finetune_lora(raw_train_df, label_names, format_fn, category_col,
                  model_name="Qwen/Qwen2.5-3B-Instruct", dataset_name="NSL-KDD",
                  n_train=5000, output_dir=None):
    """Fine-tune model with LoRA on training data."""
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import TrainingArguments, Trainer
    from torch.utils.data import Dataset as TorchDataset

    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, f"lora_{dataset_name.lower().replace('-', '_')}")

    # Check if already fine-tuned
    if os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        print(f"  LoRA adapter already exists at {output_dir}, skipping training")
        return output_dir

    print(f"\n  Fine-tuning {model_name} with LoRA on {dataset_name}")
    print(f"  Training samples: {n_train}")

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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

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
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_strategy="no",
        warmup_steps=50,
        weight_decay=0.01,
        report_to="none",
        remove_unused_columns=False,
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
    torch.cuda.empty_cache()

    return output_dir


def run_finetuned(raw_test_df, y_test, label_names, format_fn, dataset_name,
                  model_name="Qwen/Qwen2.5-3B-Instruct", adapter_path=None,
                  sample_indices=None, train_time=0.0):
    """Evaluate fine-tuned model."""
    cat_desc = build_category_description(dataset_name)
    prompt_template = f"""Classify this network connection.
{cat_desc}
Input: {{record}}
Output:"""

    model, tokenizer = load_model(model_name, adapter_path)
    device = next(model.parameters()).device

    records = [format_fn(raw_test_df.iloc[idx]) for idx in sample_indices]
    y_sub = y_test[sample_indices]

    print(f"\n  Fine-tuned: classifying {len(records)} samples...")
    preds, times = classify_batch(model, tokenizer, records, prompt_template, label_names, device)

    del model
    torch.cuda.empty_cache()

    y_pred = np.array([label_names.index(p) if p in label_names else 0 for p in preds])
    return compute_metrics("LLM Fine-tuned", y_sub, y_pred, label_names, times, train_time)


def run_all_llm_methods(raw_train_df, raw_test_df, y_test, label_names,
                        format_fn, dataset_name, category_col,
                        base_model="Qwen/Qwen2.5-7B-Instruct",
                        ft_model="Qwen/Qwen2.5-3B-Instruct",
                        n_samples=500, n_few_shot=3, n_train_ft=5000):
    """Run all LLM methods: zero-shot, few-shot, fine-tuned."""
    print(f"\n{'='*60}")
    print(f"LLM EXPERIMENTS ON {dataset_name}")
    print(f"{'='*60}")

    sample_indices = get_stratified_sample(y_test, label_names, n_samples)
    all_results = []

    # 1. Zero-shot and Few-shot with base model
    print(f"\nLoading base model for zero-shot and few-shot: {base_model}")
    model, tokenizer = load_model(base_model)
    device = next(model.parameters()).device

    zs_result = run_zero_shot(model, tokenizer, raw_test_df, y_test, label_names,
                              format_fn, dataset_name, sample_indices, device)
    all_results.append(zs_result)

    fs_result = run_few_shot(model, tokenizer, raw_train_df, raw_test_df, y_test, label_names,
                             format_fn, dataset_name, category_col, sample_indices, device, n_few_shot)
    all_results.append(fs_result)

    del model
    torch.cuda.empty_cache()

    # 2. Fine-tune with smaller model
    print(f"\n--- Fine-tuning {ft_model} with LoRA ---")
    t0 = time.time()
    adapter_path = finetune_lora(raw_train_df, label_names, format_fn, category_col,
                                 model_name=ft_model, dataset_name=dataset_name,
                                 n_train=n_train_ft)
    ft_train_time = time.time() - t0

    ft_result = run_finetuned(raw_test_df, y_test, label_names, format_fn, dataset_name,
                              model_name=ft_model, adapter_path=adapter_path,
                              sample_indices=sample_indices, train_time=ft_train_time)
    all_results.append(ft_result)

    return all_results
