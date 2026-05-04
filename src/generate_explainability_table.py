"""Generate LaTeX table and analysis for explainability experiment results."""

import os
import json
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tables")


def load_results(mode, dataset):
    path = os.path.join(RESULTS_DIR, f"explainability_{mode}_{dataset}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def select_examples(samples, n_per_category=2):
    """Select representative examples for the paper table."""
    examples = []
    categories_wanted = [
        ("correct_with_reasoning", "Correct + Reasoning"),
        ("incorrect_with_reasoning", "Incorrect + Reasoning"),
    ]
    seen_true_labels = set()
    for cat_key, cat_display in categories_wanted:
        for s in samples:
            if s["category"] == cat_key and s["true_label"] not in seen_true_labels:
                # Truncate output for table
                output = s["raw_output"].replace("\n", " ").strip()
                if len(output) > 200:
                    output = output[:197] + "..."
                examples.append({
                    "true": s["true_label"],
                    "pred": s["parsed_label"],
                    "type": cat_display,
                    "output": output,
                })
                seen_true_labels.add(s["true_label"])
                if len([e for e in examples if e["type"] == cat_display]) >= n_per_category:
                    break
    return examples


def generate_examples_table(ft_results):
    """Generate LaTeX table of representative examples."""
    examples = select_examples(ft_results["samples"], n_per_category=3)

    rows = []
    for ex in examples:
        # Escape LaTeX special chars
        output = ex["output"].replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
        true_l = ex["true"].replace("_", r"\_")
        pred_l = ex["pred"].replace("_", r"\_")
        rows.append(f"    {true_l} & {pred_l} & \\small {output} \\\\")

    table = r"""\begin{table}[t]
\centering
\caption{Representative LLM outputs with explanation-eliciting prompts (fine-tuned Qwen2.5-7B on NSL-KDD). When asked to classify \emph{and explain}, the model produces feature-grounded reasoning for every sample. Top rows show correct classifications with accurate reasoning; bottom rows show incorrect classifications where the reasoning is plausible but the conclusion is wrong.}
\label{tab:explanations}
\small
\begin{tabular}{llp{9.5cm}}
\toprule
True & Pred & LLM Output (truncated) \\
\midrule
""" + "\n    \\midrule\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return table


def generate_summary_table(results_dict):
    """Generate summary table comparing explanation modes."""
    rows = []
    for (mode, dataset), res in sorted(results_dict.items()):
        if res is None:
            continue
        cc = res["category_counts"]
        n = res["n_samples"]
        reasoning_pct = 100 * (cc.get("correct_with_reasoning", 0) + cc.get("incorrect_with_reasoning", 0)) / n
        correct_pct = 100 * (cc.get("correct_with_reasoning", 0) + cc.get("correct_label_only", 0)) / n
        rows.append(
            f"    {mode.capitalize()} & {dataset.replace('_', '-').upper()} & "
            f"{res['accuracy']:.3f} & {res['macro_f1']:.3f} & "
            f"{reasoning_pct:.0f}\\% & {correct_pct:.0f}\\% \\\\"
        )

    table = r"""\begin{table}[t]
\centering
\caption{Explainability experiment results. When prompted to classify and explain (max 150 tokens), the LLM produces feature-grounded reasoning for all samples under both zero-shot and fine-tuned conditions. ``With Reasoning'' indicates the percentage of outputs containing substantive explanations referencing specific network features.}
\label{tab:explain_summary}
\small
\begin{tabular}{llcccc}
\toprule
Mode & Dataset & Acc & Macro F1 & With Reasoning & Correct \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return table


if __name__ == "__main__":
    # Load all available results
    results = {}
    for mode in ["finetuned", "zeroshot"]:
        for dataset in ["nsl_kdd", "unsw_nb15"]:
            res = load_results(mode, dataset)
            if res:
                results[(mode, dataset)] = res
                print(f"Loaded {mode}/{dataset}: acc={res['accuracy']:.3f}, f1={res['macro_f1']:.3f}")

    if not results:
        print("No results found!")
        sys.exit(1)

    # Generate summary table
    summary = generate_summary_table(results)
    summary_path = os.path.join(TABLES_DIR, "explain_summary.tex")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSummary table: {summary_path}")

    # Generate examples table (from fine-tuned NSL-KDD if available)
    ft_nsl = results.get(("finetuned", "nsl_kdd"))
    if ft_nsl:
        examples = generate_examples_table(ft_nsl)
        examples_path = os.path.join(TABLES_DIR, "explain_examples.tex")
        with open(examples_path, "w") as f:
            f.write(examples)
        print(f"Examples table: {examples_path}")

    print("\nDone!")
