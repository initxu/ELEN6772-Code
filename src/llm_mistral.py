"""Run zero-shot with Mistral-7B as alternative to gated Llama."""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.llm_multi_model import run_zero_shot_with_model, RESULTS_DIR
from src.llm_all_methods import get_stratified_sample, format_nslkdd_record, format_unsw_record
from src.data_loader import load_nslkdd
from src.data_loader_unsw import load_unsw_nb15


def main():
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    short_name = "Mistral-7B"

    print("Loading NSL-KDD...")
    _, X_test_nsl, _, y_test_nsl, _, nsl_labels, _, raw_test_nsl = load_nslkdd()
    nsl_indices = get_stratified_sample(y_test_nsl, nsl_labels, 500)

    print("Loading UNSW-NB15...")
    _, X_test_unsw, _, y_test_unsw, _, unsw_labels, _, raw_test_unsw = load_unsw_nb15()
    unsw_indices = get_stratified_sample(y_test_unsw, unsw_labels, 500)

    results = {}

    print(f"\n--- {short_name} on NSL-KDD ---")
    r = run_zero_shot_with_model(
        model_id, short_name, raw_test_nsl, y_test_nsl,
        nsl_labels, format_nslkdd_record, "NSL-KDD", nsl_indices,
    )
    if r:
        r["model_id"] = model_id
        r["category"] = "Mistral"
        results["NSL-KDD"] = r

    print(f"\n--- {short_name} on UNSW-NB15 ---")
    r = run_zero_shot_with_model(
        model_id, short_name, raw_test_unsw, y_test_unsw,
        unsw_labels, format_unsw_record, "UNSW-NB15", unsw_indices,
    )
    if r:
        r["model_id"] = model_id
        r["category"] = "Mistral"
        results["UNSW-NB15"] = r

    with open(os.path.join(RESULTS_DIR, "mistral_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nMistral experiments complete!")


if __name__ == "__main__":
    main()
