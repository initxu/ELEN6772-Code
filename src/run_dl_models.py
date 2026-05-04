#!/usr/bin/env python3
"""Run new DL models (MLP, BiLSTM, 1D-ResNet) on both datasets."""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import load_nslkdd
from src.data_loader_unsw import load_unsw_nb15
from src.dl_models import run_all_dl

os.makedirs("results/nsl_kdd", exist_ok=True)
os.makedirs("results/unsw_nb15", exist_ok=True)

# ── NSL-KDD ──
print("\n" + "="*70)
print("  NSL-KDD")
print("="*70)
X_train, X_test, y_train, y_test, _, labels, _, _ = load_nslkdd()
nsl_results = run_all_dl(X_train, X_test, y_train, y_test, labels)
with open("results/nsl_kdd/dl_new_results.json", "w") as f:
    json.dump(nsl_results, f, indent=2)
print("\nSaved results/nsl_kdd/dl_new_results.json")

# ── UNSW-NB15 ──
print("\n" + "="*70)
print("  UNSW-NB15")
print("="*70)
X_train, X_test, y_train, y_test, _, labels, _, _ = load_unsw_nb15()
unsw_results = run_all_dl(X_train, X_test, y_train, y_test, labels)
with open("results/unsw_nb15/dl_new_results.json", "w") as f:
    json.dump(unsw_results, f, indent=2)
print("\nSaved results/unsw_nb15/dl_new_results.json")

# ── Summary ──
print("\n" + "="*70)
print("  SUMMARY")
print("="*70)
print(f"{'Model':<12} {'NSL-KDD Acc':>12} {'NSL-KDD F1':>12} {'UNSW Acc':>12} {'UNSW F1':>12}")
print("-" * 62)
for nr, ur in zip(nsl_results, unsw_results):
    print(f"{nr['model']:<12} {nr['accuracy']:>12.4f} {nr['f1_macro']:>12.4f} "
          f"{ur['accuracy']:>12.4f} {ur['f1_macro']:>12.4f}")
