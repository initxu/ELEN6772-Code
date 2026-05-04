"""Traditional ML models: Logistic Regression, SVM, Random Forest."""

import time
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, label_names):
    """Train a model and return evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")

    # Train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predict
    t0 = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - t0

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f} (macro)")
    print(f"  Recall:    {rec:.4f} (macro)")
    print(f"  F1-score:  {f1:.4f} (macro)")
    print(f"  Train time: {train_time:.2f}s")
    print(f"  Inference time: {infer_time:.4f}s ({infer_time/len(y_test)*1000:.4f}ms/sample)")
    print(f"\n{classification_report(y_test, y_pred, target_names=label_names, zero_division=0)}")

    # Per-class metrics
    per_class = {}
    prec_pc = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_pc = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_pc = f1_score(y_test, y_pred, average=None, zero_division=0)
    for i, name in enumerate(label_names):
        per_class[name] = {
            "precision": float(prec_pc[i]),
            "recall": float(rec_pc[i]),
            "f1": float(f1_pc[i]),
        }

    results = {
        "model": model_name,
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "train_time_s": float(train_time),
        "inference_time_s": float(infer_time),
        "inference_ms_per_sample": float(infer_time / len(y_test) * 1000),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
    return results


def run_all_ml(X_train, X_test, y_train, y_test, label_names):
    """Run all traditional ML models."""
    models = [
        (LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=-1),
         "Logistic Regression"),
        (SVC(C=1.0, kernel="rbf", gamma="scale", decision_function_shape="ovr"),
         "SVM (RBF)"),
        (RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42),
         "Random Forest"),
    ]

    if HAS_XGBOOST:
        models.append((
            XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          use_label_encoder=False, eval_metric="mlogloss",
                          tree_method="hist", n_jobs=-1, random_state=42),
            "XGBoost"
        ))

    all_results = []
    for model, name in models:
        result = train_and_evaluate(model, name, X_train, X_test, y_train, y_test, label_names)
        all_results.append(result)

    return all_results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_nslkdd

    X_train, X_test, y_train, y_test, feat, labels, scaler, _ = load_nslkdd()
    results = run_all_ml(X_train, X_test, y_train, y_test, labels)

    import os
    os.makedirs("results", exist_ok=True)
    with open("results/ml_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/ml_results.json")
