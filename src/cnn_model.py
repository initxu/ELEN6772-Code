"""1D CNN model for network intrusion detection."""

import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for intrusion detection."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x


def train_cnn(X_train, X_test, y_train, y_test, label_names,
              epochs=50, batch_size=256, lr=1e-3, patience=5):
    """Train and evaluate CNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Prepare data
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_tr, y_tr)
    test_ds = TensorDataset(X_te, y_te)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Model with class weights for imbalanced data
    num_classes = len(label_names)
    model = CNN1D(X_train.shape[1], num_classes).to(device)

    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    best_loss = float("inf")
    patience_counter = 0
    train_start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        avg_loss = total_loss / len(train_ds)

        # Validation loss on test set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item() * len(xb)
        val_loss /= len(test_ds)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    train_time = time.time() - train_start

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Predict
    all_preds = []
    infer_start = time.time()
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
    infer_time = time.time() - infer_start

    y_pred = np.concatenate(all_preds)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"CNN Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f} (macro)")
    print(f"  Recall:    {rec:.4f} (macro)")
    print(f"  F1-score:  {f1:.4f} (macro)")
    print(f"  Train time: {train_time:.2f}s")
    print(f"  Inference time: {infer_time:.4f}s ({infer_time/len(y_test)*1000:.4f}ms/sample)")
    print(f"\n{classification_report(y_test, y_pred, target_names=label_names, zero_division=0)}")

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
        "model": "1D-CNN",
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "train_time_s": float(train_time),
        "inference_time_s": float(infer_time),
        "inference_ms_per_sample": float(infer_time / len(y_test) * 1000),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "epochs_trained": epoch + 1,
    }
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_nslkdd

    X_train, X_test, y_train, y_test, feat, labels, scaler, _ = load_nslkdd()
    result = train_cnn(X_train, X_test, y_train, y_test, labels)

    import os
    os.makedirs("results", exist_ok=True)
    with open("results/cnn_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nResults saved to results/cnn_results.json")
