"""Additional deep learning models for network intrusion detection.

Models: MLP, BiLSTM, 1D-ResNet
All follow the same interface as cnn_model.py: train_X(X_train, X_test, y_train, y_test, label_names) -> dict
"""

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


# ── Shared training / evaluation loop ──────────────────────────────────

def _train_and_eval(model, model_name, X_train, X_test, y_train, y_test,
                    label_names, epochs=50, batch_size=256, lr=1e-3, patience=5):
    """Generic train+eval for any nn.Module that takes (batch, features)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training {model_name} on {device}")

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_tr, y_tr)
    test_ds = TensorDataset(X_te, y_te)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    num_classes = len(label_names)
    model = model.to(device)

    # Class-weighted loss
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_loss = float("inf")
    patience_counter = 0
    best_state = None
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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(test_ds)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train={avg_loss:.4f} val={val_loss:.4f}")

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
    model.load_state_dict(best_state)
    model.eval()

    all_preds = []
    infer_start = time.time()
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            all_preds.append(model(xb).argmax(dim=1).cpu().numpy())
    infer_time = time.time() - infer_start
    y_pred = np.concatenate(all_preds)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"  Train: {train_time:.2f}s  Infer: {infer_time:.4f}s "
          f"({infer_time/len(y_test)*1000:.4f}ms/sample)")
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

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

    return {
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
        "epochs_trained": epoch + 1,
    }


# ── Model 1: MLP ──────────────────────────────────────────────────────

class MLP(nn.Module):
    """Multi-layer perceptron with 3 hidden layers, BatchNorm, and Dropout."""

    def __init__(self, input_dim, num_classes, hidden=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, X_test, y_train, y_test, label_names, **kwargs):
    model = MLP(X_train.shape[1], len(label_names))
    return _train_and_eval(model, "MLP", X_train, X_test, y_train, y_test,
                           label_names, **kwargs)


# ── Model 2: BiLSTM ───────────────────────────────────────────────────

class BiLSTM(nn.Module):
    """Bidirectional LSTM treating feature vector as a sequence of scalars."""

    def __init__(self, input_dim, num_classes, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        # Group features into chunks of ~4 to form a short sequence
        self.chunk_size = 4
        self.seq_len = (input_dim + self.chunk_size - 1) // self.chunk_size
        self.pad_to = self.seq_len * self.chunk_size
        self.lstm = nn.LSTM(
            input_size=self.chunk_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, features) -> (batch, seq_len, chunk_size)
        if x.shape[1] < self.pad_to:
            x = nn.functional.pad(x, (0, self.pad_to - x.shape[1]))
        x = x.view(x.size(0), self.seq_len, self.chunk_size)
        out, _ = self.lstm(x)
        # Use last hidden state from both directions
        x = out[:, -1, :]
        return self.classifier(x)


def train_bilstm(X_train, X_test, y_train, y_test, label_names, **kwargs):
    model = BiLSTM(X_train.shape[1], len(label_names))
    return _train_and_eval(model, "BiLSTM", X_train, X_test, y_train, y_test,
                           label_names, **kwargs)


# ── Model 3: 1D-ResNet ────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """Residual block with two 1D convolutions."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResNet1D(nn.Module):
    """1D ResNet: Conv stem + 3 residual blocks + global avg pool + FC."""

    def __init__(self, input_dim, num_classes, channels=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            ResBlock1D(channels),
            ResBlock1D(channels),
            ResBlock1D(channels),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def train_resnet1d(X_train, X_test, y_train, y_test, label_names, **kwargs):
    model = ResNet1D(X_train.shape[1], len(label_names))
    return _train_and_eval(model, "1D-ResNet", X_train, X_test, y_train, y_test,
                           label_names, **kwargs)


# ── Run all 3 new DL models ───────────────────────────────────────────

def run_all_dl(X_train, X_test, y_train, y_test, label_names, **kwargs):
    """Train and evaluate MLP, BiLSTM, and 1D-ResNet."""
    results = []
    for fn in [train_mlp, train_bilstm, train_resnet1d]:
        results.append(fn(X_train, X_test, y_train, y_test, label_names, **kwargs))
    return results
