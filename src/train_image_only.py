"""
train_image_only.py
Train an image-only MLP classifier on ResNet50 features.
Input:  features/image_features.npy, image_labels.npy, image_ids.npy
        data/splits/train.csv, val.csv, test.csv
Output: results/image_only_best.pt
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

# ── Config ───────────────────────────────────────────────────────────────────
FEATURES_DIR  = "features"
SPLITS_DIR    = "data/splits"
RESULTS_DIR   = "results"
EPOCHS        = 50
BATCH_SIZE    = 64
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
SEED          = 42
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Model ────────────────────────────────────────────────────────────────────
class ImageMLP(nn.Module):
    def __init__(self, in_dim=2048, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_features():
    features = np.load(os.path.join(FEATURES_DIR, "image_features.npy"))
    labels   = np.load(os.path.join(FEATURES_DIR, "image_labels.npy"))
    ids      = np.load(os.path.join(FEATURES_DIR, "image_ids.npy"))
    return features, labels, ids


def make_split_tensors(features, labels, ids, split_csv):
    split_ids = set(pd.read_csv(split_csv)["note_idx"].tolist())
    mask      = np.array([i in split_ids for i in ids])
    X = torch.tensor(features[mask], dtype=torch.float32)
    y = torch.tensor(labels[mask],   dtype=torch.long)
    return X, y


def make_loader(X, y, shuffle=True):
    return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle)


# ── Training / Eval ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds_all, labels_all = [], []
    for X_b, y_b in loader:
        preds_all.extend(model(X_b.to(DEVICE)).argmax(1).cpu().numpy())
        labels_all.extend(y_b.numpy())
    return (accuracy_score(labels_all, preds_all),
            f1_score(labels_all, preds_all, average="macro"))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    features, labels, ids = load_features()
    X_train, y_train = make_split_tensors(features, labels, ids, f"{SPLITS_DIR}/train.csv")
    X_val,   y_val   = make_split_tensors(features, labels, ids, f"{SPLITS_DIR}/val.csv")
    X_test,  y_test  = make_split_tensors(features, labels, ids, f"{SPLITS_DIR}/test.csv")

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"Device: {DEVICE}")

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    model     = ImageMLP(in_dim=2048, num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_val_f1     = 0.0
    best_model_path = os.path.join(RESULTS_DIR, "image_only_best.pt")

    for epoch in range(1, EPOCHS + 1):
        train_loss      = train_epoch(model, train_loader, optimizer, criterion)
        val_acc, val_f1 = evaluate(model, val_loader)
        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | loss={train_loss:.4f} | "
                  f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_acc, test_f1 = evaluate(model, test_loader)

    print("\n===== Image-Only Results =====")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Macro-F1 : {test_f1:.4f}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()