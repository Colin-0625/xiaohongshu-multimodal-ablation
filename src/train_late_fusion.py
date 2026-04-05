"""
train_late_fusion.py
Train a late-fusion MLP: concat(text[768], image[2048]) → 2816 → MLP → 3 classes.
Input:  features/text_features.npy + image_features.npy (same row order as metadata.csv)
        data/splits/train.csv, val.csv, test.csv
Output: results/late_fusion_best.pt
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
class LateFusionMLP(nn.Module):
    def __init__(self, text_dim=768, img_dim=2048, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim + img_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, text_feat, img_feat):
        x = torch.cat([text_feat, img_feat], dim=1)
        return self.net(x)


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_all_features():
    """Load text and image features; both are aligned to metadata.csv row order."""
    txt_feat = np.load(os.path.join(FEATURES_DIR, "text_features.npy"))
    txt_lbl  = np.load(os.path.join(FEATURES_DIR, "text_labels.npy"))
    txt_ids  = np.load(os.path.join(FEATURES_DIR, "text_ids.npy"))

    img_feat = np.load(os.path.join(FEATURES_DIR, "image_features.npy"))
    img_ids  = np.load(os.path.join(FEATURES_DIR, "image_ids.npy"))

    # Verify alignment
    assert np.array_equal(txt_ids, img_ids), \
        "text and image feature files are not aligned! Check extraction scripts."

    return txt_feat, img_feat, txt_lbl, txt_ids


def make_split_tensors(txt_feat, img_feat, labels, ids, split_csv):
    split_ids = set(pd.read_csv(split_csv)["note_idx"].tolist())
    mask      = np.array([i in split_ids for i in ids])
    Xt = torch.tensor(txt_feat[mask], dtype=torch.float32)
    Xi = torch.tensor(img_feat[mask], dtype=torch.float32)
    y  = torch.tensor(labels[mask],   dtype=torch.long)
    return Xt, Xi, y


def make_loader(Xt, Xi, y, shuffle=True):
    ds = TensorDataset(Xt, Xi, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# ── Training / Eval ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for Xt_b, Xi_b, y_b in loader:
        Xt_b, Xi_b, y_b = Xt_b.to(DEVICE), Xi_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xt_b, Xi_b), y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds_all, labels_all = [], []
    for Xt_b, Xi_b, y_b in loader:
        logits = model(Xt_b.to(DEVICE), Xi_b.to(DEVICE))
        preds_all.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(y_b.numpy())
    return (accuracy_score(labels_all, preds_all),
            f1_score(labels_all, preds_all, average="macro"))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    txt_feat, img_feat, labels, ids = load_all_features()

    Xt_tr, Xi_tr, y_tr = make_split_tensors(txt_feat, img_feat, labels, ids, f"{SPLITS_DIR}/train.csv")
    Xt_va, Xi_va, y_va = make_split_tensors(txt_feat, img_feat, labels, ids, f"{SPLITS_DIR}/val.csv")
    Xt_te, Xi_te, y_te = make_split_tensors(txt_feat, img_feat, labels, ids, f"{SPLITS_DIR}/test.csv")

    print(f"Train: {len(Xt_tr)}  Val: {len(Xt_va)}  Test: {len(Xt_te)}")
    print(f"Device: {DEVICE}")

    train_loader = make_loader(Xt_tr, Xi_tr, y_tr, shuffle=True)
    val_loader   = make_loader(Xt_va, Xi_va, y_va, shuffle=False)
    test_loader  = make_loader(Xt_te, Xi_te, y_te, shuffle=False)

    model     = LateFusionMLP(text_dim=768, img_dim=2048, num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_val_f1     = 0.0
    best_model_path = os.path.join(RESULTS_DIR, "late_fusion_best.pt")

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

    print("\n===== Late Fusion Results =====")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Macro-F1 : {test_f1:.4f}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()