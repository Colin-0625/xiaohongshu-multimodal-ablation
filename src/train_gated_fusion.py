"""
train_gated_fusion.py
Train the gated fusion model: gate weight is dynamically computed from
text projection, image projection, and normalized text length.
Input:  features/text_features.npy + image_features.npy
        data/splits/train.csv, val.csv, test.csv  (must contain content_length)
Output: results/gated_fusion_best.pt
        results/gate_weights_test.npy   (gate weights on test set, for analysis)
        results/gate_labels_test.npy    (ground-truth labels, aligned)
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
METADATA_PATH = "data/metadata_balanced.csv"
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
class GatedFusion(nn.Module):
    def __init__(self, text_dim=768, img_dim=2048, num_classes=3):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, 256)
        self.img_proj  = nn.Linear(img_dim,  256)
        # Gate takes concatenation of both projections + 1 scalar (text_len)
        self.gate = nn.Sequential(
            nn.Linear(256 + 256 + 1, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, text_feat, img_feat, text_len):
        t = self.text_proj(text_feat)            # (B, 256)
        v = self.img_proj(img_feat)              # (B, 256)
        gate_in = torch.cat([t, v, text_len], dim=1)   # (B, 513)
        gate_w  = self.gate(gate_in)             # (B, 1)  ∈ [0, 1]
        # gate_w → weight on image; (1 - gate_w) → weight on text
        fused   = gate_w * v + (1 - gate_w) * t # (B, 256)
        logits  = self.classifier(fused)         # (B, num_classes)
        return logits, gate_w


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_all_features():
    txt_feat = np.load(os.path.join(FEATURES_DIR, "text_features.npy"))
    txt_lbl  = np.load(os.path.join(FEATURES_DIR, "text_labels.npy"))
    txt_ids  = np.load(os.path.join(FEATURES_DIR, "text_ids.npy"))
    img_feat = np.load(os.path.join(FEATURES_DIR, "image_features.npy"))
    img_ids  = np.load(os.path.join(FEATURES_DIR, "image_ids.npy"))
    assert np.array_equal(txt_ids, img_ids), \
        "Feature files are not aligned! Re-run extraction scripts."
    return txt_feat, img_feat, txt_lbl, txt_ids


def get_text_lengths(ids, metadata_df, max_len):
    """Return normalized content_length for each note_idx in order."""
    id_to_len = dict(zip(metadata_df["note_idx"], metadata_df["content_length"]))
    lengths   = np.array([id_to_len[i] for i in ids], dtype=np.float32)
    return lengths / max_len


def make_split_tensors(txt_feat, img_feat, labels, ids, norm_lengths, split_csv):
    split_ids = set(pd.read_csv(split_csv)["note_idx"].tolist())
    mask      = np.array([i in split_ids for i in ids])
    Xt = torch.tensor(txt_feat[mask],    dtype=torch.float32)
    Xi = torch.tensor(img_feat[mask],    dtype=torch.float32)
    Xl = torch.tensor(norm_lengths[mask], dtype=torch.float32).unsqueeze(1)
    y  = torch.tensor(labels[mask],       dtype=torch.long)
    return Xt, Xi, Xl, y


def make_loader(Xt, Xi, Xl, y, shuffle=True):
    ds = TensorDataset(Xt, Xi, Xl, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# ── Training / Eval ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for Xt_b, Xi_b, Xl_b, y_b in loader:
        Xt_b = Xt_b.to(DEVICE)
        Xi_b = Xi_b.to(DEVICE)
        Xl_b = Xl_b.to(DEVICE)
        y_b  = y_b.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(Xt_b, Xi_b, Xl_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, return_gates=False):
    model.eval()
    preds_all, labels_all, gates_all = [], [], []
    for Xt_b, Xi_b, Xl_b, y_b in loader:
        logits, gates = model(
            Xt_b.to(DEVICE), Xi_b.to(DEVICE), Xl_b.to(DEVICE)
        )
        preds_all.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(y_b.numpy())
        gates_all.extend(gates.squeeze(1).cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average="macro")
    if return_gates:
        return acc, f1, np.array(preds_all), np.array(labels_all), np.array(gates_all)
    return acc, f1


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    txt_feat, img_feat, labels, ids = load_all_features()

    metadata_df = pd.read_csv(METADATA_PATH)
    max_len     = metadata_df["content_length"].max()
    norm_lengths = get_text_lengths(ids, metadata_df, max_len)

    print(f"Max content length in dataset: {max_len}")

    Xt_tr, Xi_tr, Xl_tr, y_tr = make_split_tensors(
        txt_feat, img_feat, labels, ids, norm_lengths, f"{SPLITS_DIR}/train.csv")
    Xt_va, Xi_va, Xl_va, y_va = make_split_tensors(
        txt_feat, img_feat, labels, ids, norm_lengths, f"{SPLITS_DIR}/val.csv")
    Xt_te, Xi_te, Xl_te, y_te = make_split_tensors(
        txt_feat, img_feat, labels, ids, norm_lengths, f"{SPLITS_DIR}/test.csv")

    print(f"Train: {len(Xt_tr)}  Val: {len(Xt_va)}  Test: {len(Xt_te)}")
    print(f"Device: {DEVICE}")

    train_loader = make_loader(Xt_tr, Xi_tr, Xl_tr, y_tr, shuffle=True)
    val_loader   = make_loader(Xt_va, Xi_va, Xl_va, y_va, shuffle=False)
    test_loader  = make_loader(Xt_te, Xi_te, Xl_te, y_te, shuffle=False)

    model     = GatedFusion(text_dim=768, img_dim=2048, num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_val_f1     = 0.0
    best_model_path = os.path.join(RESULTS_DIR, "gated_fusion_best.pt")

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

    # Load best, evaluate on test set, and save gate weights for analysis
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_acc, test_f1, _, test_labels_arr, gate_weights = evaluate(
        model, test_loader, return_gates=True
    )

    np.save(os.path.join(RESULTS_DIR, "gate_weights_test.npy"), gate_weights)
    np.save(os.path.join(RESULTS_DIR, "gate_labels_test.npy"),  test_labels_arr)

    print("\n===== Gated Fusion Results =====")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Macro-F1 : {test_f1:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Gate weights saved to: {RESULTS_DIR}/gate_weights_test.npy")

    # Quick gate analysis
    label_names = ["fashion_beauty", "food_travel", "knowledge_tutorial"]
    print("\nAverage gate weight (image reliance) per class:")
    for idx, name in enumerate(label_names):
        mask = test_labels_arr == idx
        avg_gate = gate_weights[mask].mean() if mask.sum() > 0 else float("nan")
        print(f"  {name}: {avg_gate:.4f}")


if __name__ == "__main__":
    main()