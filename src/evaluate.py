"""
evaluate.py
Comprehensive evaluation of all 4 models. Produces:
  1. Overall results table (Accuracy + Macro-F1)
  2. Per-class F1 bar chart
  3. Confusion matrices (2×2 subplot grid)
  4. By-category gain: late_fusion vs text_only per class
  5. By-text-length gain: short / medium / long
  6. Gate weight analysis: avg gate weight per class (gated_fusion)
All figures saved to results/figures/
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# ── Config ───────────────────────────────────────────────────────────────────
FEATURES_DIR  = "features"
SPLITS_DIR    = "data/splits"
RESULTS_DIR   = "results"
FIGURES_DIR   = os.path.join(RESULTS_DIR, "figures")
METADATA_PATH = "data/metadata.csv"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_NAMES   = ["fashion_beauty", "food_travel", "knowledge_tutorial"]
LABEL2IDX     = {n: i for i, n in enumerate(LABEL_NAMES)}
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Model definitions (must match training scripts exactly)
# ══════════════════════════════════════════════════════════════════════════════
class TextMLP(nn.Module):
    def __init__(self, in_dim=768, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)


class ImageMLP(nn.Module):
    def __init__(self, in_dim=2048, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)


class LateFusionMLP(nn.Module):
    def __init__(self, text_dim=768, img_dim=2048, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim + img_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, text_feat, img_feat):
        return self.net(torch.cat([text_feat, img_feat], dim=1))


class GatedFusion(nn.Module):
    def __init__(self, text_dim=768, img_dim=2048, num_classes=3):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, 256)
        self.img_proj  = nn.Linear(img_dim,  256)
        self.gate = nn.Sequential(
            nn.Linear(256 + 256 + 1, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.classifier = nn.Linear(256, num_classes)
    def forward(self, text_feat, img_feat, text_len):
        t = self.text_proj(text_feat)
        v = self.img_proj(img_feat)
        gate_w = self.gate(torch.cat([t, v, text_len], dim=1))
        fused  = gate_w * v + (1 - gate_w) * t
        return self.classifier(fused), gate_w


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════
def load_features():
    txt_feat  = np.load(f"{FEATURES_DIR}/text_features.npy")
    txt_lbl   = np.load(f"{FEATURES_DIR}/text_labels.npy")
    txt_ids   = np.load(f"{FEATURES_DIR}/text_ids.npy")
    img_feat  = np.load(f"{FEATURES_DIR}/image_features.npy")
    img_ids   = np.load(f"{FEATURES_DIR}/image_ids.npy")
    assert np.array_equal(txt_ids, img_ids), "Feature files not aligned!"
    return txt_feat, img_feat, txt_lbl, txt_ids


def get_test_mask(ids):
    test_ids = set(pd.read_csv(f"{SPLITS_DIR}/test.csv")["note_idx"].tolist())
    return np.array([i in test_ids for i in ids])


def get_normalized_lengths(ids, metadata_df):
    max_len   = metadata_df["content_length"].max()
    id_to_len = dict(zip(metadata_df["note_idx"], metadata_df["content_length"]))
    lengths   = np.array([id_to_len[i] for i in ids], dtype=np.float32)
    return lengths / max_len, np.array([id_to_len[i] for i in ids])  # normed + raw


# ══════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_text_only(model, X):
    model.eval()
    logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
    return logits.argmax(1).cpu().numpy()


@torch.no_grad()
def predict_image_only(model, X):
    model.eval()
    logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
    return logits.argmax(1).cpu().numpy()


@torch.no_grad()
def predict_late_fusion(model, Xt, Xi):
    model.eval()
    logits = model(
        torch.tensor(Xt, dtype=torch.float32).to(DEVICE),
        torch.tensor(Xi, dtype=torch.float32).to(DEVICE),
    )
    return logits.argmax(1).cpu().numpy()


@torch.no_grad()
def predict_gated_fusion(model, Xt, Xi, Xl):
    model.eval()
    logits, gates = model(
        torch.tensor(Xt, dtype=torch.float32).to(DEVICE),
        torch.tensor(Xi, dtype=torch.float32).to(DEVICE),
        torch.tensor(Xl, dtype=torch.float32).unsqueeze(1).to(DEVICE),
    )
    return logits.argmax(1).cpu().numpy(), gates.squeeze(1).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ══════════════════════════════════════════════════════════════════════════════
def save_fig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis functions
# ══════════════════════════════════════════════════════════════════════════════
def plot_overall_table(results):
    """Print and save overall accuracy + macro-F1 table."""
    print("\n" + "="*55)
    print(f"{'Model':<20} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-"*55)
    for name, (acc, f1) in results.items():
        print(f"{name:<20} {acc:>10.4f} {f1:>10.4f}")
    print("="*55)

    # Save as a simple matplotlib table image
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.axis("off")
    table_data = [[n, f"{acc:.4f}", f"{f1:.4f}"] for n, (acc, f1) in results.items()]
    tbl = ax.table(
        cellText=table_data,
        colLabels=["Model", "Accuracy", "Macro-F1"],
        cellLoc="center", loc="center"
    )
    tbl.scale(1, 1.6)
    plt.title("Overall Results", fontsize=13, pad=10)
    save_fig("01_overall_results_table.png")


def plot_per_class_f1(all_preds, y_test):
    """Per-class F1 grouped bar chart."""
    model_names = list(all_preds.keys())
    n_classes   = len(LABEL_NAMES)
    x = np.arange(n_classes)
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(model_names):
        f1s = f1_score(y_test, all_preds[name], average=None)
        ax.bar(x + i * width, f1s, width, label=name)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(LABEL_NAMES, rotation=10)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score by Model")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_fig("02_per_class_f1.png")


def plot_confusion_matrices(all_preds, y_test):
    """2×2 grid of confusion matrices."""
    model_names = list(all_preds.keys())
    fig, axes   = plt.subplots(2, 2, figsize=(12, 10))

    for ax, name in zip(axes.flatten(), model_names):
        cm = confusion_matrix(y_test, all_preds[name])
        sns.heatmap(
            cm, annot=True, fmt="d", ax=ax,
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            cmap="Blues"
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Confusion Matrices", fontsize=14)
    plt.tight_layout()
    save_fig("03_confusion_matrices.png")


def plot_category_gain(preds_text, preds_late, y_test):
    """Per-category F1 gain: late_fusion - text_only."""
    f1_text = f1_score(y_test, preds_text, average=None)
    f1_late = f1_score(y_test, preds_late, average=None)
    gains   = f1_late - f1_text

    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ["#4CAF50" if g >= 0 else "#F44336" for g in gains]
    bars    = ax.bar(LABEL_NAMES, gains, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("F1 Gain (Late Fusion − Text-Only)")
    ax.set_title("Per-Category F1 Gain from Adding Image")

    for bar, val in zip(bars, gains):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.002 if val >= 0 else -0.012),
            f"{val:+.4f}", ha="center", va="bottom", fontsize=10
        )
    plt.tight_layout()
    save_fig("04_category_gain.png")

    print("\nPer-Category F1 Gain (Late Fusion vs Text-Only):")
    for name, g in zip(LABEL_NAMES, gains):
        print(f"  {name}: {g:+.4f}")


def plot_text_length_gain(preds_text, preds_late, y_test, raw_lengths):
    """F1 gain across short / medium / long text-length buckets."""
    q33 = np.percentile(raw_lengths, 33)
    q66 = np.percentile(raw_lengths, 66)

    buckets = {
        f"Short (≤{int(q33)})":    raw_lengths <= q33,
        f"Medium ({int(q33)+1}–{int(q66)})": (raw_lengths > q33) & (raw_lengths <= q66),
        f"Long (>{int(q66)})":     raw_lengths > q66,
    }

    gains = {}
    for label, mask in buckets.items():
        if mask.sum() == 0:
            gains[label] = 0.0
            continue
        f1_t = f1_score(y_test[mask], preds_text[mask], average="macro", zero_division=0)
        f1_l = f1_score(y_test[mask], preds_late[mask], average="macro", zero_division=0)
        gains[label] = f1_l - f1_t

    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ["#4CAF50" if g >= 0 else "#F44336" for g in gains.values()]
    bars    = ax.bar(list(gains.keys()), list(gains.values()), color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Macro-F1 Gain (Late Fusion − Text-Only)")
    ax.set_title("Image Benefit by Text Length")

    for bar, val in zip(bars, gains.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.001 if val >= 0 else -0.008),
            f"{val:+.4f}", ha="center", va="bottom", fontsize=10
        )
    plt.tight_layout()
    save_fig("05_text_length_gain.png")

    print("\nF1 Gain by Text Length:")
    for bucket, g in gains.items():
        print(f"  {bucket}: {g:+.4f}")


def plot_gate_weights(gate_weights, y_test):
    """Average gate weight per class for gated_fusion."""
    avg_gates = [gate_weights[y_test == i].mean() for i in range(len(LABEL_NAMES))]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars    = ax.bar(LABEL_NAMES, avg_gates, color="#2196F3")
    ax.set_ylabel("Average Gate Weight (image reliance)")
    ax.set_title("Gated Fusion: Image Reliance per Category")
    ax.set_ylim(0, 1)

    for bar, val in zip(bars, avg_gates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=11
        )
    plt.tight_layout()
    save_fig("06_gate_weights.png")

    print("\nGated Fusion — Avg Gate Weight per Class (higher = more image reliance):")
    for name, w in zip(LABEL_NAMES, avg_gates):
        print(f"  {name}: {w:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Load features ──────────────────────────────────────────────────────
    print("Loading features …")
    txt_feat, img_feat, all_labels, all_ids = load_features()

    metadata_df = pd.read_csv(METADATA_PATH)
    norm_lengths, raw_lengths = get_normalized_lengths(all_ids, metadata_df)

    test_mask  = get_test_mask(all_ids)
    Xt_te      = txt_feat[test_mask]
    Xi_te      = img_feat[test_mask]
    Xl_te      = norm_lengths[test_mask]
    y_test     = all_labels[test_mask]
    raw_te_len = raw_lengths[test_mask]

    print(f"Test samples: {len(y_test)}")

    # ── Load models ────────────────────────────────────────────────────────
    print("Loading model weights …")
    text_model  = TextMLP(768,  3).to(DEVICE)
    img_model   = ImageMLP(2048, 3).to(DEVICE)
    late_model  = LateFusionMLP(768, 2048, 3).to(DEVICE)
    gated_model = GatedFusion(768, 2048, 3).to(DEVICE)

    text_model.load_state_dict(
        torch.load(f"{RESULTS_DIR}/text_only_best.pt",    map_location=DEVICE))
    img_model.load_state_dict(
        torch.load(f"{RESULTS_DIR}/image_only_best.pt",   map_location=DEVICE))
    late_model.load_state_dict(
        torch.load(f"{RESULTS_DIR}/late_fusion_best.pt",  map_location=DEVICE))
    gated_model.load_state_dict(
        torch.load(f"{RESULTS_DIR}/gated_fusion_best.pt", map_location=DEVICE))

    # ── Get predictions ───────────────────────────────────────────────────
    print("Running inference …")
    preds_text  = predict_text_only(text_model, Xt_te)
    preds_img   = predict_image_only(img_model, Xi_te)
    preds_late  = predict_late_fusion(late_model, Xt_te, Xi_te)
    preds_gated, gate_weights = predict_gated_fusion(gated_model, Xt_te, Xi_te, Xl_te)

    all_preds = {
        "text-only":    preds_text,
        "image-only":   preds_img,
        "late fusion":  preds_late,
        "gated fusion": preds_gated,
    }

    # ── Print classification reports ──────────────────────────────────────
    for name, preds in all_preds.items():
        print(f"\n── {name} ──")
        print(classification_report(y_test, preds, target_names=LABEL_NAMES))

    # ── 1. Overall results table ──────────────────────────────────────────
    print("\nGenerating plots …")
    overall = {
        name: (accuracy_score(y_test, p), f1_score(y_test, p, average="macro"))
        for name, p in all_preds.items()
    }
    plot_overall_table(overall)

    # ── 2. Per-class F1 bar chart ─────────────────────────────────────────
    plot_per_class_f1(all_preds, y_test)

    # ── 3. Confusion matrices ─────────────────────────────────────────────
    plot_confusion_matrices(all_preds, y_test)

    # ── 4. By-category gain ───────────────────────────────────────────────
    plot_category_gain(preds_text, preds_late, y_test)

    # ── 5. By-text-length gain ────────────────────────────────────────────
    plot_text_length_gain(preds_text, preds_late, y_test, raw_te_len)

    # ── 6. Gate weight analysis ───────────────────────────────────────────
    plot_gate_weights(gate_weights, y_test)

    print(f"\nAll figures saved to: {FIGURES_DIR}/")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()