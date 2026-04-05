"""
extract_text_features.py
Extract Chinese RoBERTa [CLS] embeddings for all notes in metadata.csv
Output:
    features/text_features.npy   shape: (N, 768)
    features/text_labels.npy     shape: (N,)
    features/text_ids.npy        shape: (N,)
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
CSV_PATH   = "data/metadata_balanced.csv"
OUTPUT_DIR = "features"
BATCH_SIZE = 32
MAX_LENGTH = 256
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────────────────────────


def load_model():
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model     = BertModel.from_pretrained(MODEL_NAME)
    model.eval().to(DEVICE)
    print(f"Model loaded on: {DEVICE}")
    return tokenizer, model


def get_cls_embeddings(texts, tokenizer, model):
    """
    Return [CLS] embeddings for a batch of strings.
    Shape: (len(texts), 768)
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model(**encoded)

    # [CLS] token is always the first token
    cls_embeddings = output.last_hidden_state[:, 0, :]   # (batch, 768)
    return cls_embeddings.cpu().numpy()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dataset
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    print(f"Loaded {len(df)} samples from {CSV_PATH}")
    print(df["label"].value_counts())

    # Combine title + content as input text
    # Separator "。" signals a sentence boundary to the model
    texts = (
        df["title"].fillna("") + "。" + df["content"].fillna("")
    ).tolist()

    tokenizer, model = load_model()

    # Extract in batches
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting text features"):
        batch = texts[i : i + BATCH_SIZE]
        emb   = get_cls_embeddings(batch, tokenizer, model)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings)   # (N, 768)

    # Save outputs
    np.save(os.path.join(OUTPUT_DIR, "text_features.npy"), embeddings)
    np.save(os.path.join(OUTPUT_DIR, "text_labels.npy"),   df["label"].values)
    np.save(os.path.join(OUTPUT_DIR, "text_ids.npy"),      df["note_idx"].values)

    print("\nSaved:")
    print(f"  text_features.npy  shape: {embeddings.shape}")
    print(f"  text_labels.npy    shape: {df['label'].values.shape}")
    print(f"  text_ids.npy       shape: {df['note_idx'].values.shape}")


if __name__ == "__main__":
    main()