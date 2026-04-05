"""
split_metadata.py
Stratified split of metadata.csv into train / val / test sets.
Output: data/splits/train.csv, val.csv, test.csv
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────────────────────
METADATA_PATH = r"E:\xiaohongshu-multimodal-ablation\data\metadata_balanced.csv"
SPLITS_DIR    = "data/splits"
RANDOM_STATE  = 42
VAL_RATIO     = 0.10   # 10 % of total
TEST_RATIO    = 0.10   # 10 % of total
# ────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)

    df = pd.read_csv(METADATA_PATH)
    print(f"Loaded metadata: {len(df)} samples")
    print(df["label"].value_counts().to_string())

    # First split off test set (10 %)
    train_val, test = train_test_split(
        df,
        test_size=TEST_RATIO,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    # Then split train_val → train (80 %) + val (10 %)
    # val is 10/(100-10) = 1/9 of train_val
    val_ratio_adjusted = VAL_RATIO / (1.0 - TEST_RATIO)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        stratify=train_val["label"],
        random_state=RANDOM_STATE,
    )

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(SPLITS_DIR, f"{name}.csv")
        split.to_csv(path, index=False)
        print(f"Saved {name}: {len(split)} samples → {path}")
        print(f"  Label distribution:\n{split['label'].value_counts().to_string()}\n")

    print("Done.")


if __name__ == "__main__":
    main()