"""
extract_image_features.py
Extract 2048-dim ResNet50 features for all samples in metadata.csv.
Output: features/image_features.npy, image_labels.npy, image_ids.npy
Sample order is aligned with metadata.csv row order (same as text features).
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
METADATA_PATH = "data/metadata_balanced.csv"
IMAGES_ROOT   = "data/images"          # extracted tar.gz lives here
FEATURES_DIR  = "features"
BATCH_SIZE    = 64
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

LABEL2IDX = {
    "fashion_beauty":    0,
    "food_travel":       1,
    "knowledge_tutorial": 2,
}


def resolve_image_path(image_path_csv: str) -> str:
    """
    CSV stores paths like 'image/part_113/1119356/5012641.jpg'.
    Map to: data/images/part_113/1119356/5012641.jpg
    """
    rel = image_path_csv.replace("image/", "", 1)
    return os.path.join(IMAGES_ROOT, rel)


class XHSImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        img_path   = resolve_image_path(row["image_path"])
        label      = LABEL2IDX[row["label"]]
        note_idx   = row["note_idx"]

        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            print(f"[WARN] Failed to load {img_path}: {e}. Using zero tensor.")
            # Return a zero tensor with the expected shape (3, 224, 224)
            img = torch.zeros(3, 224, 224)

        return img, label, note_idx


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)

    df = pd.read_csv(METADATA_PATH)
    print(f"Loaded metadata: {len(df)} samples")

    # Load pretrained ResNet50 and remove the final FC layer
    weights = ResNet50_Weights.DEFAULT
    model   = resnet50(weights=weights)
    model.fc = torch.nn.Identity()          # output → 2048-dim
    model    = model.to(DEVICE)
    model.eval()

    transform = weights.transforms()        # official preprocessing pipeline

    dataset    = XHSImageDataset(df, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,                      # must keep order aligned with metadata.csv
        num_workers=4,
        pin_memory=(DEVICE == "cuda"),
    )

    all_features = []
    all_labels   = []
    all_ids      = []

    with torch.no_grad():
        for imgs, labels, ids in tqdm(dataloader, desc="Extracting image features"):
            imgs = imgs.to(DEVICE)
            feats = model(imgs)             # (B, 2048)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
            all_ids.extend(ids if isinstance(ids, list) else ids.numpy().tolist())

    features = np.vstack(all_features).astype(np.float32)   # (N, 2048)
    labels   = np.concatenate(all_labels).astype(np.int64)   # (N,)
    ids      = np.array(all_ids)

    np.save(os.path.join(FEATURES_DIR, "image_features.npy"), features)
    np.save(os.path.join(FEATURES_DIR, "image_labels.npy"),   labels)
    np.save(os.path.join(FEATURES_DIR, "image_ids.npy"),      ids)

    print(f"Saved image_features.npy: {features.shape}")
    print(f"Saved image_labels.npy:   {labels.shape}")
    print(f"Saved image_ids.npy:      {ids.shape}")
    print("Done.")


if __name__ == "__main__":
    main()