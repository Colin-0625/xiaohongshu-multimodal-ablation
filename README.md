# When Is Picture Information Really Necessary?
### Ablation Study Based on Multimodal Classification of Xiaohongshu Notes

> **DSAI 5207 Course Project — Hong Kong Polytechnic University**

---

## Research Question

Under what content category and text length conditions does adding image information bring significant improvement over plain text models in Xiaohongshu note classification?

---

## Key Results

| Model | Test Accuracy | Test Macro-F1 | vs. Text-only |
|-------|--------------|---------------|---------------|
| Text-only | 0.8056 | 0.8022 | — |
| Image-only (ResNet50) | 0.6250 | 0.6238 | -17.8% |
| Image-only (ResNet18) | 0.5833 | 0.5807 | -22.1% |
| Late Fusion (ResNet50) | 0.8333 | **0.8344** | +3.2% |
| Late Fusion (ResNet18) | 0.7917 | 0.7849 | -1.7% |
| Gated Fusion | 0.8056 | 0.8074 | +0.5% |

**Key finding:** No fusion model achieves statistical significance over text-only (p > 0.05) at 72 test samples. Image benefit is most reliable for `fashion_beauty` (+6.4% F1) and medium-length posts (202–521 chars, +6.8% gain).

---

## Dataset

- **Source:** [Qilin Dataset (THUIR/Qilin)](https://huggingface.co/datasets/THUIR/Qilin)
- **Size:** 714 samples, 3 balanced classes (238 per class)
- **Categories:** `fashion_beauty` / `food_travel` / `knowledge_tutorial`
- **Split:** 80% train / 10% val / 10% test (stratified, random_state=42)
- **Images:** Extracted from `part_113` to `part_116` (~12 GB raw → 42 MB after extraction)

> ⚠️ Images are stored locally and not included in this repository.  
> See [Image Setup](#image-setup) below for instructions.

---

## Repository Structure

```
xiaohongshu-multimodal-ablation/
├── data/
│   ├── metadata_balanced.csv      # 714 samples with labels
│   └── splits/
│       ├── train.csv              # 570 samples
│       ├── val.csv                # 72 samples
│       └── test.csv               # 72 samples
├── src/
│   ├── split_metadata.py          # Generate train/val/test splits
│   ├── extract_text_features.py   # RoBERTa → 768-dim features
│   ├── extract_image_features.py  # ResNet50 → 2048-dim features
│   ├── train_text_only.py         # Train text-only MLP
│   ├── train_image_only.py        # Train image-only MLP
│   ├── train_late_fusion.py       # Train late fusion model
│   ├── train_gated_fusion.py      # Train gated fusion model
│   └── evaluate.py                # Full evaluation + all figures
├── results/
│   ├── error_cases.csv            # Misclassified test samples
│   └── figures/                   # All generated plots
├── notebooks/
│   └── MDL_run.ipynb              # End-to-end Colab notebook
├── requirements.txt
└── README.md
```

---

## Environment Setup

### Requirements

```bash
pip install -r requirements.txt
```

**Key dependencies:**

```
torch
torchvision
transformers
datasets
pandas
numpy
scikit-learn
pillow
tqdm
matplotlib
seaborn
grad-cam
scipy
```

> This project was developed and run on **Google Colab with NVIDIA T4 GPU**.  
> Local CPU execution is supported but significantly slower for feature extraction.

---

## Image Setup

### Step 1: Download images from Qilin dataset

Download the following archives from the Qilin dataset (Baidu Netdisk mirror):

```
part_113.tar.gz
part_114.tar.gz
part_115.tar.gz
part_116.tar.gz
```

### Step 2: Extract archives

Extract all four archives to a single directory, e.g. `D:/qilin_raw/`:

```
D:/qilin_raw/
├── part_113/
├── part_114/
├── part_115/
└── part_116/
```

### Step 3: Extract only the needed images

Run the extraction script to copy only the 714 required images into the project:

```bash
python src/extract_needed_images.py \
    --source D:/qilin_raw \
    --dest   /path/to/project/data/images \
    --meta   /path/to/project/data/metadata_balanced.csv
```

After this step, the structure should be:

```
data/images/
├── part_113/
├── part_114/
├── part_115/
└── part_116/
```

---

## Reproducing Core Results

### Option A: Google Colab (Recommended)

1. Open `notebooks/MDL_run.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells in order

The notebook handles cloning, dependency installation, feature extraction, training, and evaluation end-to-end.

### Option B: Run scripts manually

Follow this exact order:

```bash
# 1. Generate data splits (run locally, output already in repo)
python src/split_metadata.py

# 2. Extract features (requires GPU for reasonable speed)
python src/extract_text_features.py    # ~5 min on T4
python src/extract_image_features.py   # ~2 min on T4

# 3. Fix label encoding (run once after feature extraction)
python - << 'EOF'
import numpy as np
labels = np.load('features/text_labels.npy', allow_pickle=True)
label2idx = {'fashion_beauty': 0, 'food_travel': 1, 'knowledge_tutorial': 2}
np.save('features/text_labels.npy', np.array([label2idx[l] for l in labels], dtype='int64'))
np.save('features/image_labels.npy', np.load('features/text_labels.npy'))
EOF

# 4. Train all models
python src/train_text_only.py
python src/train_image_only.py
python src/train_late_fusion.py
python src/train_gated_fusion.py

# 5. Evaluate and generate all figures
python src/evaluate.py
```

**Expected outputs:**

| File | Description |
|------|-------------|
| `features/text_features.npy` | (714, 768) RoBERTa embeddings |
| `features/image_features.npy` | (714, 2048) ResNet50 features |
| `results/text_only_best.pt` | Best text-only checkpoint |
| `results/late_fusion_best.pt` | Best late fusion checkpoint |
| `results/gate_weights_test.npy` | Gated fusion gate weights on test set |
| `results/error_cases.csv` | Misclassified samples |
| `results/figures/` | 8 analysis figures |

---

## Generated Figures

| Figure | Description |
|--------|-------------|
| `01_overall_results_table.png` | Accuracy and Macro-F1 for all models |
| `02_per_class_f1.png` | Per-class F1 grouped bar chart |
| `03_confusion_matrices.png` | 2×2 confusion matrix grid |
| `04_category_gain.png` | Per-category F1 gain (Late Fusion vs Text-only) |
| `05_text_length_gain.png` | F1 gain by text length bucket |
| `06_gate_weights.png` | Average gate weight per category |
| `07_error_analysis.png` | Error rate by text length + error count by class |
| `08_gradcam_errors.png` | GradCAM visualization of misclassified samples |

---

## Technical Stack

| Component | Choice |
|-----------|--------|
| Text encoder | `hfl/chinese-roberta-wwm-ext` (frozen) |
| Image encoder | ResNet50 / ResNet18 pretrained (frozen) |
| Classifier | Small MLP (2–3 layers) |
| Training env | Google Colab (T4 GPU) |
| Framework | PyTorch + HuggingFace Transformers |
| Evaluation | scikit-learn, scipy (t-test) |
| Visualization | matplotlib, seaborn, grad-cam |

---

## Citation

If you use the Qilin dataset, please cite:

```
THUIR. (2023). Qilin Dataset.
HuggingFace: https://huggingface.co/datasets/THUIR/Qilin
```

---

## License

This project is for academic purposes only as part of DSAI 5207 at Hong Kong Polytechnic University.