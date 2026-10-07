"""Builder script for 03_leaf_disease_segmentation.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ======================================================================================
md(r"""
# Model 3 — Banana Leaf Disease Identification (healthy vs. diseased + disease type)

**Task:** per CLAUDE.md, "Healthy vs diseased leaf (+ disease type)", architecture **U-Net-type**.

## Architecture decision (open — no ADR exists yet for Model 3)

CLAUDE.md lists Model 3 as `Planned` with no fixed API contract or ADR, unlike Models 1 and 2.
This notebook makes and documents an explicit design choice rather than guessing at one silently:

**Decision: reuse the Model 2 architecture pattern — a multi-label `smp.Unet` with one sigmoid
channel per disease type — instead of a plain whole-image classifier.**

**Why, concretely, from the data (not assumed):**
- The `Leaf Segmentation.v1i.yolov8` export already carries pixel-accurate disease-region
  polygons (the same ones Model 2 uses), so a segmentation head costs nothing extra in labels
  and additionally localizes *where* the disease is — valuable for a farmer-facing product
  ("this patch looks like Black Sigatoka") beyond a bare class label.
- Verified during the audit (Section 2): every image in this dataset carries **exactly one**
  disease category — so the per-image diagnosis CLAUDE.md asks for falls straight out of pooling
  the predicted disease masks (Section 12), no separate classification head needed.
- This keeps Model 3 architecturally consistent with Model 2 (same backbone family, same loss
  family, same post-processing shape) — two closely related leaf-vision problems sharing one
  mental model, rather than an unrelated third pattern.

**This is an experimentation-stage decision, not a production contract.** A real ADR (with the
owner's sign-off, per CLAUDE.md's working agreement) is still needed before this ships.

## Data audit findings

| Finding | Detail |
|---|---|
| Source | Same export as Model 2: `Data/Leaf Segmentation.v1i.yolov8.zip`, 161 images |
| Per-image disease label | `healthy_leaf`=85, `black_sigatoka_disease`=67, `yellow_sigatoka`=7, `cordana`=1, **no disease polygon at all**=1 |
| Excluded from this notebook | `cordana` (1 image — cannot be split into train/val/test, let alone learned) and the 1 image with no disease/healthy polygon (a labelling gap, not a real class) — **159/161 images used**, both exclusions reported again at the end |
| Class imbalance | `yellow_sigatoka` at 7/159 images (~4%) is the dominant risk: Roboflow's own split put **zero** `yellow_sigatoka` images in valid or test. Section 3 uses a **stratified** split (not just group-safe) specifically to keep every class represented in every split. |
""")

# ======================================================================================
md("## 0. Environment Setup")

code(r"""
import sys
sys.path.insert(0, ".")
from common import (
    set_seed, pick_device, extract_zip_once, strip_roboflow_suffix,
    read_yolo_label_file, polygon_row_to_mask,
    dice_score, pixel_confusion, compute_pos_weight, DiceLoss, ComboLoss,
    detect_hardware, recommend_batch_size, save_fig,
)

import os, time, shutil
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

SEED = 42
set_seed(SEED)
DEVICE = pick_device()
print("Training device:", DEVICE)
sns.set_theme(style="whitegrid")
""")

# ======================================================================================
md(r"""
## 1. Configuration

### 1.1 Hardware check — batch size is derived from it, not hard-coded

Same reasoning as Notebook 02 (same architecture family, same 512×512 U-Net memory profile).
""")

code(r"""
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

HW = detect_hardware()
print("Hardware detected:", HW)
if HW["cuda_available"]:
    print(f"GPU: {HW['gpu_name']} ({HW['vram_total_gb']} GB VRAM) -> training will use CUDA.")
else:
    print(f"No CUDA GPU detected -> training on CPU ({HW['cpu_logical_cores']} logical cores).")

RAW_ZIP = Path("../Data/Leaf Segmentation.v1i.yolov8.zip")
RAW_DIR = extract_zip_once(RAW_ZIP, Path("data_cache/leaf_seg_raw"))   # same cache as notebook 02

WORK_DIR = Path("stage3_work")
CACHE_DIR = WORK_DIR / "mask_cache"
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
OUTPUT_DIR = Path("outputs/leaf_disease")
FIGURES_DIR = Path("figures/leaf_disease")
for d in [WORK_DIR, CACHE_DIR, CHECKPOINT_DIR, OUTPUT_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 512          # consistent with Model 2's fixed contract (ADR 0012)
BATCH_SIZE = min(16, recommend_batch_size(HW, base_batch=4, base_ram_gb=8.0)) if not SMOKE_TEST else 4
GRAD_ACCUM_STEPS = 2
ENCODER_NAME = "resnet34"
EPOCHS = 80 if not SMOKE_TEST else 2
PATIENCE = 15 if not SMOKE_TEST else 2
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5

CLASS_ID_TO_NAME = {0: "Leaf_boundary", 1: "black_sigatoka_disease", 2: "cordana",
                     3: "healthy_leaf", 4: "yellow_sigatoka"}
# cordana excluded (1 sample — see title cell). healthy_leaf excluded as a channel (derived, not predicted).
CATEGORY_MAP = {
    "Leaf_boundary": "leaf",
    "black_sigatoka_disease": "black_sigatoka",
    "yellow_sigatoka": "yellow_sigatoka",
}
CHANNEL_NAMES = ["leaf", "black_sigatoka", "yellow_sigatoka"]
DISEASE_CHANNELS = ["black_sigatoka", "yellow_sigatoka"]   # channels used for the image-level diagnosis

print(f"SMOKE_TEST={SMOKE_TEST}  BATCH_SIZE={BATCH_SIZE} (hardware-adapted)  EPOCHS={EPOCHS}  PATIENCE={PATIENCE}")
assert RAW_DIR.exists()
""")

# ======================================================================================
md("## 2. Data Exploration (EDA)\n\n### 2.1 Per-image disease label & exclusions")

code(r"""
def collect_split_rows(split):
    img_dir = RAW_DIR / split / "images"
    lbl_dir = RAW_DIR / split / "labels"
    out = []
    for f in sorted(os.listdir(img_dir)):
        stem = Path(f).stem
        rows = read_yolo_label_file(lbl_dir / f"{stem}.txt")
        out.append({"split": split, "file": f, "stem": stem, "rows": rows})
    return out

all_rows = []
for s in ["train", "valid", "test"]:
    all_rows += collect_split_rows(s)
meta_df = pd.DataFrame(all_rows)

DISEASE_LABEL_MAP = {
    "healthy_leaf": "healthy",
    "black_sigatoka_disease": "black_sigatoka",
    "yellow_sigatoka": "yellow_sigatoka",
    "cordana": "cordana",
}

def image_label(rows):
    present = set(CLASS_ID_TO_NAME[int(r[0])] for r in rows) & set(DISEASE_LABEL_MAP)
    if not present:
        return None   # no disease/healthy polygon at all — a labelling gap
    assert len(present) == 1, f"unexpected multi-disease image: {present}"
    return DISEASE_LABEL_MAP[present.pop()]

meta_df["disease_label"] = meta_df["rows"].apply(image_label)
print("Raw label distribution:")
print(meta_df["disease_label"].value_counts(dropna=False))

excluded = meta_df[meta_df["disease_label"].isin([None, "cordana"])]
print(f"\nExcluding {len(excluded)} images: {excluded['disease_label'].fillna('no_label').value_counts().to_dict()}")
meta_df = meta_df[~meta_df["disease_label"].isin([None, "cordana"])].reset_index(drop=True)
print(f"Remaining: {len(meta_df)} images")

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=meta_df, x="disease_label",
              order=["healthy", "black_sigatoka", "yellow_sigatoka"], ax=ax)
ax.set_title("Image-level disease label distribution (used for training)")
plt.tight_layout()
save_fig(FIGURES_DIR / "01_disease_label_distribution.png")
plt.show()
""")

# ======================================================================================
md("### 2.2 Sample images per disease class")

code(r"""
fig, axes = plt.subplots(3, 3, figsize=(12, 11))
for row, label in enumerate(["healthy", "black_sigatoka", "yellow_sigatoka"]):
    subset = meta_df[meta_df["disease_label"] == label].sample(min(3, (meta_df["disease_label"] == label).sum()), random_state=SEED)
    for ax, (_, r) in zip(axes[row], subset.iterrows()):
        im = Image.open(RAW_DIR / r["split"] / "images" / r["file"])
        ax.imshow(im); ax.set_title(label, fontsize=10); ax.axis("off")
    for ax in axes[row][len(subset):]:
        ax.axis("off")
plt.suptitle("Sample images per disease class")
plt.tight_layout()
save_fig(FIGURES_DIR / "02_sample_images_per_disease.png")
plt.show()
""")

# ======================================================================================
md("## 3. Data Preprocessing\n\n### 3.1 Stratified 70/15/15 split (class-aware, not just group-safe)")

code(r"""
indices = meta_df.index.tolist()
labels = meta_df["disease_label"].tolist()

trainval_idx, test_idx = train_test_split(
    indices, test_size=0.15, random_state=SEED, stratify=labels
)
trainval_labels = meta_df.loc[trainval_idx, "disease_label"].tolist()
train_idx, val_idx = train_test_split(
    trainval_idx, test_size=0.15 / 0.85, random_state=SEED, stratify=trainval_labels
)

split_map = {}
for i in train_idx: split_map[i] = "train"
for i in val_idx:   split_map[i] = "val"
for i in test_idx:  split_map[i] = "test"
meta_df["my_split"] = meta_df.index.map(split_map)

print(meta_df.groupby(["my_split", "disease_label"]).size().unstack(fill_value=0))
print(f"\ntrain={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
print("Every class present in every split:",
      meta_df.groupby("my_split")["disease_label"].nunique().eq(meta_df["disease_label"].nunique()).all())
""")

# ======================================================================================
md("### 3.2 Decode & cache per-channel masks")

code(r"""
def build_channel_masks(rows, h, w):
    out = {name: np.zeros((h, w), dtype=np.uint8) for name in CHANNEL_NAMES}
    for row in rows:
        cname = CLASS_ID_TO_NAME[int(row[0])]
        channel = CATEGORY_MAP.get(cname)
        if channel is None:
            continue
        m = polygon_row_to_mask(row, h, w)
        out[channel] = np.maximum(out[channel], m)
    return out

leaf_ratios, disease_ratios = [], {c: [] for c in DISEASE_CHANNELS}
for idx, r in meta_df.iterrows():
    with Image.open(RAW_DIR / r["split"] / "images" / r["file"]) as im:
        w, h = im.size
    masks = build_channel_masks(r["rows"], h, w)
    stacked = np.stack([masks[c] for c in CHANNEL_NAMES], axis=-1)
    np.save(CACHE_DIR / f"{idx}.npy", stacked)
    leaf_ratios.append(masks["leaf"].sum() / (h * w))
    for c in DISEASE_CHANNELS:
        disease_ratios[c].append(masks[c].sum() / (h * w))

GLOBAL_LEAF_MEAN = float(np.mean(leaf_ratios))
GLOBAL_DISEASE_MEANS = {c: float(np.mean(v)) for c, v in disease_ratios.items()}
print("Leaf pixel coverage (mean):", f"{GLOBAL_LEAF_MEAN*100:.1f}%")
for c, v in GLOBAL_DISEASE_MEANS.items():
    print(f"{c} pixel coverage (mean, whole-image incl. healthy images): {v*100:.3f}%")
print(f"\nCached {len(meta_df)} masks to {CACHE_DIR}")
""")

# ======================================================================================
md("### 3.3 Augmentation pipeline (identical policy to Model 2, for a fair comparison)")

code(r"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_AREA),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.Affine(translate_percent=0.05, scale=(0.85, 1.15), rotate=(-20, 20), p=0.6, border_mode=cv2.BORDER_CONSTANT),
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.15),
    A.GaussNoise(std_range=(0.02, 0.08), p=0.15),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_AREA),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

class LeafDiseaseDataset(Dataset):
    def __init__(self, indices, transform):
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row_idx = self.indices[idx]
        r = meta_df.loc[row_idx]
        img = np.array(Image.open(RAW_DIR / r["split"] / "images" / r["file"]).convert("RGB"))
        mask = np.load(CACHE_DIR / f"{row_idx}.npy")
        augmented = self.transform(image=img, mask=mask)
        img_t = augmented["image"]
        mask_t = augmented["mask"].permute(2, 0, 1).float()
        return img_t, mask_t, r["disease_label"]

def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    labels = [b[2] for b in batch]
    return imgs, masks, labels

train_ds = LeafDiseaseDataset(train_idx, train_transform)
val_ds = LeafDiseaseDataset(val_idx, val_transform)
test_ds = LeafDiseaseDataset(test_idx, val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)

print(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}  test images: {len(test_ds)}")
""")

# ======================================================================================
md("### 3.4 Sanity check: augmented batch with all channels overlaid")

code(r"""
def denormalize(img_t):
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img = img_t.numpy() * std + mean
    return np.clip(img.transpose(1, 2, 0), 0, 1)

imgs, masks, labels = next(iter(train_loader))
n_show = min(4, imgs.shape[0])
fig, axes = plt.subplots(2, n_show, figsize=(3.5 * n_show, 7))
for i in range(n_show):
    axes[0, i].imshow(denormalize(imgs[i])); axes[0, i].set_title(labels[i]); axes[0, i].axis("off")
    axes[1, i].imshow(denormalize(imgs[i]))
    axes[1, i].imshow(masks[i, 0], alpha=0.3, cmap="Greens")
    axes[1, i].imshow(masks[i, 1], alpha=0.5, cmap="Reds")
    axes[1, i].imshow(masks[i, 2], alpha=0.5, cmap="YlOrBr")
    axes[1, i].set_title("leaf(green) blackSig(red) yellowSig(orange)", fontsize=8)
    axes[1, i].axis("off")
plt.tight_layout()
save_fig(FIGURES_DIR / "03_augmented_batch_sanity_check.png")
plt.show()
""")

# ======================================================================================
md(r"""
## 4. Architecture

Same `smp.Unet(resnet34, imagenet)` pattern as Model 2, 3 sigmoid output channels instead of 2.
Multi-label (not softmax) because `leaf` and a disease channel legitimately co-occur per pixel
(a lesion pixel is both "part of the leaf" and "black_sigatoka").
""")

code(r"""
import segmentation_models_pytorch as smp
from thop import profile

def make_model():
    return smp.Unet(encoder_name=ENCODER_NAME, encoder_weights="imagenet",
                     in_channels=3, classes=len(CHANNEL_NAMES), activation=None).to(DEVICE)

model = make_model()
n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {n_params:,}")

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
flops, _ = profile(model, inputs=(dummy,), verbose=False)
print(f"FLOPs (single {IMG_SIZE}x{IMG_SIZE} image): {flops/1e9:.2f} GFLOPs")
""")

# ======================================================================================
md(r"""
## 5. Loss function

Same weighted BCE + Dice combo as Model 2. `yellow_sigatoka`'s pos_weight will be large (it
covers a tiny fraction of total pixels) — printed below so the imbalance is visible, not hidden
inside a single aggregate loss number.
""")

code(r"""
pos_weights = torch.tensor(
    [compute_pos_weight(GLOBAL_LEAF_MEAN)] + [compute_pos_weight(GLOBAL_DISEASE_MEANS[c]) for c in DISEASE_CHANNELS],
    dtype=torch.float32,
).to(DEVICE)
print("Per-channel BCE pos_weight:", dict(zip(CHANNEL_NAMES, pos_weights.tolist())))

criterion = ComboLoss(pos_weights)
""")

# ======================================================================================
md("## 6. Deciding the optimizer")

code(r"""
def quick_train_eval(optimizer_name, n_epochs):
    m = make_model()
    if optimizer_name == "AdamW":
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        opt = torch.optim.SGD(m.parameters(), lr=LR * 10, momentum=0.9, weight_decay=WEIGHT_DECAY)
    for _ in range(n_epochs):
        m.train()
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            loss = criterion(m(imgs), masks)
            loss.backward()
            opt.step()
    m.eval()
    total_dice = torch.zeros(len(CHANNEL_NAMES))
    n_batches = 0
    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            total_dice += dice_score(m(imgs), masks).cpu()
            n_batches += 1
    return float((total_dice / n_batches).mean())

SWEEP_EPOCHS = 1 if SMOKE_TEST else 5
opt_sweep = {name: quick_train_eval(name, SWEEP_EPOCHS) for name in ["AdamW", "SGD"]}
for name, score in opt_sweep.items():
    print(f"{name}: mean val Dice after {SWEEP_EPOCHS} epochs = {score:.4f}")
best_optimizer_name = max(opt_sweep, key=opt_sweep.get)
print(f"\nSelected optimizer: {best_optimizer_name}")

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(opt_sweep.keys(), opt_sweep.values(), color=["#2A6F9A", "#C97B2A"])
ax.set_ylabel("mean val Dice"); ax.set_title(f"Optimizer sweep ({SWEEP_EPOCHS} epochs each)")
plt.tight_layout()
save_fig(FIGURES_DIR / "04_optimizer_sweep.png")
plt.show()
""")

# ======================================================================================
md("## 7. Full training setup (warmup → cosine anneal, AMP, grad accumulation)")

code(r"""
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

model = make_model()
if best_optimizer_name == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=LR * 10, momentum=0.9, weight_decay=WEIGHT_DECAY)

warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(EPOCHS - WARMUP_EPOCHS, 1))
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
""")

# ======================================================================================
md("## 8. Training loop")

code(r"""
def run_epoch(loader, train_mode):
    model.train(train_mode)
    total_loss = 0.0
    total_dice = torch.zeros(len(CHANNEL_NAMES))
    n_batches = 0
    optimizer.zero_grad()
    for step, (imgs, masks, _) in enumerate(loader):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, masks)
            if train_mode:
                loss = loss / GRAD_ACCUM_STEPS
        if train_mode:
            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        total_loss += loss.item() * (GRAD_ACCUM_STEPS if train_mode else 1)
        total_dice += dice_score(logits.detach(), masks).cpu()
        n_batches += 1
    return total_loss / n_batches, (total_dice / n_batches).tolist()
""")

code(r"""
history = {"train_loss": [], "val_loss": [], **{f"val_dice_{c}": [] for c in CHANNEL_NAMES}}
best_val_dice = -1
epochs_no_improve = 0
best_ckpt_path = CHECKPOINT_DIR / "best_model.pt"

for epoch in range(EPOCHS):
    train_loss, _ = run_epoch(train_loader, train_mode=True)
    with torch.no_grad():
        val_loss, val_dice = run_epoch(val_loader, train_mode=False)
    scheduler.step()

    mean_val_dice = float(np.mean(val_dice))
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    for c, d in zip(CHANNEL_NAMES, val_dice):
        history[f"val_dice_{c}"].append(d)

    dice_str = " ".join(f"{c}={d:.3f}" for c, d in zip(CHANNEL_NAMES, val_dice))
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
          f"val_dice[{dice_str}] | lr={optimizer.param_groups[0]['lr']:.2e}")

    if mean_val_dice > best_val_dice:
        best_val_dice = mean_val_dice
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_ckpt_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}.")
            break

print(f"\nBest mean validation Dice: {best_val_dice:.4f}  (checkpoint: {best_ckpt_path})")
""")

# ======================================================================================
md("### 8.1 Training curves")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(history["train_loss"], label="train loss")
axes[0].plot(history["val_loss"], label="val loss")
axes[0].set_title("Loss"); axes[0].legend()
for c in CHANNEL_NAMES:
    axes[1].plot(history[f"val_dice_{c}"], label=f"val Dice ({c})")
axes[1].set_title("Validation Dice per channel"); axes[1].legend()
plt.tight_layout()
save_fig(FIGURES_DIR / "05_training_curves.png")
plt.show()
""")

# ======================================================================================
md("## 9. Pixel-level evaluation on the test set")

code(r"""
model.load_state_dict(torch.load(best_ckpt_path))
model.eval()

channel_stats = {name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for name in CHANNEL_NAMES}
with torch.no_grad():
    for imgs, masks, _ in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        preds = (torch.sigmoid(logits) > 0.5).float().cpu()
        for c, name in enumerate(CHANNEL_NAMES):
            tp, fp, fn, tn = pixel_confusion(preds[:, c], masks[:, c])
            s = channel_stats[name]
            s["tp"] += tp; s["fp"] += fp; s["fn"] += fn; s["tn"] += tn

print(f"{'channel':<16} {'IoU':>7} {'Dice/F1':>9} {'Precision':>10} {'Recall':>8}")
for name in CHANNEL_NAMES:
    s = channel_stats[name]
    iou = s["tp"] / (s["tp"] + s["fp"] + s["fn"] + 1e-7)
    dice = 2 * s["tp"] / (2 * s["tp"] + s["fp"] + s["fn"] + 1e-7)
    precision = s["tp"] / (s["tp"] + s["fp"] + 1e-7)
    recall = s["tp"] / (s["tp"] + s["fn"] + 1e-7)
    print(f"{name:<16} {iou:7.3f} {dice:9.3f} {precision:10.3f} {recall:8.3f}")
""")

# ======================================================================================
md(r"""
### 9.1 Per-channel threshold tuning — on validation

Each disease channel gets its own threshold (rarer classes often need a lower bar), tuned on
`val_loader` and only then applied once to the held-out test set.
""")

code(r"""
def collect_probs_targets(loader):
    all_logits, all_targets = [], []
    with torch.no_grad():
        for imgs, masks, _ in loader:
            logits = model(imgs.to(DEVICE)).cpu()
            all_logits.append(logits)
            all_targets.append(masks)
    return torch.cat(all_logits), torch.cat(all_targets)

val_logits, val_targets = collect_probs_targets(val_loader)
thresholds_grid = np.arange(0.1, 0.95, 0.05)
best_thresholds = {}

fig, axes = plt.subplots(1, len(DISEASE_CHANNELS), figsize=(6 * len(DISEASE_CHANNELS), 4.5))
if len(DISEASE_CHANNELS) == 1:
    axes = [axes]
for ax, cname in zip(axes, DISEASE_CHANNELS):
    cidx = CHANNEL_NAMES.index(cname)
    probs = torch.sigmoid(val_logits[:, cidx])
    targets = val_targets[:, cidx]
    ious = []
    for t in thresholds_grid:
        pred = (probs > t).float()
        tp = ((pred == 1) & (targets == 1)).sum().item()
        fp = ((pred == 1) & (targets == 0)).sum().item()
        fn = ((pred == 0) & (targets == 1)).sum().item()
        ious.append(tp / (tp + fp + fn + 1e-7))
    best_t = float(thresholds_grid[int(np.argmax(ious))])
    best_thresholds[cname] = best_t
    ax.plot(thresholds_grid, ious, marker="o")
    ax.set_title(f"{cname} (best t={best_t:.2f}, IoU={max(ious):.3f})")
    ax.set_xlabel("threshold"); ax.set_ylabel("val IoU")
plt.tight_layout()
save_fig(FIGURES_DIR / "06_per_channel_threshold_sweep.png")
plt.show()
print("Val-tuned thresholds:", best_thresholds)
""")

# ======================================================================================
md(r"""
## 10. Image-level diagnosis: deriving `healthy | black_sigatoka | yellow_sigatoka`

A pixel-level model answers "where", but CLAUDE.md's Model 3 goal is "healthy vs diseased leaf
(+ disease type)" — an **image-level** verdict. We derive it by pooling: the leaf mask gates what
counts as "on the leaf", then whichever disease channel covers the largest leaf-relative area
(above its own val-tuned threshold) wins; no disease channel above threshold → `healthy`.
""")

code(r"""
def diagnose(logits, leaf_thresh=0.5, min_leaf_frac=0.005):
    leaf_mask = torch.sigmoid(logits[0]) > leaf_thresh
    leaf_px = leaf_mask.sum().item()
    scores = {}
    for cname in DISEASE_CHANNELS:
        cidx = CHANNEL_NAMES.index(cname)
        disease_mask = torch.sigmoid(logits[cidx]) > best_thresholds[cname]
        overlap = (disease_mask & leaf_mask).sum().item()
        scores[cname] = overlap / max(leaf_px, 1)
    top_disease, top_score = max(scores.items(), key=lambda kv: kv[1])
    return top_disease if top_score >= min_leaf_frac else "healthy"

y_true, y_pred = [], []
with torch.no_grad():
    for i in range(len(test_ds)):
        img_t, mask_t, true_label = test_ds[i]
        logits = model(img_t.unsqueeze(0).to(DEVICE))[0].cpu()
        y_true.append(true_label)
        y_pred.append(diagnose(logits))

labels_order = ["healthy", "black_sigatoka", "yellow_sigatoka"]
cm = confusion_matrix(y_true, y_pred, labels=labels_order)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_order, yticklabels=labels_order, ax=ax)
ax.set_xlabel("Predicted diagnosis"); ax.set_ylabel("True label")
ax.set_title("Image-level diagnosis — test set")
plt.tight_layout()
save_fig(FIGURES_DIR / "07_diagnosis_confusion_matrix.png")
plt.show()

print(classification_report(y_true, y_pred, labels=labels_order, zero_division=0))
""")

# ======================================================================================
md("## 11. Inference speed profiling")

code(r"""
sample_img, _, _ = test_ds[0]
sample_img = sample_img.unsqueeze(0).to(DEVICE)
for _ in range(5):
    with torch.no_grad():
        _ = model(sample_img)
n_runs = 20
t0 = time.perf_counter()
for _ in range(n_runs):
    with torch.no_grad():
        _ = model(sample_img)
elapsed = time.perf_counter() - t0
latency_ms = (elapsed / n_runs) * 1000
print(f"Device: {DEVICE}  |  Avg latency: {latency_ms:.1f} ms/image  ({1000/latency_ms:.1f} FPS)")
""")

# ======================================================================================
md("## 12. Qualitative analysis — misdiagnosed test images")

code(r"""
wrong = [(i, t, p) for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
print(f"{len(wrong)}/{len(y_true)} test images misdiagnosed")

if wrong:
    n_show = min(4, len(wrong))
    fig, axes = plt.subplots(1, n_show, figsize=(4.5 * n_show, 4.5))
    if n_show == 1:
        axes = [axes]
    for ax, (idx, t, p) in zip(axes, wrong[:n_show]):
        img_t, _, _ = test_ds[idx]
        ax.imshow(denormalize(img_t))
        ax.set_title(f"true={t}\npred={p}", fontsize=10)
        ax.axis("off")
    plt.suptitle("Misdiagnosed test images")
    plt.tight_layout()
    save_fig(FIGURES_DIR / "08_misdiagnosed_examples.png")
    plt.show()
else:
    print("No misdiagnoses on the test set.")
""")

# ======================================================================================
md(r"""
## 13. Known limitations & next steps

- **`yellow_sigatoka` has only 7 images total** — even with stratified splitting, the test-set
  estimate for this class rests on ~1 image. Treat its precision/recall as directional, not
  reliable, until more data exists.
- **`cordana` (1 image) and 1 unlabeled image were excluded outright** — both are real data gaps,
  not something a training trick can fix; flagged for the owner to source more examples.
- **No ADR for this architecture yet** — Section 0's decision (segmentation-based diagnosis vs. a
  plain classifier) needs owner sign-off before it becomes the production contract, per CLAUDE.md's
  working agreement.
- **CPU-only training**, same caveat as Models 1 and 2.
- **Image-level diagnosis threshold (`min_leaf_frac=0.005`) is a reasonable default, not tuned**
  — unlike the per-channel pixel thresholds in Section 9.1, this final pooling threshold wasn't
  swept; a natural next step once more validation data exists.

## 14. Save model
""")

code(r"""
shutil.copy2(best_ckpt_path, OUTPUT_DIR / "leaf_disease_best.pt")
print(f"Saved: {(OUTPUT_DIR / 'leaf_disease_best.pt').resolve()}")
""")

nb["cells"] = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3 (keraai-experiment)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

with open("03_leaf_disease_segmentation.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Wrote 03_leaf_disease_segmentation.ipynb with", len(cells), "cells")
