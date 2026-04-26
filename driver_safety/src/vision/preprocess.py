"""
preprocess.py — Kaggle Drowsiness Dataset Loading & Preprocessing

Kaggle Dataset Structure:
  data/raw/
  ├── train/
  │   ├── Open/     (alert  → label 0)
  │   └── Closed/   (drowsy → label 1)
  └── test/
      ├── Open/
      └── Closed/

Images are grayscale eye crops (various sizes → resized to 64×64).

Download from:
  https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset
  or https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset
"""

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    TRAIN_DIR, TEST_DIR, PROC_DIR,
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
    CLASS_FOLDERS, VALIDATION_SPLIT, RANDOM_SEED
)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─── Core Loaders ────────────────────────────────────────────────────────────

def load_images_from_folder(root_dir, augment_minority=True):
    """
    Load all eye images from the Kaggle folder structure.
    Applies CLAHE histogram equalisation for lighting robustness.

    Returns:
        X : np.ndarray  (N, H, W, 1)  float32 in [0,1]
        y : np.ndarray  (N,)          int
    """
    images, labels = [], []
    class_counts   = {}

    for class_name, label in CLASS_FOLDERS.items():
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            # Try case-insensitive match
            for d in os.listdir(root_dir):
                if d.lower() == class_name.lower():
                    class_dir = os.path.join(root_dir, d)
                    break

        if not os.path.isdir(class_dir):
            print(f"  [WARN] Folder not found: {class_dir}")
            continue

        files = [f for f in os.listdir(class_dir)
                 if os.path.splitext(f)[1].lower() in VALID_EXTS]

        print(f"  {class_name}: {len(files)} images")
        class_counts[label] = len(files)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

        for fname in tqdm(files, desc=f"  Loading {class_name}", leave=False):
            fpath = os.path.join(class_dir, fname)
            img   = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = clahe.apply(img)          # Lighting normalisation
            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(label)

    if not images:
        raise FileNotFoundError(
            f"No images found under: {root_dir}\n"
            f"Expected subfolders: {list(CLASS_FOLDERS.keys())}\n"
            f"Download from: https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset"
        )

    X = np.array(images, dtype=np.float32)
    X = np.expand_dims(X, axis=-1)          # (N, H, W, 1)
    y = np.array(labels, dtype=np.int32)

    print(f"\n  Loaded {len(X)} images — "
          f"Alert:{class_counts.get(0,0)}, Drowsy:{class_counts.get(1,0)}")
    return X, y


def balance_classes(X, y):
    """Undersample majority class so both classes are equal size."""
    np.random.seed(RANDOM_SEED)
    classes, counts = np.unique(y, return_counts=True)
    n = counts.min()
    idx = []
    for c in classes:
        ci = np.where(y == c)[0]
        idx.extend(np.random.choice(ci, n, replace=False))
    idx = np.array(idx)
    np.random.shuffle(idx)
    print(f"  Balanced: {n} samples per class ({len(idx)} total)")
    return X[idx], y[idx]


def augment_images(X, y, factor=2):
    """
    Light augmentation to increase training set diversity.
    Applied ONLY to training split (not val/test).
    Augmentations: horizontal flip, small rotation, brightness jitter.
    """
    aug_X, aug_y = [X], [y]
    np.random.seed(RANDOM_SEED)

    for _ in range(factor - 1):
        X_aug = X.copy()
        for i in range(len(X_aug)):
            img = X_aug[i, :, :, 0]

            # Horizontal flip (50%)
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)

            # Small rotation ±10°
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D(
                (IMG_WIDTH / 2, IMG_HEIGHT / 2), angle, 1.0
            )
            img = cv2.warpAffine(img, M, (IMG_WIDTH, IMG_HEIGHT))

            # Brightness jitter ±15%
            delta = np.random.uniform(-0.15, 0.15)
            img   = np.clip(img + delta, 0.0, 1.0)

            X_aug[i, :, :, 0] = img

        aug_X.append(X_aug)
        aug_y.append(y)

    return np.concatenate(aug_X), np.concatenate(aug_y)


# ─── Public API ───────────────────────────────────────────────────────────────

def load_kaggle_dataset():
    """
    Load Kaggle eye image dataset, balance classes,
    split into train / val / test.

    Returns
    -------
    X_train, X_val, X_test : (N, 64, 64, 1) float32
    y_train, y_val, y_test : (N,) int
    """
    os.makedirs(PROC_DIR, exist_ok=True)
    cache = os.path.join(PROC_DIR, "dataset.npz")

    if os.path.exists(cache):
        print("[INFO] Loading cached dataset...")
        d = np.load(cache)
        return d["X_tr"], d["X_v"], d["X_te"], d["y_tr"], d["y_v"], d["y_te"]

    print("\n[DATA] Loading Kaggle Drowsiness Dataset...")

    # Try train/test split folders first
    if os.path.isdir(TRAIN_DIR) and os.path.isdir(TEST_DIR):
        print(f"[DATA] Train folder: {TRAIN_DIR}")
        X_tr_raw, y_tr_raw = load_images_from_folder(TRAIN_DIR)
        print(f"[DATA] Test  folder: {TEST_DIR}")
        X_te, y_te = load_images_from_folder(TEST_DIR)

    elif os.path.isdir(os.path.join(
            os.path.dirname(TRAIN_DIR), "Open")):
        # Flat structure: data/raw/Open, data/raw/Closed
        from utils.config import RAW_DIR
        print(f"[DATA] Flat folder: {RAW_DIR}")
        X_all, y_all = load_images_from_folder(RAW_DIR)
        X_tr_raw, X_te, y_tr_raw, y_te = train_test_split(
            X_all, y_all, test_size=0.20,
            stratify=y_all, random_state=RANDOM_SEED
        )
    else:
        raise FileNotFoundError(
            "Dataset not found.\n"
            f"Expected: {TRAIN_DIR}/Open  and  {TRAIN_DIR}/Closed\n"
            "Download from: https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset"
        )

    # Balance
    X_tr_raw, y_tr_raw = balance_classes(X_tr_raw, y_tr_raw)

    # Augment training set
    print("[DATA] Augmenting training images...")
    X_tr_aug, y_tr_aug = augment_images(X_tr_raw, y_tr_raw, factor=3)

    # Validation split from augmented training set
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_tr_aug, y_tr_aug,
        test_size=VALIDATION_SPLIT,
        stratify=y_tr_aug,
        random_state=RANDOM_SEED
    )

    print(f"\n[DATA] Final splits:")
    print(f"  Train : {X_tr.shape}  classes: {np.bincount(y_tr)}")
    print(f"  Val   : {X_v.shape}   classes: {np.bincount(y_v)}")
    print(f"  Test  : {X_te.shape}  classes: {np.bincount(y_te)}")

    np.savez_compressed(cache,
                        X_tr=X_tr, X_v=X_v, X_te=X_te,
                        y_tr=y_tr, y_v=y_v, y_te=y_te)
    print(f"[DATA] Cached to {cache}")
    return X_tr, X_v, X_te, y_tr, y_v, y_te


def make_synthetic_dataset(n=2000):
    """
    Generate synthetic eye-like images for offline testing
    when Kaggle dataset is not yet downloaded.
    """
    np.random.seed(RANDOM_SEED)
    imgs, lbls = [], []
    for lbl in [0, 1]:
        for _ in range(n // 2):
            base = np.random.rand(IMG_HEIGHT, IMG_WIDTH).astype(np.float32)
            if lbl == 0:               # Open eye: brighter centre
                cx, cy = IMG_WIDTH//2, IMG_HEIGHT//2
                Y, X = np.ogrid[:IMG_HEIGHT, :IMG_WIDTH]
                mask = ((X-cx)**2 + (Y-cy)**2) < (IMG_WIDTH//4)**2
                base[mask] += 0.4
            else:                      # Closed eye: darker, flatter
                base *= 0.4
            imgs.append(np.clip(base, 0, 1))
            lbls.append(lbl)

    X = np.expand_dims(np.array(imgs, dtype=np.float32), -1)
    y = np.array(lbls, dtype=np.int32)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_tr_raw, X_te, y_tr_raw, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_tr_raw, y_tr_raw, test_size=VALIDATION_SPLIT,
        stratify=y_tr_raw, random_state=RANDOM_SEED
    )
    print(f"[DATA] Synthetic — Train:{X_tr.shape} Val:{X_v.shape} Test:{X_te.shape}")
    return X_tr, X_v, X_te, y_tr, y_v, y_te


if __name__ == "__main__":
    try:
        splits = load_kaggle_dataset()
    except FileNotFoundError as e:
        print(e)
        print("\n[INFO] Running with synthetic data for demo...")
        splits = make_synthetic_dataset()
    print("Preprocessing OK.")
