"""
train_cnn.py — CNN Training Pipeline on Kaggle Eye-Image Dataset

Steps
-----
1. Load preprocessed data (or auto-preprocess from raw images)
2. Build & compile CNN
3. Train with augmentation + callbacks
4. Evaluate on test set (accuracy, precision, recall, F1, ROC-AUC)
5. Save model + extract 256-d feature vectors for Logistic Regression
6. Plot training curves and confusion matrix
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    EPOCHS, BATCH_SIZE, MODEL_DIR, CNN_MODEL_PATH,
    CNN_FEAT_PATH, PROC_DIR, RANDOM_SEED, CLASS_NAMES
)
from vision.cnn_model import (
    build_cnn, compile_model, get_callbacks, get_feature_extractor
)
from vision.preprocess import load_kaggle_dataset, make_synthetic_dataset


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _class_weights(y):
    cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
    return dict(enumerate(cw))


def _save_training_curves(history, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("CNN Training — Kaggle Drowsiness Dataset",
                 fontsize=13, fontweight="bold")

    for ax, metric, val_metric, title, ylabel in [
        (axes[0], "accuracy", "val_accuracy", "Accuracy", "Accuracy"),
        (axes[1], "loss",     "val_loss",     "Loss",     "Loss"),
    ]:
        ax.plot(history.history[metric],     lw=2, label="Train", color="#3b82f6")
        ax.plot(history.history[val_metric], lw=2, label="Val",   color="#10b981")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "cnn_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Training curves → {path}")


def _save_confusion_matrix(y_true, y_pred, labels, out_dir, prefix="cnn"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax, annot_kws={"size": 14})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{prefix.upper()} Confusion Matrix — Kaggle", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Confusion matrix → {path}")


def evaluate_model(y_true, y_pred, y_proba, name="CNN"):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred,    zero_division=0)
    f1   = f1_score(y_true, y_pred,        zero_division=0)
    auc  = roc_auc_score(y_true, y_proba) if y_proba is not None else None

    print(f"\n{'─'*50}")
    print(f"  {name} Test Results")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    if auc:
        print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES)}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


# ─── Main Training Function ───────────────────────────────────────────────────

def train_cnn(X_train=None, y_train=None, X_val=None, y_val=None,
              X_test=None,  y_test=None):
    """
    Full CNN training pipeline.
    If no data arrays provided, loads from Kaggle dataset (or synthetic fallback).

    Returns
    -------
    model   : trained Keras model
    history : Keras History object
    metrics : dict {accuracy, precision, recall, f1, auc}
    """
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PROC_DIR,  exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────
    if X_train is None:
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = load_kaggle_dataset()
        except FileNotFoundError as e:
            print(f"\n[WARN] {e}")
            print("[INFO] Falling back to synthetic data for demonstration.\n")
            X_train, X_val, X_test, y_train, y_val, y_test = make_synthetic_dataset()

    print(f"\n[CNN] Train:{X_train.shape}  Val:{X_val.shape}  Test:{X_test.shape}")

    # ── 2. Build ──────────────────────────────────────────────────────────
    model = build_cnn()
    model = compile_model(model)
    model.summary()

    # ── 3. Train ──────────────────────────────────────────────────────────
    cw = _class_weights(y_train)
    print(f"\n[CNN] Class weights: {cw}")

    history = model.fit(
        X_train, y_train,
        batch_size       = BATCH_SIZE,
        epochs           = EPOCHS,
        validation_data  = (X_val, y_val),
        class_weight     = cw,
        callbacks        = get_callbacks(CNN_MODEL_PATH),
        verbose          = 1
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────────
    proba  = model.predict(X_test, verbose=0)            # (N, 2)
    y_pred = np.argmax(proba, axis=1)
    metrics = evaluate_model(y_test, y_pred, proba[:, 1], "CNN")

    _save_training_curves(history, MODEL_DIR)
    _save_confusion_matrix(y_test, y_pred, CLASS_NAMES, MODEL_DIR, "cnn")

    # ── 5. Extract feature vectors for Logistic Regression ───────────────
    print("\n[CNN] Extracting feature vectors for LR classifier...")
    feat_extractor = get_feature_extractor(model)

    X_all  = np.concatenate([X_train, X_val, X_test])
    y_all  = np.concatenate([y_train, y_val, y_test])
    feats  = feat_extractor.predict(X_all, batch_size=64, verbose=1)

    split_tr = len(X_train) + len(X_val)
    np.savez_compressed(
        CNN_FEAT_PATH,
        X_train=feats[:split_tr], y_train=y_all[:split_tr],
        X_test=feats[split_tr:],  y_test=y_all[split_tr:]
    )
    print(f"[CNN] Feature vectors saved → {CNN_FEAT_PATH}")

    return model, history, metrics


if __name__ == "__main__":
    train_cnn()
