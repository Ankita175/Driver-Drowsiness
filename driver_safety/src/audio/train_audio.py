"""
train_audio.py — Audio Intoxication Classifier (MFCC + Logistic Regression)

Uses the same LR approach as the vision module for consistency.
Falls back to synthetic speech features when no audio dataset is available.

To use a real audio dataset, organise it as:
  data/raw/audio/
  ├── sober/        ← WAV/MP3 files of sober speech
  └── intoxicated/  ← WAV/MP3 files of slurred / intoxicated speech
"""

import os
import sys
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.decomposition   import PCA
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score)
from sklearn.metrics         import (accuracy_score, precision_score,
                                      recall_score, f1_score, roc_auc_score,
                                      classification_report, confusion_matrix,
                                      roc_curve)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    AUDIO_MODEL_PATH, MODEL_DIR, PROC_DIR,
    AUDIO_CLASSES, RANDOM_SEED, SAMPLE_RATE, AUDIO_DURATION
)
from audio.mfcc_extractor import build_audio_dataset, extract_features


# ─── Synthetic data (when no real dataset is available) ──────────────────────

def make_synthetic_audio_features(n=500, seed=RANDOM_SEED):
    """
    Simulate sober vs intoxicated feature distributions.

    Sober     : higher ZCR, more consistent MFCC, faster tempo
    Intoxicated: lower ZCR, higher MFCC std, slower / irregular speech
    """
    np.random.seed(seed)
    n_f = 466                   # approximate feature dimension
    n2  = n // 2

    sober = np.random.randn(n2, n_f).astype(np.float32) * 12 + 45
    sober[:, :40]   += 18       # higher MFCC mean
    sober[:, 160]   += 0.06     # higher ZCR mean
    sober[:, 162]   += 0.03     # higher RMS

    intox = np.random.randn(n2, n_f).astype(np.float32) * 22 + 28
    intox[:, :40]   -= 12       # different MFCC mean
    intox[:, 40:80] += 18       # higher MFCC std (irregular)
    intox[:, 160]   -= 0.03     # lower ZCR

    X = np.vstack([sober, intox])
    y = np.array([0]*n2 + [1]*n2, dtype=np.int32)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def _build_pipeline(n_feat):
    n_comp = min(50, n_feat - 1)
    steps  = [("scaler", StandardScaler())]
    if n_feat > n_comp:
        steps.append(("pca", PCA(n_components=n_comp, random_state=RANDOM_SEED)))
    steps.append(("lr", LogisticRegression(
        C=10.0, penalty="l2", solver="lbfgs", max_iter=2000,
        class_weight="balanced", random_state=RANDOM_SEED
    )))
    return Pipeline(steps)


# ─── Main ─────────────────────────────────────────────────────────────────────

def train_audio_model(audio_dir=None, use_synthetic=False):
    """
    Train MFCC + Logistic Regression intoxication classifier.

    Returns: (pipeline, metrics_dict)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PROC_DIR,  exist_ok=True)
    np.random.seed(RANDOM_SEED)

    # ── Load data ─────────────────────────────────────────────────────────
    if use_synthetic or audio_dir is None:
        print("[AUDIO] Using synthetic speech features (no real dataset provided).")
        print("  → Supply real WAV files in data/raw/audio/sober & .../intoxicated")
        X, y = make_synthetic_audio_features(n=600)
    else:
        try:
            X, y, _ = build_audio_dataset(audio_dir)
        except FileNotFoundError as e:
            print(f"[AUDIO] {e} — falling back to synthetic data.")
            X, y = make_synthetic_audio_features(n=600)

    np.savez_compressed(os.path.join(PROC_DIR, "audio_features.npz"), X=X, y=y)

    # ── Split ─────────────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    print(f"[AUDIO] Train:{X_tr.shape}  Test:{X_te.shape}  "
          f"Balance: {np.bincount(y_te)}")

    # ── Build & cross-validate ────────────────────────────────────────────
    pipeline = _build_pipeline(X_tr.shape[1])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    cv_acc = cross_val_score(pipeline, X_tr, y_tr, cv=cv,
                              scoring="accuracy", n_jobs=-1)
    cv_f1  = cross_val_score(pipeline, X_tr, y_tr, cv=cv,
                              scoring="f1",       n_jobs=-1)
    print(f"[AUDIO] 5-CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"[AUDIO] 5-CV F1      : {cv_f1.mean():.4f}  ± {cv_f1.std():.4f}")

    # ── Fit & evaluate ────────────────────────────────────────────────────
    pipeline.fit(X_tr, y_tr)

    y_pred  = pipeline.predict(X_te)
    y_proba = pipeline.predict_proba(X_te)[:, 1]

    metrics = _eval(y_te, y_pred, y_proba)

    # ── Save ──────────────────────────────────────────────────────────────
    joblib.dump(pipeline, AUDIO_MODEL_PATH)
    print(f"[AUDIO] Saved → {AUDIO_MODEL_PATH}")

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_cm(y_te, y_pred)
    _plot_roc(y_te, y_proba)

    return pipeline, metrics


def _eval(y_true, y_pred, y_proba):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred,    zero_division=0)
    f1   = f1_score(y_true, y_pred,        zero_division=0)
    auc  = roc_auc_score(y_true, y_proba)
    print(f"\n{'─'*50}")
    print(f"  Audio LR (MFCC) — Test Results")
    print(f"{'─'*50}")
    print(f"  Accuracy : {acc:.4f}  Precision : {prec:.4f}")
    print(f"  Recall   : {rec:.4f}  F1        : {f1:.4f}  AUC: {auc:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=AUDIO_CLASSES)}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def _plot_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=AUDIO_CLASSES, yticklabels=AUDIO_CLASSES,
                linewidths=0.5, ax=ax, annot_kws={"size": 14})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Audio LR — Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    p = os.path.join(MODEL_DIR, "audio_confusion_matrix.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Confusion matrix → {p}")


def _plot_roc(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2.5, color="#10b981", label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1],"k--",lw=1.5,alpha=0.5)
    ax.set(xlabel="FPR", ylabel="TPR", title="Audio LR — ROC Curve",
           xlim=[-0.01,1.01], ylim=[-0.01,1.05])
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(MODEL_DIR, "audio_roc.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ROC curve → {p}")


# ─── Inference ────────────────────────────────────────────────────────────────

def predict_intoxication(audio_input, model_path=AUDIO_MODEL_PATH):
    """
    Predict intoxication from an audio file path or numpy array.

    Returns dict: {label, class_name, sober_prob, intox_prob}
    """
    pipeline = joblib.load(model_path)

    if isinstance(audio_input, (str, os.PathLike)):
        from audio.mfcc_extractor import extract_from_file
        feats = extract_from_file(audio_input)
    else:
        from audio.mfcc_extractor import extract_from_array
        feats = extract_from_array(audio_input)

    feats   = feats.reshape(1, -1)
    proba   = pipeline.predict_proba(feats)[0]
    label   = int(np.argmax(proba))
    return {
        "label":      label,
        "class_name": AUDIO_CLASSES[label],
        "sober_prob": float(proba[0]),
        "intox_prob": float(proba[1]),
    }


if __name__ == "__main__":
    train_audio_model(use_synthetic=True)
