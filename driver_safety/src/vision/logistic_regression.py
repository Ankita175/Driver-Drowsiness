"""
logistic_regression.py — Logistic Regression on CNN Feature Vectors

Pipeline
--------
CNN (trained on 64×64 eye images)
    → 256-d Global Average Pooling feature vector
        → StandardScaler normalisation
            → PCA (top-50 components, optional)
                → Logistic Regression (L2, C=10)
                    → Softmax probability → prediction

Why LR on CNN features?
  - Interpretable: the learned weight vector shows which CNN features matter
  - Fast at inference: single matrix multiply after CNN forward pass
  - Works well on linearly-separable CNN features
  - Comparable to SVM but with calibrated probabilities out-of-the-box
  - Easy to retrain the head without retraining the whole CNN

Evaluation
----------
  - 5-fold stratified cross-validation
  - ROC-AUC, precision-recall curve
  - Learning curve (training size vs accuracy)
  - Calibration curve (reliability diagram)
"""

import os
import sys
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model      import LogisticRegression
from sklearn.preprocessing     import StandardScaler
from sklearn.decomposition     import PCA
from sklearn.pipeline          import Pipeline
from sklearn.model_selection   import (StratifiedKFold, cross_val_score,
                                        learning_curve)
from sklearn.metrics           import (accuracy_score, precision_score,
                                        recall_score, f1_score, roc_auc_score,
                                        roc_curve, precision_recall_curve,
                                        average_precision_score,
                                        classification_report, confusion_matrix)
from sklearn.calibration        import CalibratedClassifierCV, calibration_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    LR_MODEL_PATH, SCALER_PATH, CNN_FEAT_PATH,
    MODEL_DIR, CLASS_NAMES, RANDOM_SEED
)


# ─── Pipeline Builder ─────────────────────────────────────────────────────────

def build_lr_pipeline(C=10.0, use_pca=True, n_components=50):
    """
    Return a sklearn Pipeline:
        StandardScaler → [PCA] → LogisticRegression

    Args:
        C            : inverse regularisation strength (higher = less reg)
        use_pca      : reduce 256-d features to n_components before LR
        n_components : PCA components (must be ≤ feature dimensionality)
    """
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_components,
                                  random_state=RANDOM_SEED)))
    steps.append((
        "lr",
        LogisticRegression(
            C              = C,
            penalty        = "l2",
            solver         = "lbfgs",
            max_iter       = 2000,
            random_state   = RANDOM_SEED,
            class_weight   = "balanced",
        )
    ))
    return Pipeline(steps)


# ─── Main Training ────────────────────────────────────────────────────────────

def train_logistic_regression(X_train=None, y_train=None,
                               X_test=None,  y_test=None):
    """
    Train and evaluate Logistic Regression on CNN feature vectors.

    If no arrays provided, loads from CNN_FEAT_PATH.

    Returns
    -------
    pipeline : fitted sklearn Pipeline
    metrics  : dict {accuracy, precision, recall, f1, auc}
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    # ── 1. Load features ─────────────────────────────────────────────────
    if X_train is None:
        if not os.path.exists(CNN_FEAT_PATH):
            raise FileNotFoundError(
                f"CNN features not found at {CNN_FEAT_PATH}.\n"
                f"Run train_cnn.py first."
            )
        d = np.load(CNN_FEAT_PATH)
        X_train, y_train = d["X_train"], d["y_train"]
        X_test,  y_test  = d["X_test"],  d["y_test"]

    print(f"\n[LR] Feature shape — Train:{X_train.shape}  Test:{X_test.shape}")

    # Clip n_components to available features
    n_feat = X_train.shape[1]
    n_comp = min(50, n_feat - 1)

    # ── 2. Build & Cross-validate ─────────────────────────────────────────
    pipeline = build_lr_pipeline(C=10.0, use_pca=(n_feat > n_comp),
                                  n_components=n_comp)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_acc = cross_val_score(pipeline, X_train, y_train,
                              cv=cv, scoring="accuracy", n_jobs=-1)
    cv_f1  = cross_val_score(pipeline, X_train, y_train,
                              cv=cv, scoring="f1",       n_jobs=-1)

    print(f"[LR] 5-Fold CV Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"[LR] 5-Fold CV F1       : {cv_f1.mean():.4f}  ± {cv_f1.std():.4f}")

    # ── 3. Fit on full training set ───────────────────────────────────────
    pipeline.fit(X_train, y_train)

    # ── 4. Evaluate ───────────────────────────────────────────────────────
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = _print_metrics(y_test, y_pred, y_proba, "LR on CNN features")

    # ── 5. Save ───────────────────────────────────────────────────────────
    joblib.dump(pipeline, LR_MODEL_PATH)
    print(f"[LR] Saved → {LR_MODEL_PATH}")

    # ── 6. Plots ──────────────────────────────────────────────────────────
    _plot_confusion_matrix(y_test, y_pred)
    _plot_roc_pr(y_test, y_proba)
    _plot_learning_curve(pipeline, X_train, y_train, cv)
    _plot_calibration(pipeline, X_test, y_test, y_proba)
    _plot_coefficients(pipeline)

    return pipeline, {**metrics, "cv_accuracy": cv_acc.mean(), "cv_f1": cv_f1.mean()}


# ─── Plots ────────────────────────────────────────────────────────────────────

def _print_metrics(y_true, y_pred, y_proba, name):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred,    zero_division=0)
    f1   = f1_score(y_true, y_pred,        zero_division=0)
    auc  = roc_auc_score(y_true, y_proba)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES)}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def _plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax, annot_kws={"size": 14})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("LR Classifier — Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    p = os.path.join(MODEL_DIR, "lr_confusion_matrix.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Confusion matrix → {p}")


def _plot_roc_pr(y_true, y_proba):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Logistic Regression — ROC & Precision-Recall",
                 fontweight="bold", fontsize=12)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    axes[0].plot(fpr, tpr, lw=2.5, color="#3b82f6", label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC Curve",
                xlim=[-0.01, 1.01], ylim=[-0.01, 1.05])
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Precision-Recall
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    axes[1].plot(rec_c, prec_c, lw=2.5, color="#10b981", label=f"AP = {ap:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="Precision-Recall Curve",
                xlim=[-0.01, 1.01], ylim=[-0.01, 1.05])
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(MODEL_DIR, "lr_roc_pr.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ROC & PR curves → {p}")


def _plot_learning_curve(pipeline, X, y, cv):
    """Show how accuracy improves with more training data."""
    train_sizes, tr_sc, val_sc = learning_curve(
        pipeline, X, y,
        cv=cv, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(train_sizes, tr_sc.mean(1)-tr_sc.std(1),
                    tr_sc.mean(1)+tr_sc.std(1), alpha=0.15, color="#3b82f6")
    ax.fill_between(train_sizes, val_sc.mean(1)-val_sc.std(1),
                    val_sc.mean(1)+val_sc.std(1), alpha=0.15, color="#10b981")
    ax.plot(train_sizes, tr_sc.mean(1),  lw=2, color="#3b82f6", label="Train")
    ax.plot(train_sizes, val_sc.mean(1), lw=2, color="#10b981", label="CV Val")
    ax.set(xlabel="Training samples", ylabel="Accuracy",
           title="LR Learning Curve", ylim=[0.5, 1.05])
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(MODEL_DIR, "lr_learning_curve.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Learning curve → {p}")


def _plot_calibration(pipeline, X_test, y_test, y_proba):
    """Reliability diagram: does prob=0.7 really mean 70% chance?"""
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, "s-", lw=2, color="#3b82f6",
            label="LR (CNN features)")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Perfect")
    ax.set(xlabel="Mean predicted probability",
           ylabel="Fraction of positives",
           title="Calibration Curve (Reliability Diagram)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(MODEL_DIR, "lr_calibration.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Calibration curve → {p}")


def _plot_coefficients(pipeline):
    """Bar chart of top-20 most influential LR weight magnitudes."""
    lr = pipeline.named_steps["lr"]
    coef = np.abs(lr.coef_[0])
    top_idx = np.argsort(coef)[-20:]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(20), coef[top_idx], color="#8b5cf6", alpha=0.85,
            edgecolor="white")
    ax.set_yticks(range(20))
    ax.set_yticklabels([f"feat_{i}" for i in top_idx], fontsize=9)
    ax.set_xlabel("|Coefficient|")
    ax.set_title("Top-20 LR Feature Weights (CNN feature importance)")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    p = os.path.join(MODEL_DIR, "lr_feature_weights.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Feature weights → {p}")


# ─── Inference helper ─────────────────────────────────────────────────────────

def predict_with_lr(feature_vector, model_path=LR_MODEL_PATH):
    """
    Classify a single 256-d CNN feature vector.

    Args:
        feature_vector : np.ndarray shape (256,) or (1, 256)
    Returns:
        dict {label, class_name, probability}
    """
    pipeline = joblib.load(model_path)
    fv = np.array(feature_vector).reshape(1, -1)
    label  = int(pipeline.predict(fv)[0])
    proba  = float(pipeline.predict_proba(fv)[0][label])
    return {"label": label, "class_name": CLASS_NAMES[label], "probability": proba}


if __name__ == "__main__":
    train_logistic_regression()
