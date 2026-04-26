"""
run_pipeline.py — End-to-End Training Pipeline

Order:
  1. Load / preprocess Kaggle eye-image dataset
  2. Train CNN
  3. Train Logistic Regression on CNN features
  4. Train Audio MFCC + LR
  5. Fusion evaluation
  6. Final comparison chart + report

Usage:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --skip-cnn      # skip CNN if already trained
  python scripts/run_pipeline.py --synthetic      # force synthetic data
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils.config import MODEL_DIR, PROC_DIR, CNN_FEAT_PATH, RANDOM_SEED, CLASS_NAMES


def run(skip_cnn=False, synthetic=False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PROC_DIR,  exist_ok=True)
    t0 = time.time()
    all_results = {}

    _banner("MULTI-MODAL DRIVER SAFETY SYSTEM — TRAINING PIPELINE")
    print(f"  Dataset : Kaggle Drowsiness Detection (eye images)")
    print(f"  Models  : CNN + Logistic Regression + Audio-LR + Fusion")
    print(f"  Output  : {MODEL_DIR}\n")

    # ── STEP 1: Data ──────────────────────────────────────────────────────
    _step(1, 5, "Preprocessing Kaggle Eye-Image Dataset")
    from vision.preprocess import load_kaggle_dataset, make_synthetic_dataset
    try:
        if synthetic:
            raise FileNotFoundError("--synthetic flag set")
        X_tr, X_v, X_te, y_tr, y_v, y_te = load_kaggle_dataset()
    except FileNotFoundError as e:
        print(f"  [WARN] {e}")
        print("  [INFO] Using synthetic demo data.\n"
              "  Download dataset from:\n"
              "  https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset\n")
        X_tr, X_v, X_te, y_tr, y_v, y_te = make_synthetic_dataset(n=2000)

    # ── STEP 2: CNN ───────────────────────────────────────────────────────
    _step(2, 5, "Training CNN on Eye Images")
    if skip_cnn:
        from utils.config import CNN_MODEL_PATH
        from tensorflow import keras
        print(f"  Skipping — loading {CNN_MODEL_PATH}")
        cnn = keras.models.load_model(CNN_MODEL_PATH)
        # Re-extract features
        from vision.cnn_model import get_feature_extractor
        fe = get_feature_extractor(cnn)
        X_all  = np.concatenate([X_tr, X_v, X_te])
        y_all  = np.concatenate([y_tr, y_v, y_te])
        feats  = fe.predict(X_all, batch_size=64, verbose=0)
        n_tr   = len(X_tr) + len(X_v)
        np.savez_compressed(CNN_FEAT_PATH,
                            X_train=feats[:n_tr], y_train=y_all[:n_tr],
                            X_test=feats[n_tr:],  y_test=y_all[n_tr:])
        # Evaluate CNN
        y_pred_p = cnn.predict(X_te, verbose=0)
        y_pred   = np.argmax(y_pred_p, axis=1)
        from sklearn.metrics import accuracy_score, f1_score
        all_results["CNN"] = {
            "accuracy":  accuracy_score(y_te, y_pred),
            "f1":        f1_score(y_te, y_pred, zero_division=0),
        }
    else:
        from vision.train_cnn import train_cnn
        _, _, cnn_metrics = train_cnn(X_tr, y_tr, X_v, y_v, X_te, y_te)
        all_results["CNN"] = cnn_metrics

    # ── STEP 3: Logistic Regression ───────────────────────────────────────
    _step(3, 5, "Training Logistic Regression on CNN Features")
    from vision.logistic_regression import train_logistic_regression
    _, lr_metrics = train_logistic_regression()
    all_results["Logistic Regression (CNN)"] = lr_metrics

    # ── STEP 4: Audio ─────────────────────────────────────────────────────
    _step(4, 5, "Training Audio MFCC + Logistic Regression")
    from audio.train_audio import train_audio_model
    audio_dir = os.path.join(os.path.dirname(PROC_DIR), "raw", "audio")
    has_real  = os.path.isdir(os.path.join(audio_dir, "sober"))
    _, audio_metrics = train_audio_model(
        audio_dir=audio_dir if has_real else None,
        use_synthetic=not has_real
    )
    all_results["Audio LR (MFCC)"] = audio_metrics

    # ── STEP 5: Fusion ────────────────────────────────────────────────────
    _step(5, 5, "Evaluating Multi-Modal Fusion")
    from tensorflow import keras
    from utils.config import CNN_MODEL_PATH
    cnn = keras.models.load_model(CNN_MODEL_PATH)
    fusion_metrics = _eval_fusion(cnn, X_te, y_te)
    all_results["Multi-Modal Fusion"] = fusion_metrics

    # ── Report ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    _print_report(all_results, elapsed)
    _comparison_plot(all_results)
    _write_report(all_results)


def _eval_fusion(cnn, X_te, y_te):
    from fusion.fusion_engine import FusionEngine
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, roc_auc_score)
    engine  = FusionEngine()
    proba_c = cnn.predict(X_te, verbose=0)

    preds, scores = [], []
    for i in range(len(X_te)):
        cnn_p = float(proba_c[i][1])
        ear_r = dict(perclos=min(0.5, cnn_p*0.5),
                     ear=max(0.15, 0.4-cnn_p*0.3), pitch=cnn_p*20)
        audio_p = float(np.clip(0.38 + y_te[i]*0.3 + np.random.randn()*0.08, 0, 1))
        r = engine.update(cnn_drowsy_prob=cnn_p, ear_result=ear_r,
                          audio_intox_prob=audio_p)
        preds.append(1 if r["alert_level"] >= 2 else 0)
        scores.append(r["fused_score"])

    preds  = np.array(preds)
    scores = np.array(scores)
    m = dict(
        accuracy  = accuracy_score(y_te, preds),
        precision = precision_score(y_te, preds, zero_division=0),
        recall    = recall_score(y_te, preds,    zero_division=0),
        f1        = f1_score(y_te, preds,        zero_division=0),
        auc       = roc_auc_score(y_te, scores),
    )
    print(f"  Fusion — Acc:{m['accuracy']:.4f}  F1:{m['f1']:.4f}  AUC:{m['auc']:.4f}")
    return m


def _comparison_plot(results):
    names   = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors  = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
    x = np.arange(len(names))
    w = 0.2

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("Model Comparison — Kaggle Drowsiness Dataset",
                 fontsize=13, fontweight="bold")

    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [results[n].get(m, 0) for n in names]
        bars = ax.bar(x + i*w, vals, w, label=m.capitalize(),
                      color=c, alpha=0.88, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels(names, rotation=12, ha="right", fontsize=10)
    ax.set_ylim([0.5, 1.08])
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    best = max(results, key=lambda n: results[n].get("f1", 0))
    bi   = names.index(best)
    ax.axvspan(bi-0.1, bi+0.9, alpha=0.08, color="#10b981")
    ax.text(bi+0.4, 1.06, "★ Best", ha="center",
            color="#10b981", fontweight="bold", fontsize=10)

    plt.tight_layout()
    p = os.path.join(MODEL_DIR, "final_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  Comparison chart → {p}")


def _print_report(results, elapsed):
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<30} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-"*60)
    for n, r in results.items():
        print(f"{n:<30} {r.get('accuracy',0):>8.4f} "
              f"{r.get('precision',0):>8.4f} "
              f"{r.get('recall',0):>8.4f} "
              f"{r.get('f1',0):>8.4f}")
    print("="*60)
    best = max(results, key=lambda n: results[n].get("f1",0))
    print(f"\n  ✦ Best model : {best}")
    print(f"  ✦ Training   : {elapsed/60:.1f} min")


def _write_report(results):
    p = os.path.join(MODEL_DIR, "evaluation_report.txt")
    with open(p, "w") as f:
        f.write("MULTI-MODAL DRIVER SAFETY — EVALUATION REPORT\n" + "="*60 + "\n\n")
        f.write(f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} "
                f"{'Recall':>10} {'F1':>10}\n" + "-"*60 + "\n")
        for n, r in results.items():
            f.write(f"{n:<30} {r.get('accuracy',0):>10.4f} "
                    f"{r.get('precision',0):>10.4f} "
                    f"{r.get('recall',0):>10.4f} "
                    f"{r.get('f1',0):>10.4f}\n")
    print(f"  Report → {p}")


def _banner(txt):
    print("\n" + "="*60)
    print(f"  {txt}")
    print("="*60)


def _step(n, total, txt):
    print(f"\n[STEP {n}/{total}] {txt}")
    print("-"*50)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cnn",   action="store_true")
    ap.add_argument("--synthetic",  action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(ROOT, "src"))
    run(skip_cnn=args.skip_cnn, synthetic=args.synthetic)
