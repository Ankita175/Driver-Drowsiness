"""
config.py — Central configuration for the Multi-Modal Driver Safety System
All modules import from here. Change values here to tune the entire system.

Dataset: Kaggle Drowsiness Detection Dataset
  - Eye images: open / closed
  - Download: https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset
    or: https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset
  Structure expected:
    data/raw/
    ├── train/
    │   ├── Open/       ← alert (label 0)
    │   └── Closed/     ← drowsy (label 1)
    └── test/
        ├── Open/
        └── Closed/
"""

import os

# ─── BASE PATHS ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(BASE_DIR, "data")
RAW_DIR    = os.path.join(DATA_DIR, "raw")
PROC_DIR   = os.path.join(DATA_DIR, "processed")
MODEL_DIR  = os.path.join(DATA_DIR, "models")

TRAIN_DIR  = os.path.join(RAW_DIR, "train")
TEST_DIR   = os.path.join(RAW_DIR, "test")

# Kaggle folder names → labels
CLASS_FOLDERS = {"Open": 0, "Closed": 1}   # 0=alert, 1=drowsy
CLASS_NAMES   = ["Open (Alert)", "Closed (Drowsy)"]

# ─── IMAGE CONFIG ─────────────────────────────────────────────────────────────
IMG_HEIGHT   = 64
IMG_WIDTH    = 64
IMG_CHANNELS = 1          # Grayscale (eye images)
NUM_CLASSES  = 2

# ─── CNN TRAINING ─────────────────────────────────────────────────────────────
BATCH_SIZE       = 32
EPOCHS           = 25
LEARNING_RATE    = 1e-4
DROPOUT_RATE     = 0.5
VALIDATION_SPLIT = 0.15
L2_REG           = 1e-4

CNN_MODEL_PATH   = os.path.join(MODEL_DIR, "cnn_drowsiness.keras")
LR_MODEL_PATH    = os.path.join(MODEL_DIR, "lr_classifier.pkl")
SCALER_PATH      = os.path.join(MODEL_DIR, "feature_scaler.pkl")
CNN_FEAT_PATH    = os.path.join(PROC_DIR,  "cnn_features.npz")

# ─── EAR CONFIG ──────────────────────────────────────────────────────────────
EAR_THRESHOLD     = 0.18   # Below → eye closed
EAR_CONSEC_FRAMES = 25     # Consecutive closed frames → drowsy alert
EAR_WARNING_FRAMES = 6     # Consecutive closed frames → warning

# MediaPipe FaceMesh landmark indices for each eye (6 points each)
LEFT_EYE_IDX  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# ─── AUDIO CONFIG ─────────────────────────────────────────────────────────────
SAMPLE_RATE    = 22050
AUDIO_DURATION = 3.0        # seconds per clip
N_MFCC         = 40
N_FFT          = 2048
HOP_LENGTH     = 512
N_MELS         = 128
AUDIO_CLASSES  = ["Sober", "Intoxicated"]
AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "audio_classifier.pkl")

# ─── FUSION CONFIG ────────────────────────────────────────────────────────────
VISION_WEIGHT = 0.65        # Must sum to 1.0 with AUDIO_WEIGHT
AUDIO_WEIGHT  = 0.35

FUSION_WARN_THRESH     = 0.30
FUSION_DROWSY_THRESH   = 0.52
FUSION_CRITICAL_THRESH = 0.72

# ─── ALERT LEVELS ─────────────────────────────────────────────────────────────
ALERT_LEVELS = {
    0: {"label": "ALERT",    "color": "#10b981", "hex": "10b981", "emoji": "✅"},
    1: {"label": "WARNING",  "color": "#f59e0b", "hex": "f59e0b", "emoji": "⚠️"},
    2: {"label": "DROWSY",   "color": "#f97316", "hex": "f97316", "emoji": "🟠"},
    3: {"label": "CRITICAL", "color": "#ef4444", "hex": "ef4444", "emoji": "🚨"},
}

# ─── APP CONFIG ───────────────────────────────────────────────────────────────
WEBCAM_SOURCE  = 0
TARGET_FPS     = 30
ALERT_COOLDOWN = 3.0        # seconds between repeat audio alerts
RANDOM_SEED    = 42
