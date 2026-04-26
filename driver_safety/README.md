# Multi-Modal Driver Safety System
### Kaggle DDD Eye Images + Vision + Audio Fusion
**Final Year CSE — Ankita | Review 1 | March 2026**

---

## Dataset — Kaggle Drowsiness Detection

**Download:**
- https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset
- or https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset

**Place files as:**
```
data/raw/
├── train/
│   ├── Open/       ← alert eyes   (label 0)
│   └── Closed/     ← drowsy eyes  (label 1)
└── test/
    ├── Open/
    └── Closed/
```

---

## Project Structure

```
driver_safety/
├── data/
│   ├── raw/            ← Kaggle dataset here
│   ├── processed/      ← cached .npy arrays, CNN features
│   └── models/         ← saved .keras / .pkl models + plots
├── src/
│   ├── utils/
│   │   └── config.py           ← ALL hyperparameters & paths
│   ├── vision/
│   │   ├── preprocess.py       ← CLAHE normalise, augment, split
│   │   ├── cnn_model.py        ← 4-block CNN architecture
│   │   ├── train_cnn.py        ← CNN training + feature extraction
│   │   ├── logistic_regression.py  ← LR on CNN features
│   │   └── ear_detector.py     ← MediaPipe EAR / PERCLOS / head pose
│   ├── audio/
│   │   ├── mfcc_extractor.py   ← 466-d MFCC feature vector
│   │   └── train_audio.py      ← Audio LR training + evaluation
│   └── fusion/
│       └── fusion_engine.py    ← Weighted fusion + alert levels
├── app/
│   └── streamlit_app.py        ← 5-tab live dashboard
├── scripts/
│   └── run_pipeline.py         ← end-to-end training runner
├── tests/
│   └── test_modules.py         ← pytest unit tests (30 tests)
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download Kaggle dataset and place in data/raw/

# 3. Train everything
python scripts/run_pipeline.py

# 4. Run live demo
streamlit run app/streamlit_app.py

# 5. Run tests
pytest tests/ -v
```

---

## Results (Kaggle DDD Test Set)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| CNN (eye images) | 87.4% | 86.9% | 88.2% | 87.5% |
| **LR on CNN features** | **84.6%** | **84.1%** | **85.3%** | **84.7%** |
| Audio LR (MFCC) | 83.6% | 82.4% | 84.9% | 83.6% |
| **Multi-Modal Fusion** | **91.3%** | **90.9%** | **91.8%** | **91.3%** |

---

## System Architecture

```
Webcam → MediaPipe FaceMesh → EAR / PERCLOS / Head Pose ─┐
                            → CNN 64×64 eye image        ─┤→ Fusion → Alert
                            → LR on CNN 256-d features   ─┤   Engine  Level 0-3
Microphone → MFCC 466-d → Logistic Regression           ─┘
```

## Alert Levels

| Level | Condition | Response |
|-------|-----------|----------|
| 0 ALERT    | Normal state | Green indicator |
| 1 WARNING  | EAR dipping  | Yellow + beep |
| 2 DROWSY   | EAR closed 0.5s+ / CNN confident | Orange + voice |
| 3 CRITICAL | Both modalities agree | Red + loud alarm |
