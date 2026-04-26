"""
mfcc_extractor.py — MFCC Audio Feature Extraction for Intoxication Detection

Feature vector per clip (~3 s audio):
  MFCC mean+std        (40×2 = 80)
  Delta-MFCC mean+std  (40×2 = 80)
  Chroma mean+std      (12×2 = 24)
  Spectral contrast    ( 7×2 = 14)
  ZCR mean+std         (2)
  RMS mean+std         (2)
  Spectral rolloff     (2)
  Spectral centroid    (2)
  Mel-spectrogram      (128×2 = 256)
  Onset / tempo        (4)
  ──────────────────
  Total               ≈ 466 features
"""

import os
import sys
import numpy as np
from pathlib import Path
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    SAMPLE_RATE, AUDIO_DURATION, N_MFCC,
    N_FFT, HOP_LENGTH, N_MELS, RANDOM_SEED
)

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# ─── Core extraction ─────────────────────────────────────────────────────────

def load_audio(path_or_array, sr=SAMPLE_RATE, dur=AUDIO_DURATION):
    """Load from file path or numpy array. Pad / trim to `dur` seconds."""
    if isinstance(path_or_array, (str, Path)):
        y, _ = librosa.load(str(path_or_array), sr=sr, duration=dur)
    else:
        y = np.asarray(path_or_array, dtype=np.float32)

    target = int(sr * dur)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    return y.astype(np.float32), sr


def extract_features(y, sr=SAMPLE_RATE):
    """Return 1-D feature vector from raw audio samples."""
    feats = []

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    feats += list(np.mean(mfcc, 1)) + list(np.std(mfcc, 1))

    # Delta MFCC
    dmfcc = librosa.feature.delta(mfcc)
    feats += list(np.mean(dmfcc, 1)) + list(np.std(dmfcc, 1))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    feats += list(np.mean(chroma, 1)) + list(np.std(chroma, 1))

    # Spectral contrast
    sc = librosa.feature.spectral_contrast(y=y, sr=sr,
                                            n_fft=N_FFT, hop_length=HOP_LENGTH)
    feats += list(np.mean(sc, 1)) + list(np.std(sc, 1))

    # ZCR, RMS, rolloff, centroid
    for fn in [librosa.feature.zero_crossing_rate,
               librosa.feature.rms]:
        v = fn(y=y, hop_length=HOP_LENGTH)
        feats += [float(np.mean(v)), float(np.std(v))]

    for fn in [librosa.feature.spectral_rolloff,
               librosa.feature.spectral_centroid]:
        v = fn(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        feats += [float(np.mean(v)), float(np.std(v))]

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feats += list(np.mean(mel_db, 1)) + list(np.std(mel_db, 1))

    # Onset + tempo
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    feats += [float(np.mean(onset)), float(np.std(onset)), float(np.max(onset))]
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
        feats.append(float(tempo))
    except Exception:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32)


def extract_from_file(path):
    y, sr = load_audio(path)
    return extract_features(y, sr)


def extract_from_array(arr, sr=SAMPLE_RATE):
    y, sr = load_audio(arr, sr)
    return extract_features(y, sr)


# ─── Dataset builder ─────────────────────────────────────────────────────────

def build_audio_dataset(root_dir):
    """
    Scan root_dir/sober/ and root_dir/intoxicated/ for audio files.
    Returns X (N, n_feat), y (N,), paths.
    """
    class_map = {"sober": 0, "intoxicated": 1}
    pairs = []
    for cls, label in class_map.items():
        d = os.path.join(root_dir, cls)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if Path(fn).suffix.lower() in _AUDIO_EXTS:
                pairs.append((os.path.join(d, fn), label))

    if not pairs:
        raise FileNotFoundError(f"No audio files in {root_dir}/sober or .../intoxicated")

    X_list, y_list, paths = [], [], []
    errs = 0
    for fp, lbl in pairs:
        try:
            X_list.append(extract_from_file(fp))
            y_list.append(lbl)
            paths.append(fp)
        except Exception as e:
            errs += 1
            if errs <= 3:
                print(f"  [WARN] {fp}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.int32)
    print(f"[AUDIO] {len(X)} samples, {X.shape[1]} features. Errors: {errs}")
    return X, y, paths


if __name__ == "__main__":
    sr = SAMPLE_RATE
    t  = np.linspace(0, AUDIO_DURATION, int(sr * AUDIO_DURATION))
    sig = (0.5*np.sin(2*np.pi*200*t) + 0.1*np.random.randn(len(t))).astype(np.float32)
    f   = extract_features(sig, sr)
    print(f"Feature vector shape: {f.shape}")
    print("MFCC extractor OK.")
