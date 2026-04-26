"""
cnn_model.py — Lightweight CNN for Kaggle Eye-Image Drowsiness Detection

Input  : 64 × 64 × 1 (grayscale eye crop)
Output : 2-class softmax (Open=0, Closed=1)

Architecture (custom VGG-style):
  Block 1 : Conv32  → BN → ReLU → Conv32  → BN → ReLU → MaxPool → Drop(0.25)
  Block 2 : Conv64  → BN → ReLU → Conv64  → BN → ReLU → MaxPool → Drop(0.25)
  Block 3 : Conv128 → BN → ReLU → Conv128 → BN → ReLU → MaxPool → Drop(0.35)
  Block 4 : Conv256 → BN → ReLU → MaxPool → Drop(0.35)
  Head    : GlobalAvgPool → Dense(256, relu) → Drop(0.5) → Dense(2, softmax)

Design decisions:
  - GlobalAvgPool instead of Flatten → fewer params, less overfitting
  - BatchNorm  → robust across lighting conditions in eye images
  - L2 regularisation in conv layers
  - Small input (64×64) → ~2 ms/frame CPU inference
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, NUM_CLASSES,
    DROPOUT_RATE, L2_REG, LEARNING_RATE, EPOCHS,
    BATCH_SIZE, VALIDATION_SPLIT, CNN_MODEL_PATH, MODEL_DIR, RANDOM_SEED
)


# ─── Architecture ────────────────────────────────────────────────────────────

def build_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
              num_classes=NUM_CLASSES,
              dropout=DROPOUT_RATE,
              l2=L2_REG,
              return_features=False):
    """
    Build drowsiness detection CNN.

    Args:
        return_features : if True return feature-extractor model
                          (output = 256-d Dense, used by LR classifier)
    """
    reg    = regularizers.l2(l2)
    inputs = keras.Input(shape=input_shape, name="eye_image")

    # ── Block 1 ──────────────────────────────────────────────────────────
    x = layers.Conv2D(32, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="b1_c1")(inputs)
    x = layers.BatchNormalization(name="b1_bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="b1_c2")(x)
    x = layers.BatchNormalization(name="b1_bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="b1_pool")(x)
    x = layers.Dropout(dropout * 0.5, name="b1_drop")(x)

    # ── Block 2 ──────────────────────────────────────────────────────────
    x = layers.Conv2D(64, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="b2_c1")(x)
    x = layers.BatchNormalization(name="b2_bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="b2_c2")(x)
    x = layers.BatchNormalization(name="b2_bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="b2_pool")(x)
    x = layers.Dropout(dropout * 0.5, name="b2_drop")(x)

    # ── Block 3 ──────────────────────────────────────────────────────────
    x = layers.Conv2D(128, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="b3_c1")(x)
    x = layers.BatchNormalization(name="b3_bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="b3_c2")(x)
    x = layers.BatchNormalization(name="b3_bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="b3_pool")(x)
    x = layers.Dropout(dropout * 0.7, name="b3_drop")(x)

    # ── Block 4 ──────────────────────────────────────────────────────────
    x = layers.Conv2D(256, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="b4_c1")(x)
    x = layers.BatchNormalization(name="b4_bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="b4_pool")(x)
    x = layers.Dropout(dropout * 0.7, name="b4_drop")(x)

    # ── Feature Head ─────────────────────────────────────────────────────
    gap  = layers.GlobalAveragePooling2D(name="gap")(x)
    feat = layers.Dense(256, activation="relu",
                        kernel_regularizer=reg, name="features")(gap)
    feat = layers.Dropout(dropout, name="feat_drop")(feat)

    if return_features:
        return keras.Model(inputs, feat, name="CNN_FeatureExtractor")

    # ── Classifier ───────────────────────────────────────────────────────
    out = layers.Dense(num_classes, activation="softmax",
                       name="classifier")(feat)
    return keras.Model(inputs, out, name="DrowsinessCNN")


def compile_model(model, lr=LEARNING_RATE):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy"
        ]
    )
    return model


def get_callbacks(save_path=CNN_MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            save_path, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4,
            min_lr=1e-7, verbose=1
        ),
    ]


def get_feature_extractor(cnn_model):
    """Return sub-model up to the 'features' Dense layer (256-d output)."""
    return keras.Model(
        inputs  = cnn_model.input,
        outputs = cnn_model.get_layer("features").output,
        name    = "FeatureExtractor"
    )


if __name__ == "__main__":
    tf.random.set_seed(RANDOM_SEED)
    m = build_cnn()
    m.summary()
    feat_m = build_cnn(return_features=True)
    print(f"\nFeature extractor output: {feat_m.output_shape}")
    print(f"Total params : {m.count_params():,}")
