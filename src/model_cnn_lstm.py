"""
Multimodal Deep Learning Module (Section 3.3).

CNN-LSTM with Multi-Head Attention for cryptocurrency volatility forecasting.
Architecture:
    Input → 1D-CNN → Bi-LSTM → Multi-Head Attention → Dense → Output

REPRODUCIBILITY NOTE
---------------------
The original module only called `tf.random.set_seed(SEED)` once, at import
time. That does not make repeated runs of the full pipeline reproducible:
- Only TensorFlow's RNG was seeded; Python's `random` and NumPy's global RNG
  were not, so anything in the pipeline that draws from them (including
  Keras' own array-shuffling path for model.fit) could differ run to run.
- GPU execution (cuDNN conv/LSTM kernels) is non-deterministic by default
  regardless of any seed unless op-level determinism is explicitly enabled.
- Because build_model()/train_model() are called ~100-250 times over a full
  run (main model, residual hybrid, DL ablations, walk-forward windows,
  cross-asset generalization), each call draws from wherever the global RNG
  stream happened to be left by everything that ran before it, so the exact
  same nominal SEED could produce different initial weights on different
  runs, or on the same run's different tickers/horizons.
`set_global_determinism()` fixes this by seeding Python/NumPy/TF together
and enabling op-level determinism, and `build_model()` now calls it again
right before building each model so every model's initial weights are
pinned to a known seed regardless of what ran earlier in the process. Pass
a different `seed` to `build_model()` (as the REPRO_SEEDS sweep in main.py
does) to deliberately vary it.
"""
import os
import random

# Must be set before TensorFlow is imported to take effect.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, Model

from config import (
    CNN_FILTERS, CNN_KERNEL_SIZE,
    LSTM_UNITS, NUM_ATTENTION_HEADS, ATTENTION_KEY_DIM,
    DROPOUT_RATE, LEARNING_RATE, L2_REG,
    BATCH_SIZE, EPOCHS, PATIENCE, VALIDATION_SPLIT, SEED,
)


def set_global_determinism(seed: int = SEED) -> None:
    """Seed Python/NumPy/TensorFlow together and enable op-level determinism."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        keras.utils.set_random_seed(seed)
    except Exception:
        pass
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Older TF versions (<2.8) don't expose this; TF_DETERMINISTIC_OPS
        # above still covers most conv/LSTM kernels on those versions.
        pass


set_global_determinism(SEED)


# ──────────────────────────────────────────────────────────────
# Custom Attention layer that exposes weights
# ──────────────────────────────────────────────────────────────
class AttentionWithWeights(layers.Layer):
    """Multi-Head Attention wrapper that caches attention_weights."""

    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim
        )

    def call(self, query, value, training=False):
        attn_output, attn_weights = self.mha(
            query, value, return_attention_scores=True, training=training
        )
        # Store for later extraction
        self.last_attn_weights = attn_weights
        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.mha._num_heads,
            "key_dim": self.mha._key_dim,
        })
        return config


# ──────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────
def build_model(window_size: int, num_features: int, seed: int | None = None) -> Model:
    """
    Build and compile the CNN-BiLSTM-Attention model with L2 regularization.
    Reseeds Python/NumPy/TF immediately before construction so weight
    initialization is deterministic and reproducible across runs regardless
    of how much other random-drawing code ran earlier in the process. Pass
    `seed` to deliberately build a model with a different initialization
    (used by the REPRO_SEEDS sweep in main.py); omitted, it uses the global
    config SEED.
    Returns: keras.Model
    """
    set_global_determinism(seed if seed is not None else SEED)
    reg = keras.regularizers.l2(L2_REG)
    inp = layers.Input(shape=(window_size, num_features), name="input")

    # ── 1D-CNN block ──
    x = layers.Conv1D(
        filters=CNN_FILTERS, kernel_size=CNN_KERNEL_SIZE,
        activation="relu", padding="same", name="conv1d",
        kernel_regularizer=reg
    )(inp)
    x = layers.BatchNormalization(name="bn_cnn")(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop_cnn")(x)

    # ── Bidirectional LSTM block ──
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=True, name="lstm",
                    kernel_regularizer=reg),
        name="bi_lstm"
    )(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop_lstm")(x)

    # ── Multi-Head Attention block ──
    mha = layers.MultiHeadAttention(
        num_heads=NUM_ATTENTION_HEADS,
        key_dim=ATTENTION_KEY_DIM,
        name="mha",
    )
    attn_output, attn_scores = mha(
        x, x, return_attention_scores=True
    )
    x = layers.LayerNormalization(name="ln_attn")(attn_output)

    # ── Aggregation & output ──
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="dense1", kernel_regularizer=reg)(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop_dense")(x)
    out = layers.Dense(1, activation="linear", name="output", kernel_regularizer=reg)(x)

    model = Model(inputs=inp, outputs=out, name="CNN_BiLSTM_Attention")
    attention_model = Model(
        inputs=inp,
        outputs=[out, attn_scores],
        name="CNN_BiLSTM_Attention_Probe",
    )
    model.attention_probe = attention_model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────
def train_model(
    model: Model,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    epochs: int = EPOCHS,
    verbose: int = 1,
):
    """
    Train with EarlyStopping on a held-out validation split from train (not test).
    X_test, y_test are kept for backward compatibility but no longer used in fit.
    """
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=verbose,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, verbose=verbose,
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=epochs, batch_size=BATCH_SIZE,
        callbacks=cb, verbose=verbose,
    )
    return history


# ──────────────────────────────────────────────────────────────
# Attention weight extraction
# ──────────────────────────────────────────────────────────────
def build_attention_model(trained_model: Model) -> Model:
    """
    Build a secondary model that outputs both prediction and
    attention weights for interpretability.
    """
    if not hasattr(trained_model, "attention_probe"):
        raise ValueError("Model does not expose an attention_probe model.")
    return trained_model.attention_probe


def extract_attention_weights(trained_model: Model, X_sample: np.ndarray):
    """
    Run forward pass on X_sample and return attention weights.
    Shape: (batch, num_heads, window_size, window_size)
    """
    attn_model = build_attention_model(trained_model)
    _, attn_weights = attn_model.predict(X_sample, verbose=0)
    return attn_weights
