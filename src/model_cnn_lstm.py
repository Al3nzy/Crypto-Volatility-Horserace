"""
Multimodal Deep Learning Module (Section 3.3).

CNN-LSTM with Multi-Head Attention for cryptocurrency volatility forecasting.
Architecture:
    Input → 1D-CNN → Bi-LSTM → Multi-Head Attention → Dense → Output
"""
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

tf.random.set_seed(SEED)


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
def build_model(window_size: int, num_features: int) -> Model:
    """
    Build and compile the CNN-BiLSTM-Attention model with L2 regularization.
    Returns: keras.Model
    """
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
