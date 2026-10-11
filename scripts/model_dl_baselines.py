"""
Ablated deep baselines: CNN-only, unidirectional LSTM, GRU — same input tensor
and training loop as the main CNN-BiLSTM-Attention model.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error
from tensorflow import keras
from tensorflow.keras import layers, Model

from config import (
    CNN_FILTERS,
    CNN_KERNEL_SIZE,
    DROPOUT_RATE,
    L2_REG,
    LEARNING_RATE,
    LSTM_UNITS,
)
from src.evaluate import rmse
from src.model_cnn_lstm import set_global_determinism, train_model


def build_cnn_only(window_size: int, num_features: int) -> Model:
    reg = keras.regularizers.l2(L2_REG)
    inp = layers.Input(shape=(window_size, num_features), name="input")
    x = layers.Conv1D(
        filters=CNN_FILTERS,
        kernel_size=CNN_KERNEL_SIZE,
        activation="relu",
        padding="same",
        kernel_regularizer=reg,
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(1, activation="linear", kernel_regularizer=reg)(x)
    model = Model(inputs=inp, outputs=out, name="CNN_only")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_lstm_only(window_size: int, num_features: int) -> Model:
    reg = keras.regularizers.l2(L2_REG)
    inp = layers.Input(shape=(window_size, num_features), name="input")
    x = layers.LSTM(LSTM_UNITS, return_sequences=False, kernel_regularizer=reg)(inp)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    out = layers.Dense(1, activation="linear", kernel_regularizer=reg)(x)
    model = Model(inputs=inp, outputs=out, name="LSTM_only")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_gru_only(window_size: int, num_features: int) -> Model:
    reg = keras.regularizers.l2(L2_REG)
    inp = layers.Input(shape=(window_size, num_features), name="input")
    x = layers.GRU(LSTM_UNITS, return_sequences=False, kernel_regularizer=reg)(inp)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    out = layers.Dense(1, activation="linear", kernel_regularizer=reg)(x)
    model = Model(inputs=inp, outputs=out, name="GRU_only")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def run_dl_baseline_variant(
    pipeline: dict,
    variant: str,
) -> dict:
    """
    variant: "cnn_only" | "lstm_only" | "gru_only"
    """
    X_train = pipeline["X_train"]
    X_test = pipeline["X_test"]
    y_train = pipeline["y_train"]
    y_test = pipeline["y_test"]
    tgt = pipeline["tgt_scaler"]
    ws, nf = X_train.shape[1], X_train.shape[2]

    builders = {
        "cnn_only": build_cnn_only,
        "lstm_only": build_lstm_only,
        "gru_only": build_gru_only,
    }
    labels = {
        "cnn_only": "CNN-only",
        "lstm_only": "LSTM-only",
        "gru_only": "GRU-only",
    }
    if variant not in builders:
        raise ValueError(f"Unknown variant: {variant}")

    set_global_determinism()  # reseed immediately before construction, same as build_model()
    model = builders[variant](ws, nf)
    train_model(model, X_train, y_train, X_test, y_test)
    pred_s = model.predict(X_test, verbose=0)
    y_pred = tgt.inverse_transform(pred_s).flatten()
    y_actual = tgt.inverse_transform(y_test).flatten()
    n = min(len(y_pred), len(y_actual))
    dates = pipeline["test_dates"][:n]

    return {
        "name": labels[variant],
        "predictions": y_pred[:n],
        "actual": y_actual[:n],
        "dates": dates,
        "rmse": rmse(y_actual[:n], y_pred[:n]),
        "mae": mean_absolute_error(y_actual[:n], y_pred[:n]),
        "model": model,
    }
