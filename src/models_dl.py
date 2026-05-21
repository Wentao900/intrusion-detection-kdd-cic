"""1D-CNN deep learning model for tabular IDS features."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.config import CNN_BATCH_SIZE, CNN_EPOCHS, CNN_PATIENCE, RANDOM_STATE
from src.metrics import compute_metrics, measure_latency_ms
from src.preprocess import PreprocessResult


def build_cnn_model(input_dim: int, n_classes: int) -> Any:
    """Build Keras 1D-CNN treating features as a sequence."""
    import tensorflow as tf

    tf.random.set_seed(RANDOM_STATE)
    inputs = tf.keras.Input(shape=(input_dim, 1))
    x = tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _reshape_cnn(X: np.ndarray) -> np.ndarray:
    return X[..., np.newaxis].astype(np.float32)


def train_and_evaluate_cnn(
    data: PreprocessResult,
    dataset: str,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Train 1D-CNN and return metrics."""
    import tensorflow as tf

    epochs = epochs or CNN_EPOCHS
    n_classes = len(data.label_encoder.classes_)
    input_dim = data.X_train.shape[1]

    X_train = _reshape_cnn(data.X_train)
    X_val = _reshape_cnn(data.X_val)
    X_test = _reshape_cnn(data.X_test)

    model = build_cnn_model(input_dim, n_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=CNN_PATIENCE, restore_best_weights=True
        )
    ]

    t0 = time.perf_counter()
    model.fit(
        X_train,
        data.y_train,
        validation_data=(X_val, data.y_val),
        epochs=epochs,
        batch_size=CNN_BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )
    train_time = time.perf_counter() - t0

    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    metrics = compute_metrics(
        data.y_test, y_pred, label_names=list(data.label_encoder.classes_)
    )
    metrics["latency_ms_per_sample"] = _cnn_latency_ms(model, X_test)
    metrics["train_time_sec"] = train_time
    metrics["model"] = "cnn_1d"
    metrics["dataset"] = dataset
    metrics["model_type"] = "dl"

    from sklearn.metrics import f1_score
    y_val_proba = model.predict(X_val, verbose=0)
    y_val_pred = np.argmax(y_val_proba, axis=1)
    metrics["val_f1_macro"] = float(
        f1_score(data.y_val, y_val_pred, average="macro", zero_division=0)
    )
    return metrics


def _cnn_latency_ms(model: Any, X: np.ndarray) -> float:
    """Latency for Keras model."""
    import time

    from src.config import LATENCY_SAMPLE_SIZE

    n = min(LATENCY_SAMPLE_SIZE, len(X))
    X_sub = X[:n]
    _ = model.predict(X_sub[: min(50, n)], verbose=0)
    t0 = time.perf_counter()
    _ = model.predict(X_sub, verbose=0)
    return (time.perf_counter() - t0) * 1000 / n
