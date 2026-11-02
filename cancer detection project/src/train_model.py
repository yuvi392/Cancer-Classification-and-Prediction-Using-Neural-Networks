"""Train the logistic baseline and configurable TensorFlow ANN."""
import json
import logging
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.data_loader import load_dataset
from src.preprocessing import make_splits
from config import MODEL_PATH, MODELS_DIR, RANDOM_STATE, SCALER_PATH, ensure_directories

LOGGER = logging.getLogger(__name__)

def build_ann(input_size: int = 30):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_size,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train(epochs: int = 150, batch_size: int = 32) -> dict[str, object]:
    import tensorflow as tf
    ensure_directories()
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    bundle = load_dataset()
    splits = make_splits(bundle.features, bundle.target)
    baseline = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(splits.x_train, splits.y_train)
    model = build_ann(len(bundle.feature_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=8, factor=0.5),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(splits.x_train, splits.y_train, validation_data=(splits.x_validation, splits.y_validation),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
    model.save(MODEL_PATH)
    joblib.dump(splits.scaler, SCALER_PATH)
    metadata = {"feature_names": bundle.feature_names, "target_mapping": {"0": "benign", "1": "malignant"},
                "random_state": RANDOM_STATE, "history": {k: [float(v) for v in values] for k, values in history.history.items()},
                "baseline_model": baseline}
    # LogisticRegression is intentionally saved separately; JSON stores only serializable metadata.
    joblib.dump(baseline, MODELS_DIR / "logistic_regression.joblib")
    metadata.pop("baseline_model")
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez(MODELS_DIR / "test_split.npz", x_test=splits.x_test, y_test=splits.y_test,
             x_validation=splits.x_validation, y_validation=splits.y_validation)
    return metadata
