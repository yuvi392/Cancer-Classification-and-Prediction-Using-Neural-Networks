"""Inference using the exact scaler and feature order used during training."""
import json
from pathlib import Path
import joblib
import numpy as np
from config import MODEL_PATH, MODELS_DIR, SCALER_PATH
from src.preprocessing import validate_features

def predict(values: object, model_path: Path = MODEL_PATH) -> dict[str, object]:
    if not model_path.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Model artifacts are missing. Run `python scripts/train.py` first.")
    metadata_path = MODELS_DIR / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Model metadata is missing. Retrain the project.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = len(metadata["feature_names"])
    array = validate_features(values, expected)
    scaler = joblib.load(SCALER_PATH)
    model = _load_model(model_path)
    malignant_probability = float(np.asarray(model.predict(scaler.transform(array), verbose=0)).reshape(-1)[0])
    malignant_probability = min(max(malignant_probability, 0.0), 1.0)
    return {"class": "malignant" if malignant_probability >= 0.5 else "benign",
            "malignant_probability": malignant_probability,
            "benign_probability": 1.0 - malignant_probability,
            "explanation": "These probabilities are model estimates and must not be treated as a diagnosis or clinical certainty."}

def _load_model(path: Path):
    import tensorflow as tf
    return tf.keras.models.load_model(path)
