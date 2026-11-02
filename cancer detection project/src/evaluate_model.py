"""Evaluate saved models on the held-out test split and generate real charts."""
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.data_loader import load_dataset
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from config import FIGURES_DIR, METRICS_DIR, MODEL_PATH, MODELS_DIR, ensure_directories

def evaluate() -> dict[str, object]:
    import tensorflow as tf
    ensure_directories()
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the models before evaluation: python scripts/train.py")
    split = np.load(MODELS_DIR / "test_split.npz")
    x_test, y_test = split["x_test"], split["y_test"]
    ann = tf.keras.models.load_model(MODEL_PATH)
    baseline = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    probabilities = {"ANN": ann.predict(x_test, verbose=0).reshape(-1), "Logistic Regression": baseline.predict_proba(x_test)[:, 1]}
    metrics: dict[str, object] = {}
    for name, scores in probabilities.items():
        predicted = (scores >= 0.5).astype(int)
        matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        metrics[name] = {"accuracy": float(accuracy_score(y_test, predicted)),
                         "precision": float(precision_score(y_test, predicted, zero_division=0)),
                         "recall_sensitivity_malignant": float(recall_score(y_test, predicted, zero_division=0)),
                         "specificity": float(tn / (tn + fp) if tn + fp else 0),
                         "f1": float(f1_score(y_test, predicted, zero_division=0)),
                         "roc_auc": float(roc_auc_score(y_test, scores)), "confusion_matrix": matrix.tolist()}
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_dataset()
    plt.figure(figsize=(5, 4)); sns.countplot(x=bundle.target.map({0: "benign", 1: "malignant"})); plt.xlabel("Class"); plt.ylabel("Count"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "class_distribution.png"); plt.close()
    plt.figure(figsize=(12, 10)); sns.heatmap(bundle.features.corr(), cmap="vlag", center=0, xticklabels=False, yticklabels=False); plt.title("Feature correlation"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "feature_correlation.png"); plt.close()
    plt.figure(figsize=(6, 5)); sns.heatmap(metrics["ANN"]["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "confusion_matrix.png"); plt.close()
    plt.figure(figsize=(7, 5))
    for name, scores in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, scores); plt.plot(fpr, tpr, label=f"{name} (AUC={metrics[name]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.tight_layout(); plt.savefig(FIGURES_DIR / "roc_curve.png"); plt.close()
    history_path = MODELS_DIR / "model_metadata.json"
    history = json.loads(history_path.read_text(encoding="utf-8")).get("history", {})
    if history:
        plt.figure(figsize=(7, 5)); plt.plot(history.get("loss", []), label="training loss"); plt.plot(history.get("val_loss", []), label="validation loss"); plt.legend(); plt.tight_layout(); plt.savefig(FIGURES_DIR / "training_history.png"); plt.close()
    plt.figure(figsize=(8, 5)); names = list(metrics); x = np.arange(len(names)); width = 0.2
    for i, metric_name in enumerate(("accuracy", "precision", "recall_sensitivity_malignant", "roc_auc")):
        plt.bar(x + (i - 1.5) * width, [metrics[n][metric_name] for n in names], width, label=metric_name)
    plt.xticks(x, names, rotation=15); plt.ylim(0, 1); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIGURES_DIR / "model_comparison.png"); plt.close()
    with (METRICS_DIR / "metrics.json").open("w", encoding="utf-8") as handle: json.dump(metrics, handle, indent=2)
    return metrics
