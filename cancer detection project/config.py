"""Central paths and reproducibility settings for the project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

MODEL_PATH = MODELS_DIR / "ann_model.keras"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

def ensure_directories() -> None:
    for directory in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
