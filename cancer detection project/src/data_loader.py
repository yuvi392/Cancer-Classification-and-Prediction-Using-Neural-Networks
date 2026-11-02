"""Load and validate the built-in Wisconsin Diagnostic dataset."""
from dataclasses import dataclass
import logging
import pandas as pd
from sklearn.datasets import load_breast_cancer

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class DatasetBundle:
    features: pd.DataFrame
    target: pd.Series
    feature_names: list[str]
    target_names: list[str]
    original_target: pd.Series

def load_dataset() -> DatasetBundle:
    raw = load_breast_cancer(as_frame=True)
    features = raw.data.copy()
    features.columns = list(raw.feature_names)
    # sklearn encodes malignant=0 and benign=1. Internally, malignant=1.
    target = (raw.target == 0).astype("int64")
    target.name = "malignant"
    return DatasetBundle(features, target, list(raw.feature_names), ["benign", "malignant"], raw.target.copy())

def dataset_summary(bundle: DatasetBundle) -> dict[str, object]:
    return {
        "rows": int(len(bundle.features)),
        "features": int(bundle.features.shape[1]),
        "missing_values": int(bundle.features.isna().sum().sum()),
        "duplicate_rows": int(bundle.features.duplicated().sum()),
        "class_counts": {"benign": int((bundle.target == 0).sum()), "malignant": int((bundle.target == 1).sum())},
        "dtypes": {str(k): str(v) for k, v in bundle.features.dtypes.items()},
    }
