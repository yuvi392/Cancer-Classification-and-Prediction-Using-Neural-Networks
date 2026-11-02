"""Leakage-safe preprocessing and stratified data splitting."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE

@dataclass
class DataSplits:
    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler

def validate_features(values: object, expected_count: int = 30) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != expected_count:
        raise ValueError(f"Expected {expected_count} numeric features in the documented order.")
    if not np.isfinite(array).all():
        raise ValueError("Features must contain only finite numeric values.")
    return array

def make_splits(features: pd.DataFrame, target: pd.Series) -> DataSplits:
    x = validate_features(features.to_numpy(), features.shape[1])
    y = np.asarray(target, dtype="int64")
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=TEST_SIZE + VALIDATION_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    validation_fraction = VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp, y_temp, test_size=1 - validation_fraction, stratify=y_temp, random_state=RANDOM_STATE
    )
    scaler = StandardScaler().fit(x_train)
    return DataSplits(scaler.transform(x_train), scaler.transform(x_validation), scaler.transform(x_test),
                      y_train, y_validation, y_test, scaler)
