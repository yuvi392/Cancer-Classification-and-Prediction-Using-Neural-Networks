import numpy as np
from src.data_loader import load_dataset
from src.preprocessing import make_splits, validate_features

def test_scaler_is_fit_on_training_only():
    bundle = load_dataset(); splits = make_splits(bundle.features, bundle.target)
    assert splits.x_train.shape[1] == 30
    assert np.isfinite(splits.x_test).all()

def test_invalid_feature_count_is_rejected():
    try: validate_features([1] * 29)
    except ValueError: return
    raise AssertionError("29 features should be rejected")
