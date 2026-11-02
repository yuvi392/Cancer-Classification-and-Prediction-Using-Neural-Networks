import pytest
from src.predict import predict

def test_missing_artifacts_fail_helpfully(tmp_path):
    with pytest.raises(FileNotFoundError, match="artifacts"):
        predict([1] * 30, tmp_path / "missing.keras")
