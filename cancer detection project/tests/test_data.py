from src.data_loader import load_dataset

def test_dataset_shape_and_mapping():
    bundle = load_dataset()
    assert bundle.features.shape == (569, 30)
    assert set(bundle.target.unique()) == {0, 1}
    assert bundle.target.iloc[0] == 1  # original first target is malignant (0)
