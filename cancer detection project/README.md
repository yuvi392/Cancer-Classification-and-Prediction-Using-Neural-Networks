# Cancer Classification and Prediction Using Neural Networks

An educational end-to-end machine-learning project using the Breast Cancer Wisconsin Diagnostic dataset. It compares a scaled Logistic Regression baseline with a TensorFlow/Keras feed-forward ANN and exposes the trained ANN through Flask.

## Setup on Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow support varies by Python version and platform. The currently selected interpreter in this workspace is Python 3.14.7, while TensorFlow wheels may lag behind new Python releases. For a reliable Windows setup, use a supported 64-bit Python 3.11–3.13 interpreter, create the virtual environment with that interpreter, and then install the requirements.

## Train, evaluate, test, and run

```powershell
python scripts/train.py
python scripts/evaluate.py
pytest -q
python scripts/run_app.py
```

Then open `http://127.0.0.1:5000`. Training creates real model artifacts under `models/`; evaluation creates real metrics and charts under `reports/`. The dashboard remains empty until those commands complete.

## Project structure

`src/data_loader.py` loads and validates data; `src/preprocessing.py` performs leakage-safe stratified splitting and scaling; `src/train_model.py` trains both models; `src/evaluate_model.py` evaluates the held-out test set; `src/predict.py` performs validated inference; `app.py` serves the website.

## Metrics

Accuracy is the fraction correct. Precision measures how many predicted malignant cases are malignant. Recall/sensitivity measures how many malignant cases are found. Specificity measures how many benign cases are correctly rejected. F1 balances precision and recall; ROC-AUC summarizes ranking quality across thresholds.

## Medical disclaimer

This is a classroom demonstration, not a medical device or diagnosis system. It must not be used to make patient-care decisions. The dataset is small, historical, and not representative of every clinical population.

## Troubleshooting

If Flask reports missing artifacts, run training first. If TensorFlow cannot be imported, confirm that the virtual environment is active and that the selected Python version is supported. The notebook can be opened with `jupyter notebook` after dependencies are installed.
