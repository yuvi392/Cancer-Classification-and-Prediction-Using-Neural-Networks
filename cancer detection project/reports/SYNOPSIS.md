# Synopsis

## Title
Cancer Classification and Prediction Using Neural Networks

## Problem and objectives
The project studies whether numerical tumor measurements can be classified into benign and malignant classes in a reproducible educational workflow. Objectives are data exploration, leakage-safe preprocessing, baseline comparison, neural-network training, evaluation, and a local web demonstration.

## Scope and methodology
The built-in scikit-learn dataset is loaded programmatically. StandardScaler is fitted on training data only. Stratified train/validation/test partitions are used. The selected ANN is compared with Logistic Regression using accuracy, precision, malignant recall, specificity, F1, ROC-AUC, and confusion matrices.

## Expected outcomes
Actual outcomes are written to `reports/metrics/metrics.json` after training and evaluation. Insert those values into the final college submission rather than inventing them.

## References
Use the official scikit-learn `load_breast_cancer` documentation and TensorFlow/Keras documentation. Add institution-required bibliographic details before submission.
