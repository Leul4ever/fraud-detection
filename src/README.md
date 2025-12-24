# Source Code (src)

This directory contains the primary Python modules for the fraud detection system.

## Modules

- **`data_cleaning.py`**: Handles missing values, duplicates, and data type corrections for raw datasets.
- **`feature_engineering.py`**: Implements geolocation mapping, time-based features, and velocity metrics.
- **`data_transformation.py`**: Manages scaling (StandardScaler), encoding (One-Hot Encoding), and class imbalance handling (SMOTE).
- **`model_training.py`**: orchestrates model training, cross-validation, and performance evaluation.

## Design Pattern

Code in this directory is designed to be modular and reusable. These modules are utilized by the execution scripts in `scripts/` and by the analysis notebooks in `notebooks/`.
