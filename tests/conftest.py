import pytest
import pandas as pd
import os
from pathlib import Path

# Define base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"

def _load_csv_or_skip(filepath, description):
    """Helper function to load CSV or skip test if file doesn't exist"""
    if not filepath.exists():
        pytest.skip(f"{description} not found at {filepath}")
    return pd.read_csv(filepath)

@pytest.fixture
def fraud_data_cleaned():
    """Load cleaned Fraud_Data"""
    path = PROCESSED_DIR / "Fraud_Data_cleaned.csv"
    return _load_csv_or_skip(path, "Cleaned fraud data")

@pytest.fixture
def credit_data_cleaned():
    """Load cleaned creditcard data"""
    path = PROCESSED_DIR / "creditcard_cleaned.csv"
    return _load_csv_or_skip(path, "Cleaned credit card data")

@pytest.fixture
def fraud_data_features():
    """Load feature-engineered Fraud_Data"""
    path = PROCESSED_DIR / "Fraud_Data_features.csv"
    return _load_csv_or_skip(path, "Feature-engineered fraud data")

@pytest.fixture
def fraud_X_train():
    """Load Fraud_Data training features"""
    path = PROCESSED_DIR / "Fraud_X_train.csv"
    return _load_csv_or_skip(path, "Fraud X_train")

@pytest.fixture
def fraud_X_test():
    """Load Fraud_Data test features"""
    path = PROCESSED_DIR / "Fraud_X_test.csv"
    return _load_csv_or_skip(path, "Fraud X_test")

@pytest.fixture
def fraud_y_train():
    """Load Fraud_Data training labels"""
    path = PROCESSED_DIR / "Fraud_y_train.csv"
    return _load_csv_or_skip(path, "Fraud y_train")

@pytest.fixture
def fraud_y_test():
    """Load Fraud_Data test labels"""
    path = PROCESSED_DIR / "Fraud_y_test.csv"
    return _load_csv_or_skip(path, "Fraud y_test")

@pytest.fixture
def credit_X_train():
    """Load Credit Card training features"""
    path = PROCESSED_DIR / "Credit_X_train.csv"
    return _load_csv_or_skip(path, "Credit X_train")

@pytest.fixture
def credit_X_test():
    """Load Credit Card test features"""
    path = PROCESSED_DIR / "Credit_X_test.csv"
    return _load_csv_or_skip(path, "Credit X_test")

@pytest.fixture
def credit_y_train():
    """Load Credit Card training labels"""
    path = PROCESSED_DIR / "Credit_y_train.csv"
    return _load_csv_or_skip(path, "Credit y_train")

@pytest.fixture
def credit_y_test():
    """Load Credit Card test labels"""
    path = PROCESSED_DIR / "Credit_y_test.csv"
    return _load_csv_or_skip(path, "Credit y_test")
