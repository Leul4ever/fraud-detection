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

@pytest.fixture
def fraud_data_cleaned():
    """Load cleaned Fraud_Data"""
    path = PROCESSED_DIR / "Fraud_Data_cleaned.csv"
    if not path.exists():
        pytest.skip(f"Cleaned fraud data not found at {path}")
    return pd.read_csv(path)

@pytest.fixture
def credit_data_cleaned():
    """Load cleaned creditcard data"""
    path = PROCESSED_DIR / "creditcard_cleaned.csv"
    if not path.exists():
        pytest.skip(f"Cleaned credit card data not found at {path}")
    return pd.read_csv(path)

@pytest.fixture
def fraud_data_features():
    """Load feature-engineered Fraud_Data"""
    path = PROCESSED_DIR / "Fraud_Data_features.csv"
    if not path.exists():
        pytest.skip(f"Feature-engineered fraud data not found at {path}")
    return pd.read_csv(path)

@pytest.fixture
def fraud_X_train():
    """Load Fraud_Data training features"""
    path = PROCESSED_DIR / "Fraud_X_train.csv"
    if not path.exists():
        pytest.skip(f"Fraud X_train not found at {path}")
    return pd.read_csv(path)

@pytest.fixture
def fraud_X_test():
    """Load Fraud_Data test features"""
    path = PROCESSED_DIR / "Fraud_X_test.csv"
    if not path.exists():
        pytest.skip(f"Fraud X_test not found at {path}")
    return pd.read_csv(path)

@pytest.fixture
def fraud_y_train():
    """Load Fraud_Data training labels"""
    path = PROCESSED_DIR / "Fraud_y_train.csv"
    if not path.exists():
        pytest.skip(f"Fraud y_train not found at {path}")
    return pd.read_csv(path)

@pytest.fixture
def fraud_y_test():
    """Load Fraud_Data test labels"""
    path = PROCESSED_DIR / "Fraud_y_test.csv"
    if not path.exists():
        pytest.skip(f"Fraud y_test not found at {path}")
    return pd.read_csv(path)

@pytest.fixture
def credit_X_train():
    """Load Credit Card training features"""
    path = PROCESSED_DIR / "Credit_X_train.csv"
    if not path.exists():
        pytest.skip(f"Credit X_train not found at {path}")
    return pd.read_csv(path)

@pytest.fixture
def credit_X_test():
    """Load Credit Card test features"""
    path = PROCESSED_DIR / "Credit_X_test.csv"
    if not path.exists():
        pytest.skip(f"Credit X_test not found at {path}")
    return pd.read_csv(path)

@pytest.fixture
def credit_y_train():
    """Load Credit Card training labels"""
    path = PROCESSED_DIR / "Credit_y_train.csv"
    if not path.exists():
        pytest.skip(f"Credit y_train not found at {path}")
    return pd.read_csv(path)

@pytest.fixture
def credit_y_test():
    """Load Credit Card test labels"""
    path = PROCESSED_DIR / "Credit_y_test.csv"
    if not path.exists():
        pytest.skip(f"Credit y_test not found at {path}")
    return pd.read_csv(path)
