import pytest
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def test_fraud_data_cleaned_exists():
    """Verify cleaned Fraud_Data file exists"""
    path = PROCESSED_DIR / "Fraud_Data_cleaned.csv"
    assert path.exists(), f"Cleaned fraud data not found at {path}"


def test_credit_data_cleaned_exists():
    """Verify cleaned creditcard file exists"""
    path = PROCESSED_DIR / "creditcard_cleaned.csv"
    assert path.exists(), f"Cleaned credit card data not found at {path}"


def test_fraud_data_no_missing_values(fraud_data_cleaned):
    """Verify cleaned Fraud_Data has no missing values"""
    missing_count = fraud_data_cleaned.isnull().sum().sum()
    assert missing_count == 0, f"Found {missing_count} missing values in cleaned Fraud_Data"


def test_credit_data_no_missing_values(credit_data_cleaned):
    """Verify cleaned creditcard data has no missing values"""
    missing_count = credit_data_cleaned.isnull().sum().sum()
    assert missing_count == 0, f"Found {missing_count} missing values in cleaned credit card data"


def test_fraud_data_no_duplicates(fraud_data_cleaned):
    """Verify no duplicates in cleaned Fraud_Data"""
    duplicates = fraud_data_cleaned.duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} duplicate rows in cleaned Fraud_Data"


def test_credit_data_no_duplicates(credit_data_cleaned):
    """Verify no duplicates in cleaned creditcard data"""
    duplicates = credit_data_cleaned.duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} duplicate rows in cleaned credit card data"


def test_fraud_data_datetime_conversion(fraud_data_cleaned):
    """Verify signup_time and purchase_time are datetime objects"""
    # Convert to datetime if they're strings
    signup_time = pd.to_datetime(fraud_data_cleaned['signup_time'], errors='coerce')
    purchase_time = pd.to_datetime(fraud_data_cleaned['purchase_time'], errors='coerce')
    
    # Check no conversion errors
    assert signup_time.isnull().sum() == 0, "signup_time has invalid datetime values"
    assert purchase_time.isnull().sum() == 0, "purchase_time has invalid datetime values"


def test_fraud_data_has_expected_columns(fraud_data_cleaned):
    """Verify Fraud_Data has all expected columns"""
    expected_columns = [
        'user_id', 'signup_time', 'purchase_time', 'purchase_value',
        'device_id', 'source', 'browser', 'sex', 'age', 'ip_address', 'class'
    ]
    for col in expected_columns:
        assert col in fraud_data_cleaned.columns, f"Missing expected column: {col}"


def test_credit_data_has_expected_columns(credit_data_cleaned):
    """Verify creditcard data has expected structure (V1-V28, Time, Amount, Class)"""
    # Should have V1 through V28, Time, Amount, Class = 31 columns
    assert 'Time' in credit_data_cleaned.columns, "Missing 'Time' column"
    assert 'Amount' in credit_data_cleaned.columns, "Missing 'Amount' column"
    assert 'Class' in credit_data_cleaned.columns, "Missing 'Class' column"
    
    # Check for V columns
    v_columns = [f'V{i}' for i in range(1, 29)]
    for col in v_columns:
        assert col in credit_data_cleaned.columns, f"Missing PCA column: {col}"


def test_fraud_data_shape(fraud_data_cleaned):
    """Verify Fraud_Data has reasonable shape"""
    assert fraud_data_cleaned.shape[0] > 0, "Fraud_Data is empty"
    assert fraud_data_cleaned.shape[1] == 11, f"Expected 11 columns, got {fraud_data_cleaned.shape[1]}"


def test_credit_data_shape(credit_data_cleaned):
    """Verify creditcard data has reasonable shape"""
    assert credit_data_cleaned.shape[0] > 0, "Credit card data is empty"
    assert credit_data_cleaned.shape[1] == 31, f"Expected 31 columns, got {credit_data_cleaned.shape[1]}"
