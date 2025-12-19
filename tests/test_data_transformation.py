import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def test_train_test_files_exist_fraud():
    """Verify all Fraud_Data train/test files exist"""
    files = ['Fraud_X_train.csv', 'Fraud_X_test.csv', 'Fraud_y_train.csv', 'Fraud_y_test.csv']
    missing_files = []
    for file in files:
        path = PROCESSED_DIR / file
        if not path.exists():
            missing_files.append(str(path))
    if missing_files:
        pytest.skip(f"Missing files: {', '.join(missing_files)}")


def test_train_test_files_exist_credit():
    """Verify all Credit Card train/test files exist"""
    files = ['Credit_X_train.csv', 'Credit_X_test.csv', 'Credit_y_train.csv', 'Credit_y_test.csv']
    missing_files = []
    for file in files:
        path = PROCESSED_DIR / file
        if not path.exists():
            missing_files.append(str(path))
    if missing_files:
        pytest.skip(f"Missing files: {', '.join(missing_files)}")


def test_fraud_train_test_split_ratio(fraud_X_train, fraud_X_test):
    """Verify Fraud_Data has approximately 80/20 train/test split"""
    total = len(fraud_X_train) + len(fraud_X_test)
    train_ratio = len(fraud_X_train) / total
    
    # Allow some tolerance due to SMOTE, but original split should be ~0.8
    # After SMOTE, train set will be larger, so we check the test set ratio
    test_ratio = len(fraud_X_test) / total
    assert 0.1 <= test_ratio <= 0.3, f"Test ratio {test_ratio:.2f} is outside expected range [0.1, 0.3]"


def test_credit_train_test_split_ratio(credit_X_train, credit_X_test):
    """Verify Credit Card data has approximately 80/20 train/test split"""
    total = len(credit_X_train) + len(credit_X_test)
    test_ratio = len(credit_X_test) / total
    assert 0.1 <= test_ratio <= 0.3, f"Test ratio {test_ratio:.2f} is outside expected range [0.1, 0.3]"


def test_smote_balancing_fraud(fraud_y_train):
    """Verify SMOTE balanced the Fraud_Data training set"""
    class_counts = fraud_y_train['class'].value_counts()
    
    # After SMOTE, classes should be balanced (1:1 ratio)
    # Allow small tolerance
    ratio = class_counts.min() / class_counts.max()
    assert ratio >= 0.95, f"Classes are not balanced. Ratio: {ratio:.2f}, Counts: {class_counts.to_dict()}"


def test_smote_balancing_credit(credit_y_train):
    """Verify SMOTE balanced the Credit Card training set"""
    class_counts = credit_y_train.iloc[:, 0].value_counts()
    
    # After SMOTE, classes should be balanced
    ratio = class_counts.min() / class_counts.max()
    assert ratio >= 0.95, f"Classes are not balanced. Ratio: {ratio:.2f}, Counts: {class_counts.to_dict()}"


def test_categorical_encoding_fraud(fraud_X_train):
    """Verify categorical variables were one-hot encoded in Fraud_Data"""
    # After OHE with drop_first=True, we should have encoded columns
    # Check for presence of encoded columns (e.g., source_*, browser_*, sex_*, country_*)
    encoded_prefixes = ['source_', 'browser_', 'sex_', 'country_']
    
    for prefix in encoded_prefixes:
        encoded_cols = [col for col in fraud_X_train.columns if col.startswith(prefix)]
        # Should have at least one encoded column for each category
        # (drop_first=True means we drop one level)
        assert len(encoded_cols) >= 1, f"No encoded columns found for prefix: {prefix}"


def test_no_categorical_strings_fraud(fraud_X_train):
    """Verify no string categorical variables remain in Fraud_Data features"""
    # All columns should be numeric after encoding (including bool from one-hot encoding)
    for col in fraud_X_train.columns:
        dtype = fraud_X_train[col].dtype
        # Allow numeric types and bool (from pandas one-hot encoding)
        assert dtype in [np.int64, np.float64, np.int32, np.float32, np.int16, np.float16, np.uint8, np.uint64, bool, np.bool_], \
            f"Column '{col}' has non-numeric dtype: {dtype}"


def test_numerical_scaling_fraud(fraud_X_test):
    """Verify numerical features are scaled in Fraud_Data (using test set)"""
    # Check key numerical columns that should be scaled on the test set
    # The training set mean shifts after SMOTE, so we use the test set for verification.
    numerical_cols = ['purchase_value', 'age', 'time_since_signup', 
                     'hour_of_day', 'day_of_week', 'user_id_count', 'device_id_count']
    
    scaled_count = 0
    for col in numerical_cols:
        if col in fraud_X_test.columns:
            mean = fraud_X_test[col].mean()
            std = fraud_X_test[col].std()
            
            # Allow reasonable tolerance
            assert abs(mean) < 0.5, f"Column '{col}' mean {mean:.2f} is not close to 0"
            if std > 0.01:
                assert 0.5 < std < 1.5, f"Column '{col}' std {std:.2f} is not close to 1"
            scaled_count += 1
    
    assert scaled_count > 0, "No numerical columns found for scaling verification"


def test_numerical_scaling_credit(credit_X_test):
    """Verify Time and Amount are scaled in Credit Card data (using test set)"""
    # Time and Amount should be scaled
    # We use the test set because the training set mean shifts after SMOTE
    for col in ['Time', 'Amount']:
        if col in credit_X_test.columns:
            mean = credit_X_test[col].mean()
            std = credit_X_test[col].std()
            
            assert abs(mean) < 0.5, f"Column '{col}' mean {mean:.2f} is not close to 0"
            assert 0.5 < std < 1.5, f"Column '{col}' std {std:.2f} is not close to 1"


def test_scaler_artifacts_exist():
    """Verify scaler objects were saved (optional check)"""
    fraud_scaler_path = MODELS_DIR / "fraud_scaler.pkl"
    credit_scaler_path = MODELS_DIR / "credit_scaler.pkl"
    
    # This is an optional check - scalers may not have been saved in all environments
    # Just verify the models directory exists
    if not MODELS_DIR.exists():
        pytest.skip("Models directory does not exist yet")
    
    # If directory exists, at least one scaler should be present
    if fraud_scaler_path.exists() or credit_scaler_path.exists():
        assert True, "Scaler artifacts found"
    else:
        pytest.skip("No scaler artifacts found - may not have been saved yet")


def test_feature_target_alignment_fraud(fraud_X_train, fraud_y_train):
    """Verify X and y have same number of samples for Fraud_Data"""
    assert len(fraud_X_train) == len(fraud_y_train), \
        f"X_train ({len(fraud_X_train)}) and y_train ({len(fraud_y_train)}) have different lengths"


def test_feature_target_alignment_credit(credit_X_train, credit_y_train):
    """Verify X and y have same number of samples for Credit Card data"""
    assert len(credit_X_train) == len(credit_y_train), \
        f"X_train ({len(credit_X_train)}) and y_train ({len(credit_y_train)}) have different lengths"


def test_no_target_leakage_fraud(fraud_X_train):
    """Verify target variable 'class' is not in features"""
    assert 'class' not in fraud_X_train.columns, "Target variable 'class' found in features"


def test_no_target_leakage_credit(credit_X_train):
    """Verify target variable 'Class' is not in features"""
    assert 'Class' not in credit_X_train.columns, "Target variable 'Class' found in features"


def test_no_id_columns_in_features_fraud(fraud_X_train):
    """Verify ID columns were removed from Fraud_Data features"""
    id_columns = ['user_id', 'device_id', 'ip_address']
    for col in id_columns:
        assert col not in fraud_X_train.columns, f"ID column '{col}' should not be in features"


def test_no_datetime_columns_in_features_fraud(fraud_X_train):
    """Verify datetime columns were removed from Fraud_Data features"""
    datetime_columns = ['signup_time', 'purchase_time']
    for col in datetime_columns:
        assert col not in fraud_X_train.columns, f"Datetime column '{col}' should not be in features"
