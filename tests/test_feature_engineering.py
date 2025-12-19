import pytest
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def test_fraud_data_features_exists():
    """Verify feature-engineered Fraud_Data file exists"""
    path = PROCESSED_DIR / "Fraud_Data_features.csv"
    assert path.exists(), f"Feature-engineered fraud data not found at {path}"


def test_geolocation_integration(fraud_data_features):
    """Verify country column exists from geolocation integration"""
    assert 'country' in fraud_data_features.columns, "Missing 'country' column from geolocation integration"
    
    # Check that country has values (not all Unknown/NaN)
    valid_countries = fraud_data_features['country'].notna().sum()
    assert valid_countries > 0, "No valid country mappings found"


def test_time_since_signup_exists(fraud_data_features):
    """Verify time_since_signup feature exists"""
    assert 'time_since_signup' in fraud_data_features.columns, "Missing 'time_since_signup' feature"


def test_time_since_signup_calculation(fraud_data_features):
    """Verify time_since_signup values are reasonable"""
    time_since_signup = fraud_data_features['time_since_signup']
    
    # Should be positive (purchase after signup)
    assert (time_since_signup >= 0).all(), "Found negative time_since_signup values"
    
    # Should have some variation
    assert time_since_signup.std() > 0, "time_since_signup has no variation"


def test_hour_of_day_feature(fraud_data_features):
    """Verify hour_of_day feature exists and is valid"""
    assert 'hour_of_day' in fraud_data_features.columns, "Missing 'hour_of_day' feature"
    
    hour_of_day = fraud_data_features['hour_of_day']
    
    # Should be in range [0, 23]
    assert hour_of_day.min() >= 0, "hour_of_day has values < 0"
    assert hour_of_day.max() <= 23, "hour_of_day has values > 23"


def test_day_of_week_feature(fraud_data_features):
    """Verify day_of_week feature exists and is valid"""
    assert 'day_of_week' in fraud_data_features.columns, "Missing 'day_of_week' feature"
    
    day_of_week = fraud_data_features['day_of_week']
    
    # Should be in range [0, 6] (Monday=0, Sunday=6)
    assert day_of_week.min() >= 0, "day_of_week has values < 0"
    assert day_of_week.max() <= 6, "day_of_week has values > 6"


def test_user_id_count_feature(fraud_data_features):
    """Verify user_id_count velocity feature exists"""
    assert 'user_id_count' in fraud_data_features.columns, "Missing 'user_id_count' velocity feature"
    
    # Should be at least 1 for all users
    assert (fraud_data_features['user_id_count'] >= 1).all(), "user_id_count has values < 1"


def test_device_id_count_feature(fraud_data_features):
    """Verify device_id_count velocity feature exists"""
    assert 'device_id_count' in fraud_data_features.columns, "Missing 'device_id_count' velocity feature"
    
    # Should be at least 1 for all devices
    assert (fraud_data_features['device_id_count'] >= 1).all(), "device_id_count has values < 1"


def test_all_engineered_features_present(fraud_data_features):
    """Verify all expected engineered features are present"""
    required_features = [
        'country',
        'time_since_signup',
        'hour_of_day',
        'day_of_week',
        'user_id_count',
        'device_id_count'
    ]
    
    for feature in required_features:
        assert feature in fraud_data_features.columns, f"Missing required feature: {feature}"


def test_original_columns_preserved(fraud_data_features):
    """Verify original columns are still present after feature engineering"""
    original_columns = [
        'user_id', 'signup_time', 'purchase_time', 'purchase_value',
        'device_id', 'source', 'browser', 'sex', 'age', 'ip_address', 'class'
    ]
    
    for col in original_columns:
        assert col in fraud_data_features.columns, f"Original column missing: {col}"


def test_no_missing_values_in_engineered_features(fraud_data_features):
    """Verify no missing values in newly engineered features"""
    engineered_features = [
        'country', 'time_since_signup', 'hour_of_day',
        'day_of_week', 'user_id_count', 'device_id_count'
    ]
    
    for feature in engineered_features:
        missing = fraud_data_features[feature].isnull().sum()
        assert missing == 0, f"Feature '{feature}' has {missing} missing values"
