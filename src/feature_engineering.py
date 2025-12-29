import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path

def integrate_geolocation(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges fraud data with IP-to-country mapping using range-based lookup.
    
    Args:
        fraud_df (pd.DataFrame): Cleaned fraud data.
        ip_df (pd.DataFrame): IP-to-country mapping data.
        
    Returns:
        pd.DataFrame: Fraud data with country information.
    """
    print("Integrating geolocation...")
    
    # Ensure types match for merge_asof
    fraud_df['ip_address'] = fraud_df['ip_address'].astype(np.int64)
    for col in ['lower_bound_ip_address', 'upper_bound_ip_address']:
        ip_df[col] = ip_df[col].astype(np.int64)
        
    # Sort for merge_asof
    fraud_df = fraud_df.sort_values('ip_address')
    ip_df = ip_df.sort_values('lower_bound_ip_address')
    
    # Range lookup
    df_merged = pd.merge_asof(
        fraud_df, 
        ip_df, 
        left_on='ip_address', 
        right_on='lower_bound_ip_address'
    )
    
    # Validate range
    df_merged['country'] = np.where(
        df_merged['ip_address'] <= df_merged['upper_bound_ip_address'],
        df_merged['country'],
        'Unknown'
    )
    
    return df_merged.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1)

def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes time-based and frequency-based features for fraud detection.
    
    Args:
        df (pd.DataFrame): Fraud data with geolocation.
        
    Returns:
        pd.DataFrame: Data with engineered features.
    """
    print("Engineering features for Fraud_Data...")
    
    # Duration between signup and purchase
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    
    # Time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Frequency/Velocity features
    df['user_id_count'] = df.groupby('user_id')['user_id'].transform('count')
    df['device_id_count'] = df.groupby('device_id')['device_id'].transform('count')
    
    return df

def perform_feature_engineering():
    """Main orchestration for feature engineering task."""
    DATA_DIR = Path('data')
    
    try:
        # Load
        fraud_df = pd.read_csv(DATA_DIR / 'processed/Fraud_Data_cleaned.csv')
        ip_df = pd.read_csv(DATA_DIR / 'raw/IpAddress_to_Country.csv')
        
        # Geolocation
        df_geo = integrate_geolocation(fraud_df, ip_df)
        
        # Features
        df_final = engineer_fraud_features(df_geo)
        
        # Save
        df_final.to_csv(DATA_DIR / 'processed/Fraud_Data_features.csv', index=False)
        
        # Credit Card (Copy cleaned to features stage)
        credit_df = pd.read_csv(DATA_DIR / 'processed/creditcard_cleaned.csv')
        credit_df.to_csv(DATA_DIR / 'processed/creditcard_features.csv', index=False)
        
        print("✓ Feature engineering complete.")
        
    except FileNotFoundError as e:
        print(f"Error: Required file missing for feature engineering. {e}")
    except Exception as e:
        print(f"Unexpected error in feature engineering: {e}")

if __name__ == "__main__":
    perform_feature_engineering()
