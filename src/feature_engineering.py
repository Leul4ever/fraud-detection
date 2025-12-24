import pandas as pd
import numpy as np

def perform_feature_engineering():
    print("Loading data for feature engineering...")
    fraud_df = pd.read_csv('data/processed/Fraud_Data_cleaned.csv')
    ip_df = pd.read_csv('data/raw/IpAddress_to_Country.csv')
    
    # 1. Geolocation Integration
    print("Integrating geolocation...")
    # Convert IP to int64 to ensure matching types
    fraud_df['ip_address'] = fraud_df['ip_address'].astype(np.int64)
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(np.int64)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(np.int64)
    
    # Sort for merge_asof
    fraud_df = fraud_df.sort_values('ip_address')
    ip_df = ip_df.sort_values('lower_bound_ip_address')
    
    # Range lookup using merge_asof (Requirement: IP-to-Country Join)
    # This precisely handles the logic: lower_bound <= ip_address <= upper_bound
    df_merged = pd.merge_asof(
        fraud_df, 
        ip_df, 
        left_on='ip_address', 
        right_on='lower_bound_ip_address'
    )
    
    # Validation step to ensure IP falls within defined range
    df_merged['country'] = np.where(
        df_merged['ip_address'] <= df_merged['upper_bound_ip_address'],
        df_merged['country'],
        'Unknown'
    )
    
    # Drop intermediate columns
    df_merged = df_merged.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1)
    
    # 2. Feature Engineering (Requirement: Time and Frequency Features)
    print("Engineering features for Fraud_Data...")
    df_merged['signup_time'] = pd.to_datetime(df_merged['signup_time'])
    df_merged['purchase_time'] = pd.to_datetime(df_merged['purchase_time'])
    
    # Requirement: time_since_signup (Duration between signup and purchase)
    df_merged['time_since_signup'] = (df_merged['purchase_time'] - df_merged['signup_time']).dt.total_seconds()
    
    # Requirement: Time-based features (hour_of_day, day_of_week)
    df_merged['hour_of_day'] = df_merged['purchase_time'].dt.hour
    df_merged['day_of_week'] = df_merged['purchase_time'].dt.dayofweek
    
    # Requirement: Transaction frequency and velocity (Count of transactions per user/device)
    df_merged['user_id_count'] = df_merged.groupby('user_id')['user_id'].transform('count')
    df_merged['device_id_count'] = df_merged.groupby('device_id')['device_id'].transform('count')
    
    # Save processed data
    print("Saving feature-engineered data...")
    df_merged.to_csv('data/processed/Fraud_Data_features.csv', index=False)
    
    # Credit Card Data doesn't need much feature engineering as per PCA, 
    # but we can copy it to the same stage for consistency if needed.
    credit_df = pd.read_csv('data/processed/creditcard_cleaned.csv')
    credit_df.to_csv('data/processed/creditcard_features.csv', index=False)
    
    print("Feature engineering complete.")

if __name__ == "__main__":
    perform_feature_engineering()
