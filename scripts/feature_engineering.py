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
    
    # Range lookup using merge_asof
    # It merges where fraud_df['ip_address'] >= ip_df['lower_bound_ip_address']
    df_merged = pd.merge_asof(
        fraud_df, 
        ip_df, 
        left_on='ip_address', 
        right_on='lower_bound_ip_address'
    )
    
    # Validate the upper bound
    df_merged['country'] = np.where(
        df_merged['ip_address'] <= df_merged['upper_bound_ip_address'],
        df_merged['country'],
        'Unknown'
    )
    
    # Drop intermediate columns
    df_merged = df_merged.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1)
    
    # 2. Feature Engineering (Fraud_Data)
    print("Engineering features for Fraud_Data...")
    df_merged['signup_time'] = pd.to_datetime(df_merged['signup_time'])
    df_merged['purchase_time'] = pd.to_datetime(df_merged['purchase_time'])
    
    # time_since_signup
    df_merged['time_since_signup'] = (df_merged['purchase_time'] - df_merged['signup_time']).dt.total_seconds()
    
    # time-based features
    df_merged['hour_of_day'] = df_merged['purchase_time'].dt.hour
    df_merged['day_of_week'] = df_merged['purchase_time'].dt.dayofweek
    
    # Transaction frequency per user/device
    # Note: This is global frequency in the dataset
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
