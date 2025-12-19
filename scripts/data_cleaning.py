import pandas as pd
import numpy as np

def clean_fraud_data(df):
    print("Cleaning Fraud_Data...")
    # 1. Handle missing values
    print(f"Missing values before:\n{df.isnull().sum()}")
    # No missing values expected based on typical dataset, but handle if any
    df = df.dropna() 
    
    # 2. Remove duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicates found: {duplicates}")
    df = df.drop_duplicates()
    
    # 3. Correct data types
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    print("Fraud_Data cleaning complete.")
    return df

def clean_credit_card_data(df):
    print("Cleaning creditcard data...")
    # 1. Handle missing values
    print(f"Missing values before:\n{df.isnull().sum()}")
    df = df.dropna()
    
    # 2. Remove duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicates found: {duplicates}")
    df = df.drop_duplicates()
    
    # creditcard.csv is mostly numerical and PCA transformed, so types should be fine
    print("creditcard data cleaning complete.")
    return df

if __name__ == "__main__":
    # Load data
    fraud_df = pd.read_csv('data/raw/Fraud_Data.csv')
    credit_df = pd.read_csv('data/raw/creditcard.csv')
    
    # Ensure processed directory exists
    from pathlib import Path
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Clean
    fraud_df_cleaned = clean_fraud_data(fraud_df)
    credit_df_cleaned = clean_credit_card_data(credit_df)
    
    # Save processed data
    fraud_df_cleaned.to_csv('data/processed/Fraud_Data_cleaned.csv', index=False)
    credit_df_cleaned.to_csv('data/processed/creditcard_cleaned.csv', index=False)
    print("Processed data saved to data/processed/")
