import pandas as pd
import numpy as np
from typing import Optional

def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning on the Fraud_Data e-commerce dataset.
    
    Args:
        df (pd.DataFrame): The raw fraud dataset.
        
    Returns:
        pd.DataFrame: The cleaned dataset.
        
    Raises:
        ValueError: If the input is not a pandas DataFrame.
        KeyError: If required timestamp columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    print("Cleaning Fraud_Data...")
    
    # 1. Handle missing values
    try:
        missing = df.isnull().sum().sum()
        if missing > 0:
            print(f"Dropping {missing} missing values.")
            df = df.dropna()
    except Exception as e:
        print(f"Error handling missing values: {e}")
    
    # 2. Remove duplicates
    try:
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicates.")
            df = df.drop_duplicates()
    except Exception as e:
        print(f"Error handling duplicates: {e}")
    
    # 3. Correct data types
    required_cols = ['signup_time', 'purchase_time']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' is missing from the dataset.")
    
    try:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    except Exception as e:
        print(f"Error converting timestamps: {e}")
        raise
    
    print("Fraud_Data cleaning complete.")
    return df

def clean_credit_card_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning on the creditcard banking dataset.
    
    Args:
        df (pd.DataFrame): The raw credit card dataset.
        
    Returns:
        pd.DataFrame: The cleaned dataset.
        
    Raises:
        ValueError: If the input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
        
    print("Cleaning creditcard data...")
    
    # 1. Handle missing values
    df = df.dropna()
    
    # 2. Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Removing {duplicates} duplicates.")
        df = df.drop_duplicates()
    
    print("creditcard data cleaning complete.")
    return df

if __name__ == "__main__":
    from pathlib import Path
    
    # Paths
    RAW_DIR = Path('data/raw')
    PROCESSED_DIR = Path('data/processed')
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Loading data...")
        fraud_df = pd.read_csv(RAW_DIR / 'Fraud_Data.csv')
        credit_df = pd.read_csv(RAW_DIR / 'creditcard.csv')
        
        # Clean
        fraud_df_cleaned = clean_fraud_data(fraud_df)
        credit_df_cleaned = clean_credit_card_data(credit_df)
        
        # Save
        fraud_df_cleaned.to_csv(PROCESSED_DIR / 'Fraud_Data_cleaned.csv', index=False)
        credit_df_cleaned.to_csv(PROCESSED_DIR / 'creditcard_cleaned.csv', index=False)
        print(f"✓ Processed data saved to {PROCESSED_DIR}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find raw data files. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
