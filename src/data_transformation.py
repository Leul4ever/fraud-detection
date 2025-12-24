import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
from typing import Tuple, List

def process_fraud_transformation(df: pd.DataFrame) -> None:
    """
    Handles encoding, scaling, and SMOTE for the e-commerce fraud dataset.
    
    Args:
        df (pd.DataFrame): Feature-engineered fraud data.
    """
    print("Transforming Fraud_Data...")
    
    # 1. Drop metadata
    cols_to_drop = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address']
    df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)
    
    # 2. Categorical Encoding
    cat_cols = ['source', 'browser', 'sex', 'country']
    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True)
    
    # 3. Separate Features/Target
    if 'class' not in df.columns:
        raise KeyError("Target column 'class' missing from dataset.")
    X = df.drop('class', axis=1).astype(float)
    y = df['class']
    
    # 4. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Scaling
    scaler = StandardScaler()
    num_cols = ['purchase_value', 'age', 'time_since_signup', 'hour_of_day', 
                'day_of_week', 'user_id_count', 'device_id_count']
    num_cols = [c for c in num_cols if c in X_train.columns]
    
    X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])
    joblib.dump(scaler, 'models/fraud_scaler.pkl')
    
    # 6. SMOTE (Training only)
    print(f"Distribution before SMOTE: {y_train.value_counts().to_dict()}")
    X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    print(f"Distribution after SMOTE: {y_train_res.value_counts().to_dict()}")
    
    # 7. Save
    X_train_res.to_csv('data/processed/Fraud_X_train.csv', index=False)
    X_test.to_csv('data/processed/Fraud_X_test.csv', index=False)
    y_train_res.to_csv('data/processed/Fraud_y_train.csv', index=False)
    y_test.to_csv('data/processed/Fraud_y_test.csv', index=False)

def process_credit_transformation(df: pd.DataFrame) -> None:
    """
    Handles scaling and SMOTE for the bank credit card dataset.
    
    Args:
        df (pd.DataFrame): Cleaned credit card data.
    """
    print("Transforming Credit Card Data...")
    
    if 'Class' not in df.columns:
        raise KeyError("Target column 'Class' missing from dataset.")
        
    X = df.drop('Class', axis=1).astype(float)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale Time and Amount
    scaler = StandardScaler()
    X_train.loc[:, ['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test.loc[:, ['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    joblib.dump(scaler, 'models/credit_scaler.pkl')
    
    # SMOTE
    X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    
    # Save
    X_train_res.to_csv('data/processed/Credit_X_train.csv', index=False)
    X_test.to_csv('data/processed/Credit_X_test.csv', index=False)
    y_train_res.to_csv('data/processed/Credit_y_train.csv', index=False)
    y_test.to_csv('data/processed/Credit_y_test.csv', index=False)

if __name__ == "__main__":
    Path('models').mkdir(parents=True, exist_ok=True)
    try:
        fraud_data = pd.read_csv('data/processed/Fraud_Data_features.csv')
        process_fraud_transformation(fraud_data)
        
        credit_data = pd.read_csv('data/processed/creditcard_features.csv')
        process_credit_transformation(credit_data)
        print("✓ Data transformation complete.")
    except Exception as e:
        print(f"Error in transformation: {e}")
