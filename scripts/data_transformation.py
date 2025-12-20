import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

def transform_fraud_data():
    print("Transforming Fraud_Data...")
    df = pd.read_csv('data/processed/Fraud_Data_features.csv')
    
    # Ensure models directory exists
    from pathlib import Path
    Path('models').mkdir(parents=True, exist_ok=True)
    
    # Drop unnecessary columns
    cols_to_drop = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address']
    df = df.drop(cols_to_drop, axis=1)
    
    # Requirement: Encode categorical features (One-Hot Encoding)
    cat_cols = ['source', 'browser', 'sex', 'country']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Separate features and target
    X = df.drop('class', axis=1).astype(float)
    y = df['class']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Requirement: Normalize/scale numerical features (StandardScaler)
    # Applied ONLY to training data (fit_transform) and test data (transform)
    scaler = StandardScaler()
    num_cols = ['purchase_value', 'age', 'time_since_signup', 'hour_of_day', 'day_of_week', 'user_id_count', 'device_id_count']
    X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])
    
    # Requirement: Handle Class Imbalance (SMOTE on training split ONLY)
    print(f"--- Fraud Data distribution BEFORE SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"--- Fraud Data distribution AFTER SMOTE: {y_train_res.value_counts().to_dict()}")
    
    # Save processed subsets
    X_train_res.to_csv('data/processed/Fraud_X_train.csv', index=False)
    X_test.to_csv('data/processed/Fraud_X_test.csv', index=False)
    y_train_res.to_csv('data/processed/Fraud_y_train.csv', index=False)
    y_test.to_csv('data/processed/Fraud_y_test.csv', index=False)
    
    # Save scaler
    joblib.dump(scaler, 'models/fraud_scaler.pkl')
    print("Fraud_Data transformation complete.")

def transform_creditcard_data():
    print("Transforming Credit Card Data...")
    df = pd.read_csv('data/processed/creditcard_features.csv')
    
    # Features and target
    X = df.drop('Class', axis=1).astype(float)
    y = df['Class']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale Time and Amount
    scaler = StandardScaler()
    X_train.loc[:, ['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test.loc[:, ['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    
    # Requirement: Handle Class Imbalance (SMOTE on training split ONLY)
    print(f"--- Credit Card distribution BEFORE SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"--- Credit Card distribution AFTER SMOTE: {y_train_res.value_counts().to_dict()}")
    
    # Save
    X_train_res.to_csv('data/processed/Credit_X_train.csv', index=False)
    X_test.to_csv('data/processed/Credit_X_test.csv', index=False)
    y_train_res.to_csv('data/processed/Credit_y_train.csv', index=False)
    y_test.to_csv('data/processed/Credit_y_test.csv', index=False)
    
    joblib.dump(scaler, 'models/credit_scaler.pkl')
    print("Credit Card Data transformation complete.")

if __name__ == "__main__":
    transform_fraud_data()
    transform_creditcard_data()
