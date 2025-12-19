import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

def transform_fraud_data():
    print("Transforming Fraud_Data...")
    df = pd.read_csv('data/processed/Fraud_Data_features.csv')
    
    # Drop unnecessary columns
    cols_to_drop = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address']
    df = df.drop(cols_to_drop, axis=1)
    
    # Handle Categorical
    cat_cols = ['source', 'browser', 'sex', 'country']
    # For country, we might have many categories. Let's keep top N or just OHE if manageable.
    # Total rows are ~150k. OHE might be large but manageable.
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Separate features and target
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    num_cols = ['purchase_value', 'age', 'time_since_signup', 'hour_of_day', 'day_of_week', 'user_id_count', 'device_id_count']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # Handle Class Imbalance (SMOTE on training only)
    print(f"Class distribution before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Class distribution after SMOTE: {y_train_res.value_counts().to_dict()}")
    
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
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale Time and Amount
    scaler = StandardScaler()
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    
    # Handle Class Imbalance (SMOTE on training only)
    print(f"Class distribution before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Class distribution after SMOTE: {y_train_res.value_counts().to_dict()}")
    
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
