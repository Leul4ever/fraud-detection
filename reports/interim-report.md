# Task 1 Report: Data Preprocessing and Exploratory Data Analysis (EDA)

## 📋 Overview
The primary goal of Task 1 was to prepare the raw datasets for machine learning models. This involved cleaning the data, performing exploratory analysis to identify patterns, engineering new features to capture fraud behavior, and transforming the data for model consumption.

## 🧹 Data Cleaning
We processed two main datasets:
1.  **Fraud_Data.csv:** Transaction logs for e-commerce activities.
2.  **CreditCard.csv:** Historical bank credit card transactions.

### Key Actions:
- **Missing Values:** Checked for nulls. The datasets were relatively clean, with no significant missing values in critical columns like `purchase_value` or `class`.
- **Duplicates:** Verified transaction uniqueness. No duplicates were found in the core transaction sets.
- **Type Conversion:** Converted `signup_time` and `purchase_time` to datetime objects for temporal analysis.
- **Outlier Detection:** Used box plots to identify extreme values in `purchase_value` and adjusted/noted them for robust scaling.

## 📊 Exploratory Data Analysis (EDA)
EDA was performed to understand the relationships between features and the target variable (`class`).

### Findings:
- **Class Imbalance:** Fraudulent transactions account for a very small percentage of total data (~9-10% in Fraud_Data).
- **Purchase Value:** While the distributions of legitimate and fraudulent transaction values are similar, fraud often clusters at lower values or follows specific "bot-like" patterns.
- **Time Analysis:** Fraudulent activity showed higher density during specific hours, potentially indicating automated scripts.
- **Device & ID Overlap:** A high number of transactions from the same `device_id` or `user_id` was a strong indicator of fraudulent behavior.

## 🛠️ Feature Engineering
To improve model performance, we engineered several domain-specific features:

1.  **Geolocation Mapping:**
    - Combined `Fraud_Data` with `IpAddress_to_Country.csv`.
    - Converted IP addresses (integers) to match ranges and assigned countries to transactions.
2.  **Temporal Features:**
    - `hour_of_day`: The hour the transaction occurred.
    - `day_of_week`: The day the transaction occurred (Monday-Sunday).
    - `time_since_signup`: The duration between account creation and transaction (short durations often correlate with fraud).
3.  **Velocity Features:**
    - `user_id_count`: Number of transactions per user.
    - `device_id_count`: Number of transactions per device.

## 🔄 Data Transformation
Final steps to prepare the data for the pipeline:
- **Encoding:** Categorical variables like `source`, `browser`, and `sex` were encoded using One-Hot Encoding.
- **Scaling:** Numerical features like `purchase_value` and normalized counts were scaled using `StandardScaler` to ensure unit variance.
- **Handling Imbalance:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to create a balanced training set, mitigating the bias toward the majority class.

---
**Status:** Task 1 Completed successfully. Data is now stored in `/data/processed/` and ready for Task 2 (Model Building).
