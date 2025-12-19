# Interim-1 Report: Data Analysis and Preprocessing

## Task Completion Status
- [x] **Data Cleaning**: Handled missing values, duplicates, and type corrections.
- [x] **EDA**: Completed univariate, bivariate, and class distribution analysis.
- [x] **Geolocation**: Integrated IP-to-country mapping.
- [x] **Feature Engineering**: Engineered velocity and time-based features.
- [x] **Data Transformation**: Scaled features and encoded categorical variables.
- [x] **Class Imbalance**: Applied SMOTE to balance datasets.

---

## 1. Summary of Data Cleaning
- **Fraud_Data.csv**:
    - Checked for missing values: None found.
    - Checked for duplicates: None found.
    - Converted `signup_time` and `purchase_time` to datetime objects.
- **creditcard.csv**:
    - Checked for missing values: None found.
    - Identified and removed **1,081 duplicate transactions** to ensure data integrity.
    - PCA features (V1-V28) were already normalized and scaled.

## 2. Exploratory Data Analysis (EDA) Insights
- **Class Imbalance**:
    - **Fraud_Data**: ~9.3% of transactions are fraudulent.
    - ![Fraud Class Distribution](file:///d:/kifyaAi/fraud-detection/reports/figures/fraud_class_distribution.png)
- **Credit Card**: ~0.17% of transactions are fraudulent (extreme imbalance).
    - ![Credit Class Distribution](file:///d:/kifyaAi/fraud-detection/reports/figures/creditcard_class_distribution.png)

## 3. Key Discovery: Feature Visualization
- **Purchase Value vs Class**:
    - ![Purchase Value Boxplot](file:///d:/kifyaAi/fraud-detection/reports/figures/fraud_purchase_value_vs_class.png)
    - Fraudulent transactions in e-commerce often occur shortly after signup.
    - High-frequency transactions from the same `device_id` or `user_id` are strong indicators of fraud.

## 4. Geolocation and Country-Based Analysis
- **Merge Logic**: IP addresses were converted to `int64` and merged using a range lookup.
- **Fraud by Country**: High-volume countries like the United States exhibit stable fraud rates, while some regions show higher volatility or higher relative fraud percentages.
- ![Fraud Rate by Country](file:///d:/kifyaAi/fraud-detection/reports/figures/fraud_rate_by_country.png)

## 5. Feature Engineering
- **time_since_signup**: Calculated the duration between signup and purchase. Small values are highly correlated with fraud.
- **Time-based Features**: Extracted `hour_of_day` and `day_of_week` to identify peak fraud times.
- **Transaction Velocity**: Engineered `user_id_count` and `device_id_count` to capture transaction frequency.

## 6. Data Transformation and Class Imbalance Strategy
- **Categorical Encoding**: Applied One-Hot Encoding (OHE) to `source`, `browser`, and `sex`.
- **Scaling**: Used `StandardScaler` on numerical features.
- **Handling Imbalance**:
    - Used **SMOTE (Synthetic Minority Over-sampling Technique)** on the **training set only**.
    - **Fraud_Data**: Balanced from ~11k fraud to ~109k fraud.
    - **Credit Card**: Balanced from 378 fraud to ~226k fraud.

## Conclusion
Task 1 is complete. The datasets are now clean, feature-rich, and balanced, providing a solid foundation for model building in Task 2.
