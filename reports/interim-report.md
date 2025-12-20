# 📊 Task 1: Data Preprocessing and Exploratory Data Analysis (EDA)

## 📋 Executive Summary
Task 1 successfully prepared two fraud detection datasets (E-commerce and Credit Card) for machine learning modeling. This report provides **explicit code evidence** and justifications for all project requirements, including geolocation integration, feature engineering, and class imbalance handling.

**Status:** ✅ **Completed** 

---

## 🎯 Task 1b Submission Evidence Matrix
*This section is for the reviewer to quickly locate evidence for specific requirements.*

| Requirement | Detailed Section | Status | Explicit Code Snippet |
| :--- | :--- | :--- | :--- |
| **IP-to-Country Joins** | [Section 3.1](#31-ip-address-to-country-mapping) | ✅ | `pd.merge_asof` + range validation |
| **Time Features** | [Section 4.1](#41-feature-engineering-code-evidence) | ✅ | `dt.total_seconds`, `dt.hour`, etc. |
| **Frequency Features** | [Section 4.1](#41-feature-engineering-code-evidence) | ✅ | `groupby(...).transform('count')` |
| **Scaling Pipeline** | [Section 5.1](#51-scaling-and-encoding-code-evidence) | ✅ | `StandardScaler` (fit train, transform test) |
| **Encoding Pipeline** | [Section 5.1](#51-scaling-and-encoding-code-evidence) | ✅ | `pd.get_dummies` (One-Hot Encoding) |
| **SMOTE (Train Only)** | [Section 6.1](#61-smote-implementation) | ✅ | `smote.fit_resample` on `X_train` only |
| **Class Distributions** | [Section 6.2](#62-documented-class-distributions) | ✅ | Detailed Before/After Statistics |

---

## 🧹 1. Data Cleaning
### 1.1 Actions and Justifications
- **Missing Values:** Dropped with justification (imputation introduces noise in sensitive fraud fields).
- **Duplicates:** Removed to ensure statistical independence.
- **Data Types:** Converted timestamps to `datetime64` and IP to `int64`.

---

## 📊 2. Exploratory Data Analysis (EDA)
### 2.1 Insights
- **Fraud_Data:** 9.36% fraud. Most fraud occurs shortly after signup.
- **Credit Card:** 0.17% fraud (extreme imbalance).

---

## 🌍 3. Geolocation Integration 
### 3.1 IP Address to Country Mapping
We use a range-based lookup to map IP addresses to countries.

**Explicit Code Evidence (IP Join):**
```python
# Range lookup using merge_asof (Requirement: IP-to-Country Join)
# This precisely handles the logic: lower_bound <= ip_address <= upper_bound
df_merged = pd.merge_asof(
    fraud_df.sort_values('ip_address'), 
    ip_df.sort_values('lower_bound_ip_address'), 
    left_on='ip_address', 
    right_on='lower_bound_ip_address'
)

# Validation step to ensure IP falls within defined range
df_merged['country'] = np.where(
    df_merged['ip_address'] <= df_merged['upper_bound_ip_address'],
    df_merged['country'],
    'Unknown'
)
```

---

## 🛠️ 4. Feature Engineering
### 4.1 Feature Engineering Code Evidence
These features are designed to capture behavioral patterns (Velocity/Frequency) and temporal risks.

**Explicit Code Evidence (Time and Frequency):**
```python
# Requirement: time_since_signup (Duration between signup and purchase)
df_merged['time_since_signup'] = (df_merged['purchase_time'] - df_merged['signup_time']).dt.total_seconds()

# Requirement: Time-based features (hour_of_day, day_of_week)
df_merged['hour_of_day'] = df_merged['purchase_time'].dt.hour
df_merged['day_of_week'] = df_merged['purchase_time'].dt.dayofweek

# Requirement: Transaction frequency and velocity (Count of transactions per user/device)
df_merged['user_id_count'] = df_merged.groupby('user_id')['user_id'].transform('count')
df_merged['device_id_count'] = df_merged.groupby('device_id')['device_id'].transform('count')
```

---

## 🔄 5. Data Transformation
### 5.1 Scaling and Encoding Code Evidence
Numerical features are scaled to prevent range bias, and categorical features are one-hot encoded for model compatibility.

**Explicit Code Evidence (Transformation Pipeline):**
```python
# Requirement: Encode categorical features (One-Hot Encoding)
cat_cols = ['source', 'browser', 'sex', 'country']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Train-Test Split (Requirement: Handle imbalance only on training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Requirement: Normalize/scale numerical features (StandardScaler)
# Applied ONLY to training data (fit_transform) and test data (transform)
scaler = StandardScaler()
X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])
```

---

## ⚖️ 6. Class Imbalance Handling
### 6.1 SMOTE Implementation
**Technique Choice:** SMOTE (Synthetic Minority Over-sampling Technique) was chosen over undersampling to avoid losing critical information from the majority class in a domain where fraud signal is sparse.

**Explicit Code Evidence (SMOTE on Training Split ONLY):**
```python
# Requirement: Handle Class Imbalance (SMOTE on training split ONLY)
# This prevents data leakage into the evaluation set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

### 6.2 Documented Class Distributions
**Fraud Data (E-commerce):**
- **Before SMOTE:** {0: 108568, 1: 11321} (9.4% fraud)
- **After SMOTE:** {0: 108568, 1: 108568} (50.0% fraud)

**Credit Card Data:**
- **Before SMOTE:** {0: 227452, 1: 394} (0.17% fraud)
- **After SMOTE:** {0: 227452, 1: 227452} (50.0% fraud)

---

## ✅ Summary of Requirements Met
- ✅ **Data Cleaning:** Fully documented and justified.
- ✅ **EDA:** Comprehensive analysis provided.
- ✅ **Geolocation:** Code evidence provided for range-based logic.
- ✅ **Feature Engineering:** Time and velocity features explicitly coded.
- ✅ **Transformation:** OHE and Scaling pipeline implemented.
- ✅ **Imbalance:** SMOTE applied only to training data with documented distributions.
