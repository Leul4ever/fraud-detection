# 🛡️ Fraud Detection for E-commerce & Banking 💳

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/Leul4ever/fraud-detection/workflows/Unit%20Tests/badge.svg)](https://github.com/Leul4ever/fraud-detection/actions)

## 💡 Project Idea

In the rapidly evolving digital landscape, fraudulent transactions pose a significant threat to financial security and user trust. This project aims to build a robust **Fraud Detection System** that analyzes transaction patterns in e-commerce and banking data to identify and prevent malicious activities.

By leveraging advanced machine learning techniques, we seek to distinguish legitimate users from bad actors, ensuring safer transactions for both consumers and businesses.

## 🚀 Project Context

This project is part of a comprehensive data science workflow aimed at:
1. **Detecting patterns** associated with fraudulent activities.
2. **Bridging geolocation data** with transaction logs.
3. **Handling class imbalance** (since fraud cases are rare).
4. **Developing real-time detection** APIs and interactive dashboards.

## 📁 Complete Project Structure

```
fraud-detection/
├── .github/
│   └── workflows/
│       └── unittests.yml          # CI/CD pipeline for automated testing
│
├── data/
│   ├── raw/                        # Raw data files (not in git)
│   │   ├── Fraud_Data.csv         # E-commerce transaction data
│   │   ├── creditcard.csv         # Bank credit card transaction data
│   │   └── IpAddress_to_Country.csv  # IP to country mapping
│   │
│   └── processed/                  # Processed data files (not in git)
│       ├── Fraud_Data_cleaned.csv
│       ├── Fraud_Data_features.csv
│       ├── creditcard_cleaned.csv
│       ├── creditcard_features.csv
│       ├── Fraud_X_train.csv, Fraud_X_test.csv
│       ├── Fraud_y_train.csv, Fraud_y_test.csv
│       ├── Credit_X_train.csv, Credit_X_test.csv
│       └── Credit_y_train.csv, Credit_y_test.csv
│
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── eda-fraud-data.ipynb       # EDA for e-commerce fraud data
│   ├── eda-creditcard.ipynb       # EDA for credit card fraud data
│   ├── feature-engineering.ipynb  # Feature engineering process
│   ├── data-transformation.ipynb  # Data transformation and SMOTE
│   ├── modeling.ipynb              # Model building (Future)
│   ├── shap-explainability.ipynb  # Model explainability (Future)
│   └── README.md                  # Notebooks documentation
│
├── scripts/                        # Production-ready Python modules
│   ├── data_cleaning.py           # Data cleaning functions
│   ├── feature_engineering.py     # Feature engineering pipeline
│   ├── data_transformation.py     # Scaling, encoding, SMOTE
│   ├── run_data_pipeline.py       # Complete data pipeline runner
│   ├── create_test_data.py        # Test data generation
│   └── README.md                  # Scripts documentation
│
├── src/                            # Source code (for future API/dashboard)
│   └── __init__.py
│
├── models/                         # Trained models and scalers
│   ├── best_model_fraud_data.pkl  # Final XGBoost model for fraud
│   ├── best_model_credit_card.pkl # Final XGBoost model for credit card
│   ├── fraud_scaler.pkl           # StandardScaler for fraud data
│   └── credit_scaler.pkl           # StandardScaler for credit card data
│
├── reports/                        # Analysis reports and visualizations
│   ├── figures/                    # Generated plots and charts
│   │   ├── fraud_class_distribution.png
│   │   ├── fraud_univariate_analysis.png
│   │   ├── fraud_purchase_value_vs_class.png
│   │   ├── fraud_rate_by_country.png
│   │   ├── fraud_smote_comparison.png
│   │   ├── creditcard_class_distribution.png
│   │   ├── creditcard_amount_distribution.png
│   │   ├── creditcard_bivariate_analysis.png
│   │   ├── creditcard_feature_engineering.png
│   │   └── creditcard_smote_comparison.png
│   ├── interim-report.md          # Task 1 comprehensive report
│   ├── task-2.md                  # Task 2 comprehensive report
│   └── model_comparison_results.csv # Metrics for all models
│
├── tests/                          # Unit and integration tests
│   ├── conftest.py                # Pytest fixtures and configuration
│   ├── test_data_cleaning.py      # Data cleaning tests
│   ├── test_feature_engineering.py # Feature engineering tests
│   └── test_data_transformation.py # Data transformation tests
│
├── venv/                           # Virtual environment (not in git)
│
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+ (tested with Python 3.13)
- Git

### Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Leul4ever/fraud-detection.git
   cd fraud-detection
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Data Pipeline:**
   ```bash
   python scripts/run_data_pipeline.py
   ```

5. **Run Tests:**
   ```bash
   pytest tests/ -v
   ```

## 📈 Roadmap (Tasks)

### ✅ Task 1: Data Preprocessing & EDA (Completed)

**Objective:** Prepare clean, feature-rich datasets ready for modeling.

**Completed Components:**

- **Data Cleaning:**
  - ✅ Missing value handling with justification
  - ✅ Duplicate removal
  - ✅ Data type corrections (datetime, integer conversions)

- **Exploratory Data Analysis:**
  - ✅ Univariate analysis (distributions of key variables)
  - ✅ Bivariate analysis (relationships between features and target)
  - ✅ Class distribution analysis (quantified imbalance)

- **Geolocation Integration (Fraud_Data.csv):**
  - ✅ IP addresses converted to integer format
  - ✅ Range-based merge with IpAddress_to_Country.csv
  - ✅ Fraud patterns analyzed by country

- **Feature Engineering (Fraud_Data.csv):**
  - ✅ Transaction frequency features (user_id_count, device_id_count)
  - ✅ Time-based features (hour_of_day, day_of_week, time_since_signup)
  - ✅ Velocity features (user_transaction_velocity)

- **Data Transformation:**
  - ✅ StandardScaler for numerical features
  - ✅ OneHotEncoder for categorical features

- **Class Imbalance Handling:**
  - ✅ SMOTE applied to training data only
  - ✅ Justification documented
  - ✅ Class distribution before/after documented

**Deliverables:**
- Clean, processed datasets in `data/processed/`
- Comprehensive EDA notebooks with visualizations
- Feature-engineered datasets ready for modeling
- Detailed interim report with findings

**See:** [`reports/interim-report.md`](reports/interim-report.md) for complete Task 1 report

### ✅ Task 2: Model Building & Training (Completed)

**Objective:** Build, train, and evaluate classification models to detect fraudulent transactions.

**Completed Components:**
- **Baseline Modeling:**
  - ✅ Logistic Regression trained as interpretable baseline
- **Ensemble Modeling:**
  - ✅ Random Forest (n=100, depth=10)
  - ✅ XGBoost (n=100, depth=5, lr=0.1)
- **Robustness:**
  - ✅ 5-fold Stratified K-Fold Cross-Validation implemented
- **Model Selection:**
  - ✅ Side-by-side comparison of all models
  - ✅ XGBoost selected as final model with documented justification

**Evaluation Metrics:**
- ✅ **AUC-PR:** Primary metric for class imbalance
- ✅ **F1-Score:** Balanced performance measure
- ✅ **Confusion Matrix:** Prediction visualization

**See:** [`reports/task-2.md`](reports/task-2.md) for complete Task 2 report

### 📋 Task 3: Model Explainability (Planned)

**Objective:** Interpret model predictions using SHAP to understand fraud detection drivers.

**Planned Components:**
- Feature importance analysis
- SHAP summary plots
- Individual prediction explanations
- Business recommendations

### 🚀 Task 4: Model Deployment (Planned)

**Objective:** Deploy fraud detection model as a REST API.

**Planned Components:**
- Flask/FastAPI implementation
- Model serving endpoint
- Request/response handling
- API documentation

### 📊 Task 5: Interactive Dashboard (Planned)

**Objective:** Create interactive dashboard for fraud detection monitoring.

**Planned Components:**
- Streamlit/Dash dashboard
- Real-time fraud detection
- Visualization of predictions
- Model performance metrics

## 🧰 Tech Stack

### Core Libraries
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, Imbalanced-learn
- **Testing:** Pytest
- **CI/CD:** GitHub Actions

### Future Additions
- **Explainability:** SHAP, LIME
- **API:** Flask/FastAPI
- **Dashboard:** Streamlit/Dash

## 📊 Dataset Overview

### Fraud_Data.csv (E-commerce)
- **Size:** ~151,000 transactions
- **Features:** 11 original + 6 engineered = 17 total
- **Class Distribution:** 90.64% legitimate, 9.36% fraud
- **Key Features:** purchase_value, age, source, browser, country, time_since_signup

### creditcard.csv (Banking)
- **Size:** ~284,000 transactions
- **Features:** 30 (Time, V1-V28, Amount, Class)
- **Class Distribution:** 99.83% legitimate, 0.17% fraud
- **Key Features:** Time, Amount, V1-V28 (PCA-transformed)

## 🧪 Testing

The project includes comprehensive unit tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_cleaning.py -v

# Run with coverage
pytest tests/ --cov=scripts --cov-report=html
```

**Test Coverage:**
- ✅ Data cleaning validation
- ✅ Feature engineering verification
- ✅ Data transformation checks
- ✅ File existence and structure validation

## 📝 Key Findings

### Class Imbalance
- **Fraud_Data:** 9.7:1 imbalance ratio (manageable with SMOTE)
- **creditcard:** 599:1 imbalance ratio (extreme, requires careful handling)

### Critical Fraud Indicators
1. **Time Since Signup:** Transactions within hours of signup show high fraud rates
2. **Transaction Frequency:** High-frequency users indicate bot activity
3. **Geographic Patterns:** Certain countries show elevated fraud rates
4. **Source/Browser:** Compromised channels identified
5. **Amount Patterns:** Fraud clusters at lower purchase values

## 🤝 Contributing

This project is part of the Kifiya AI Mentorship Program. For contributions, please follow the project guidelines and submit pull requests.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Leul** - [GitHub](https://github.com/Leul4ever)

Created as part of the **Kifiya AI Mentorship Program**.

---

**Last Updated:** Task 2 Completed ✅  
**Next Milestone:** Task 3 - Model Explainability
