# 🛡️ Fraud Detection for E-commerce & Banking 💳

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/Leul4ever/fraud-detection/workflows/Unit%20Tests/badge.svg)](https://github.com/Leul4ever/fraud-detection/actions)

## 💡 Project Idea

In the rapidly evolving digital landscape, fraudulent transactions pose a significant threat to financial security and user trust. This project aims to build a robust **Fraud Detection System** that analyzes transaction patterns in e-commerce and banking data to identify and prevent malicious activities.

## 🚀 Project Context

This project is part of a comprehensive data science workflow aimed at:
1. **Detecting patterns** associated with fraudulent activities.
2. **Bridging geolocation data** with transaction logs.
3. **Handling class imbalance** (since fraud cases are rare).
4. **Developing real-time detection** APIs and interactive dashboards.

## 📁 Project Structure

Following the project requirements, the repository is organized as follows:

```
fraud-detection/
├── .vscode/
│   └── settings.json             # Workspace settings
├── .github/
│   └── workflows/
│       └── unittests.yml         # CI/CD pipeline
├── data/                         # Project datasets (ignored except documentation)
│   ├── raw/                      # Original, immutable datasets
│   └── processed/                # Cleaned and feature-engineered data
├── notebooks/
│   ├── __init__.py
│   ├── eda-fraud-data.ipynb       # EDA for e-commerce data
│   ├── eda-creditcard.ipynb       # EDA for bank credit data
│   ├── feature-engineering.ipynb  # Feature engineering logic
│   ├── modeling.ipynb              # Model building and evaluation
│   ├── shap-explainability.ipynb  # Model interpretability
│   └── README.md                  # Notebooks documentation
├── src/                          # Core production modules
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_transformation.py
│   └── model_training.py
├── tests/                        # Automated unit tests
│   ├── __init__.py
│   └── ...
├── models/                       # Saved model artifacts (.pkl files)
├── scripts/                      # Runner scripts
│   ├── __init__.py
│   └── README.md
├── requirements.txt              # Project dependencies
├── README.md                     # Main documentation
└── .gitignore                    # Git ignore rules
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
- **Data Cleaning**: Missing values, duplicates, and type corrections.
- **EDA**: Univariate/Bivariate analysis and class distribution.
- **Geolocation**: Mapping IP addresses to countries.
- **Feature Engineering**: Frequency, velocity, and time-based features.
- **Transformation**: Scaling and SMOTE for imbalance handling.

### ✅ Task 2: Model Building & Training (Completed)
- **Baseline**: Logistic Regression (AUC-PR, F1-Score).
- **Ensemble**: Random Forest & Tuned XGBoost.
- **Stability**: 5-fold Stratified Cross-Validation.
- **Selection**: XGBoost chosen for production based on AUC-PR.

### 📋 Task 3: Model Explainability (Planned)
- SHAP global and local feature importance.

### 🚀 Task 4: Model Deployment (Planned)
- REST API serving with Flask/FastAPI.

### 📊 Task 5: Interactive Dashboard (Planned)
- Monitoring dashboard with Streamlit/Dash.

## 🧰 Tech Stack
- **Data**: Pandas, NumPy
- **ML**: Scikit-learn, XGBoost, Imbalanced-learn
- **Viz**: Matplotlib, Seaborn
- **Testing**: Pytest & GitHub Actions

## 🤝 Contributing
Part of the **Kifiya AI Mentorship Program**.

## 📄 License
MIT License

## 👤 Author
**Leul** - [GitHub](https://github.com/Leul4ever)

---
**Last Updated:** Task 2 Completed ✅  
**Next Milestone:** Task 3 - Model Explainability
