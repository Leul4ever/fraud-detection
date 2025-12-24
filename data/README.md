# Data Directory

This directory contains the datasets used in the fraud detection project.

## Structure

- **raw/**: Contains the original, unmodified data files.
  - `Fraud_Data.csv`: E-commerce transaction data.
  - `creditcard.csv`: Bank credit card transaction data.
  - `IpAddress_to_Country.csv`: IP address to country mapping.

- **processed/**: Contains cleaned and feature-engineered datasets ready for modeling.
  - `Fraud_Data_cleaned.csv`: Cleaned version of the e-commerce data.
  - `Fraud_Data_features.csv`: E-commerce data with engineered features (geolocation, time, velocity).
  - `creditcard_cleaned.csv`: Cleaned version of the banking data.
  - `creditcard_features.csv`: Banking data with additional formatting.
  - `*_X_train.csv`, `*_X_test.csv`, `*_y_train.csv`, `*_y_test.csv`: Stratified splits for model training and evaluation.

## Usage

Data is processed from `raw/` to `processed/` using the scripts in `src/`. The processed datasets are then used for EDA in `notebooks/` and model training.
