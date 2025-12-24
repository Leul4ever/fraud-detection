import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
import joblib
from pathlib import Path

# Ensure plots and models directories exist
Path('reports/figures').mkdir(parents=True, exist_ok=True)
Path('models').mkdir(parents=True, exist_ok=True)

def evaluate_model(y_true, y_pred, y_probs, model_name, dataset_name):
    print(f"\n--- {model_name} Evaluation on {dataset_name} ---")
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name} ({dataset_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'reports/figures/cm_{model_name.lower().replace(" ", "_")}_{dataset_name.lower()}.png')
    plt.close()
    
    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    auc_pr = auc(recall, precision)
    print(f"AUC-PR: {auc_pr:.4f}")
    
    # F1-Score
    f1 = f1_score(y_true, y_pred)
    print(f"F1-Score: {f1:.4f}")
    
    return {"AUC-PR": auc_pr, "F1": f1}

def cross_validate_model(model, X, y, model_name, dataset_name):
    print(f"\nRunning 5-fold Stratified CV for {model_name} on {dataset_name}...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = ['precision', 'recall', 'f1', 'average_precision']
    cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring)
    
    results = {
        "Mean F1": np.mean(cv_results['test_f1']),
        "Std F1": np.std(cv_results['test_f1']),
        "Mean AUC-PR": np.mean(cv_results['test_average_precision']),
        "Std AUC-PR": np.std(cv_results['test_average_precision'])
    }
    
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
        
    return results

def tune_xgboost(X, y, dataset_name):
    print(f"\n--- Tuning XGBoost for {dataset_name} ---")
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    # Stratified K-Fold for inner loop of CV tuning
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Use AUC-PR (average_precision) as the scoring metric for tuning
    random_search = RandomizedSearchCV(
        xgb, param_distributions=param_dist, n_iter=5, 
        scoring='average_precision', cv=skf, verbose=1, random_state=42, n_jobs=-1
    )
    
    random_search.fit(X, y)
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV AUC-PR: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def run_modeling_task():
    datasets = {
        "Fraud_Data": {
            "X_train": 'data/processed/Fraud_X_train.csv',
            "X_test": 'data/processed/Fraud_X_test.csv',
            "y_train": 'data/processed/Fraud_y_train.csv',
            "y_test": 'data/processed/Fraud_y_test.csv'
        },
        "Credit_Card": {
            "X_train": 'data/processed/Credit_X_train.csv',
            "X_test": 'data/processed/Credit_X_test.csv',
            "y_train": 'data/processed/Credit_y_train.csv',
            "y_test": 'data/processed/Credit_y_test.csv'
        }
    }
    
    all_results = []
    
    for name, paths in datasets.items():
        print(f"\n{'='*30}\nProcessing {name} Dataset\n{'='*30}")
        
        # Load Data
        X_train = pd.read_csv(paths["X_train"])
        X_test = pd.read_csv(paths["X_test"])
        y_train = pd.read_csv(paths["y_train"]).values.ravel()
        y_test = pd.read_csv(paths["y_test"]).values.ravel()
        
        # 1. Baseline: Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        y_probs_lr = lr.predict_proba(X_test)[:, 1]
        
        lr_metrics = evaluate_model(y_test, y_pred_lr, y_probs_lr, "Logistic Regression", name)
        lr_cv = cross_validate_model(lr, X_train, y_train, "Logistic Regression", name)
        
        all_results.append({
            "Dataset": name, "Model": "Logistic Regression", 
            "Test AUC-PR": lr_metrics["AUC-PR"], "Test F1": lr_metrics["F1"],
            "CV AUC-PR Mean": lr_cv["Mean AUC-PR"], "CV AUC-PR Std": lr_cv["Std AUC-PR"],
            "CV F1 Mean": lr_cv["Mean F1"], "CV F1 Std": lr_cv["Std F1"]
        })
        
        # 2. Ensemble: Random Forest (Baseline Ensemble)
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        y_probs_rf = rf.predict_proba(X_test)[:, 1]
        
        rf_metrics = evaluate_model(y_test, y_pred_rf, y_probs_rf, "Random Forest", name)
        rf_cv = cross_validate_model(rf, X_train, y_train, "Random Forest", name)
        
        all_results.append({
            "Dataset": name, "Model": "Random Forest", 
            "Test AUC-PR": rf_metrics["AUC-PR"], "Test F1": rf_metrics["F1"],
            "CV AUC-PR Mean": rf_cv["Mean AUC-PR"], "CV AUC-PR Std": rf_cv["Std AUC-PR"],
            "CV F1 Mean": rf_cv["Mean F1"], "CV F1 Std": rf_cv["Std F1"]
        })

        # 3. Tuned XGBoost
        # Requirement: Add explicit code for hyperparameter tuning
        xgb_tuned = tune_xgboost(X_train, y_train, name)
        
        y_pred_xgb = xgb_tuned.predict(X_test)
        y_probs_xgb = xgb_tuned.predict_proba(X_test)[:, 1]
        
        xgb_metrics = evaluate_model(y_test, y_pred_xgb, y_probs_xgb, "Tuned XGBoost", name)
        xgb_cv = cross_validate_model(xgb_tuned, X_train, y_train, "Tuned XGBoost", name)
        
        all_results.append({
            "Dataset": name, "Model": "Tuned XGBoost", 
            "Test AUC-PR": xgb_metrics["AUC-PR"], "Test F1": xgb_metrics["F1"],
            "CV AUC-PR Mean": xgb_cv["Mean AUC-PR"], "CV AUC-PR Std": xgb_cv["Std AUC-PR"],
            "CV F1 Mean": xgb_cv["Mean F1"], "CV F1 Std": xgb_cv["Std F1"]
        })
        
        # Save best model
        joblib.dump(xgb_tuned, f'models/best_model_{name.lower().replace(" ", "_")}.pkl')
        print(f"Best model for {name} saved.")

    # Final Comparison
    results_df = pd.DataFrame(all_results)
    print("\n--- Final Model Comparison ---")
    print(results_df.to_string(index=False))
    results_df.to_csv('reports/model_comparison_results.csv', index=False)

if __name__ == "__main__":
    run_modeling_task()
