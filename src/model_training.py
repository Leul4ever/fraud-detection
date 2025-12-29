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
from typing import Dict, Any, List, Optional

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, 
                   model_name: str, dataset_name: str) -> Dict[str, float]:
    """
    Evaluates a model and saves the confusion matrix plot.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted class labels.
        y_probs: Predicted fraud probabilities.
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        
    Returns:
        Dict containing AUC-PR and F1 metrics.
    """
    print(f"\n--- {model_name} Evaluation on {dataset_name} ---")
    
    # 1. Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'CM: {model_name} ({dataset_name})')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    
    out_path = Path('reports/figures')
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / f'cm_{model_name.lower().replace(" ", "_")}_{dataset_name.lower()}.png')
    plt.close()
    
    # 3. Metrics
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    auc_pr = auc(recall, precision)
    f1 = f1_score(y_true, y_pred)
    
    print(f"AUC-PR: {auc_pr:.4f} | F1: {f1:.4f}")
    return {"AUC-PR": auc_pr, "F1": f1}

def cross_validate_model(model: Any, X: pd.DataFrame, y: np.ndarray, 
                         model_name: str, dataset_name: str) -> Dict[str, float]:
    """
    Runs 5-fold Stratified CV and reports mean/std of metrics.
    
    Args:
        model: Estimator to evaluate.
        X: Features.
        y: Labels.
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        
    Returns:
        Dict with mean/std for F1 and AUC-PR.
    """
    print(f"Running 5-fold Stratified CV for {model_name} on {dataset_name}...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = ['precision', 'recall', 'f1', 'average_precision']
    try:
        cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    except Exception as e:
        print(f"Error during Cross-Validation: {e}")
        raise

    results = {
        "Mean F1": np.mean(cv_results['test_f1']),
        "Std F1": np.std(cv_results['test_f1']),
        "Mean AUC-PR": np.mean(cv_results['test_average_precision']),
        "Std AUC-PR": np.std(cv_results['test_average_precision'])
    }
    
    print(f"  Mean AUC-PR: {results['Mean AUC-PR']:.4f} (\u00b1{results['Std AUC-PR']:.4f})")
    return results

def tune_xgboost(X: pd.DataFrame, y: np.ndarray, dataset_name: str) -> Any:
    """Performs RandomizedSearchCV for XGBoost."""
    print(f"Tuning XGBoost for {dataset_name}...")
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    search = RandomizedSearchCV(
        xgb, param_distributions=param_dist, n_iter=5, 
        scoring='average_precision', cv=3, verbose=0, random_state=42, n_jobs=-1
    )
    
    search.fit(X, y)
    print(f"Best parameters: {search.best_params_}")
    return search.best_estimator_

def run_modeling_task() -> None:
    """Main execution loop for model training and comparison."""
    DATA_MAP = {
        "Fraud_Data": "Fraud",
        "Credit_Card": "Credit"
    }
    
    all_results = []
    
    for display_name, file_prefix in DATA_MAP.items():
        print(f"\n{'='*40}\nProcessing {display_name}\n{'='*40}")
        
        try:
            # Load
            X_train = pd.read_csv(f'data/processed/{file_prefix}_X_train.csv')
            X_test = pd.read_csv(f'data/processed/{file_prefix}_X_test.csv')
            y_train = pd.read_csv(f'data/processed/{file_prefix}_y_train.csv').values.ravel()
            y_test = pd.read_csv(f'data/processed/{file_prefix}_y_test.csv').values.ravel()
            
            # Models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                "Tuned XGBoost": tune_xgboost(X_train, y_train, display_name)
            }
            
            for m_name, model in models.items():
                print(f"Training {m_name}...")
                model.fit(X_train, y_train)
                
                # Eval
                m = evaluate_model(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1], m_name, display_name)
                cv = cross_validate_model(model, X_train, y_train, m_name, display_name)
                
                all_results.append({
                    "Dataset": display_name, "Model": m_name, 
                    "Test AUC-PR": m["AUC-PR"], "Test F1": m["F1"],
                    "CV AUC-PR Mean": cv["Mean AUC-PR"], "CV AUC-PR Std": cv["Std AUC-PR"],
                    "CV F1 Mean": cv["Mean F1"], "CV F1 Std": cv["Std F1"]
                })
            
            # Save Best
            best_r = max([r for r in all_results if r['Dataset'] == display_name], key=lambda x: x['CV AUC-PR Mean'])
            joblib.dump(models[best_r['Model']], f'models/best_model_{display_name.lower().replace(" ", "_")}.pkl')
            
        except FileNotFoundError as e:
            print(f"Error: Missing data files for {display_name}. {e}")
        except Exception as e:
            print(f"Unexpected error training {display_name}: {e}")

    # Results table
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('reports/model_comparison_results.csv', index=False)
        print("\n--- Model Comparison Summary ---")
        print(results_df[['Dataset', 'Model', 'CV AUC-PR Mean', 'CV F1 Mean']].to_string(index=False))

if __name__ == "__main__":
    run_modeling_task()
