import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import shap
from pathlib import Path

def load_data_and_model(model_path, x_test_path, y_test_path):
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()
    return model, X_test, y_test

def perform_explainability(model, X_test, y_test, output_dir):
    print("Generating BUILT-IN feature importance...")
    importance = model.feature_importances_
    indices = np.argsort(importance)[-10:]
    plt.figure(figsize=(10, 6))
    plt.title('Top 10 Feature Importance (Random Forest)')
    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [X_test.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance_rf.png"))
    plt.close()

    print("Calculating SHAP values using modern API...")
    explainer = shap.Explainer(model)
    # This returns an Explanation object which contains .values, .base_values, .data
    # For some tree models, Explainer(model) uses TreeExplainer internally.
    shap_explanation = explainer(X_test)
    
    print(f"SHAP explanation object shape: {shap_explanation.shape}")
    
    # Handle the case where shap returns multiclass outputs (even for binary)
    # Random Forest in SKLearn often does this.
    if len(shap_explanation.shape) == 3: # (samples, features, classes)
        print("Detected multiclass SHAP output, selecting class 1.")
        shap_explanation = shap_explanation[..., 1]

    # Global Summary Plot
    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_explanation, show=False)
    plt.title('SHAP Beeswarm Plot (Fraud Detection)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'))
    plt.close()

    # Local Force Plots
    print("Generating individual SHAP plots...")
    y_pred = model.predict(X_test)
    tp_idx = np.where((y_test == 1) & (y_pred == 1))[0]
    fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]

    tn_idx = np.where((y_test == 0) & (y_pred == 0))[0]

    cases = {
        'True Positive': tp_idx[0] if len(tp_idx) > 0 else None,
        'False Positive': fp_idx[0] if len(fp_idx) > 0 else None,
        'False Negative': fn_idx[0] if len(fn_idx) > 0 else None,
        'True Negative': tn_idx[0] if len(tn_idx) > 0 else None
    }

    for label, idx in cases.items():
        if idx is not None:
            print(f"Generating plot for {label} (Index {idx})")
            # Waterfall plots are often better for individual explanations than force plots in static files
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_explanation[idx], show=False)
            plt.title(f'SHAP Waterfall Plot: {label}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'shap_{label.lower().replace(" ", "_")}.png'))
            plt.close()
            
            # Also try force plot if specifically requested, but waterfall is a good alternative
            try:
                plt.figure(figsize=(15, 3))
                shap.plots.force(shap_explanation[idx], matplotlib=True, show=False)
                plt.savefig(os.path.join(output_dir, f'shap_force_{label.lower().replace(" ", "_")}.png'), bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Could not generate force plot for {label}: {e}")
        else:
            print(f"No instance found for {label}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE_DIR / "models" / "best_model_fraud_data.pkl"
    X_TEST_PATH = BASE_DIR / "data" / "processed" / "Fraud_X_test.csv"
    Y_TEST_PATH = BASE_DIR / "data" / "processed" / "Fraud_y_test.csv"
    FIGURE_DIR = BASE_DIR / "reports" / "figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        model, X_test, y_test = load_data_and_model(MODEL_PATH, X_TEST_PATH, Y_TEST_PATH)
        perform_explainability(model, X_test, y_test, FIGURE_DIR)
        print("\nAll tasks completed successfully!")
    except Exception as e:
        import traceback
        traceback.print_exc()
