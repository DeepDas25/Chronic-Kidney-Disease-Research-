import os
def _find_data_file(filename):
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        for sub in [os.path.join("3_Datasets", "experiment_splits"), os.path.join("3_Datasets", "processed"), ""]:
            cand = os.path.join(current, sub, filename)
            if os.path.exists(cand):
                return cand
        current = os.path.dirname(current)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_predictions(y_true, y_pred, y_prob, name="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": roc_auc
    }

def run_cascade(X, gnb_model, stage2_model, gnb_threshold=0.50):
    # Stage 1: Get GNB probabilities
    gnb_probs = gnb_model.predict_proba(X)[:, 1]
    gnb_positive_mask = gnb_probs >= gnb_threshold
    
    # Initialize final predictions and probabilities
    final_preds = np.zeros(len(X))
    final_probs = np.copy(gnb_probs)
    
    # Stage 2: Predict on Stage-1 positive suspected cases only
    if np.sum(gnb_positive_mask) > 0:
        X_suspected = X[gnb_positive_mask]
        final_preds[gnb_positive_mask] = stage2_model.predict(X_suspected)
        final_probs[gnb_positive_mask] = stage2_model.predict_proba(X_suspected)[:, 1]
        
    return final_preds, final_probs

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Load dataset splits
    print("=" * 60)
    print("[INFO] Loading datasets from parent directory...")
    print("=" * 60)
    X_train = pd.read_csv(_find_data_file("X_train_experiment.csv"))
    X_test = pd.read_csv(_find_data_file("X_test_experiment.csv"))
    y_train = pd.read_csv(_find_data_file("y_train_experiment.csv")).squeeze()
    y_test = pd.read_csv(_find_data_file("y_test_experiment.csv")).squeeze()
    
    print(f"Train Shape: {X_train.shape}")
    print(f"Test Shape : {X_test.shape}")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Train Models
    # -------------------------------------------------------------------------
    print("\nTraining Classifiers...")
    
    print("  Training Stage 1: Gaussian Naive Bayes...")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    print("  Training Stage 2: CatBoost (Tuned)...")
    catboost = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05, random_state=42, verbose=0)
    catboost.fit(X_train, y_train)
    
    print("  Training Stage 2: XGBoost (Tuned)...")
    xgboost = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
    xgboost.fit(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # Run Cascades under multiple configurations
    # -------------------------------------------------------------------------
    print("\nRunning Cascades...")
    print("=" * 60)
    
    cascade_configs = [
        {"name": "GNB (0.50) -> CatBoost", "stage2": catboost, "threshold": 0.50},
        {"name": "GNB (0.50) -> XGBoost", "stage2": xgboost, "threshold": 0.50},
        {"name": "GNB (0.30) -> CatBoost", "stage2": catboost, "threshold": 0.30},
        {"name": "GNB (0.30) -> XGBoost", "stage2": xgboost, "threshold": 0.30},
    ]
    
    cascade_results = []
    
    for config in cascade_configs:
        preds, probs = run_cascade(X_test, gnb, config["stage2"], gnb_threshold=config["threshold"])
        metrics = evaluate_predictions(y_test, preds, probs, config["name"])
        cascade_results.append(metrics)
        
        print(f"{config['name']:<25}: F1 = {metrics['F1']:.4f} | Recall = {metrics['Recall']:.4f} | ROC_AUC = {metrics['ROC_AUC']:.4f}")
        
    # Add individual base models for reference
    base_results = []
    # GNB Reference
    gnb_preds = gnb.predict(X_test)
    gnb_probs = gnb.predict_proba(X_test)[:, 1]
    base_results.append(evaluate_predictions(y_test, gnb_preds, gnb_probs, "GaussianNB (Base)"))
    
    # CatBoost Reference
    cb_preds = catboost.predict(X_test)
    cb_probs = catboost.predict_proba(X_test)[:, 1]
    base_results.append(evaluate_predictions(y_test, cb_preds, cb_probs, "CatBoost (Base)"))
    
    # XGBoost Reference
    xgb_preds = xgboost.predict(X_test)
    xgb_probs = xgboost.predict_proba(X_test)[:, 1]
    base_results.append(evaluate_predictions(y_test, xgb_preds, xgb_probs, "XGBoost (Base)"))
    
    results_df = pd.concat([pd.DataFrame(cascade_results), pd.DataFrame(base_results)], ignore_index=True)
    
    # Save CSV
    out_csv = os.path.join(script_dir, "cascade_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Saved detailed cascade results to: {out_csv}")
    
    # Print comparison
    print("\n" + "="*80)
    print("CLINICAL CASCADE VS BASE MODELS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    # Detailed reports for best cascade (based on F1)
    best_cascade = results_df.iloc[results_df[results_df["Model"].str.contains("->")]["F1"].idxmax()]
    best_cascade_name = best_cascade["Model"]
    
    best_config = [c for c in cascade_configs if c["name"] == best_cascade_name][0]
    best_preds, _ = run_cascade(X_test, gnb, best_config["stage2"], gnb_threshold=best_config["threshold"])
    
    print(f"\nDETAILED REPORT - BEST CASCADE: {best_cascade_name}")
    print("-"*80)
    print(classification_report(y_test, best_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, best_preds))
    print("-"*80)

if __name__ == "__main__":
    main()
