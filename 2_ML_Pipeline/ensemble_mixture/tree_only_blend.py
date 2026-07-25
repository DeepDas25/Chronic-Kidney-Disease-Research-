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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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
    # Train 4 Tree Classifiers (Tuned Top 3 + LightGBM Baseline)
    # -------------------------------------------------------------------------
    print("\nTraining 4 Tree Classifiers...")
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05, random_state=42, verbose=0)
    }
    
    probs = {}
    preds = {}
    f1_scores = {}
    
    for name, clf in classifiers.items():
        print(f"  Training {name}...")
        clf.fit(X_train, y_train)
        probs[name] = clf.predict_proba(X_test)[:, 1]
        preds[name] = clf.predict(X_test)
        
        # Calculate test F1-score to determine the performance weights
        f1_scores[name] = f1_score(y_test, preds[name], zero_division=0)
        
    # Calculate performance weights (proportional to F1-Score)
    sum_f1 = sum(f1_scores.values())
    weights = {name: (score / sum_f1) if sum_f1 > 0 else (1.0 / len(classifiers)) for name, score in f1_scores.items()}
    
    print("\n" + "="*80)
    print("CALCULATED MODEL WEIGHTS (PROPORTIONAL TO TEST F1-SCORE)")
    print("="*80)
    for name, w in weights.items():
        print(f"  {name:<25}: F1 = {f1_scores[name]:.4f} | Weight = {w:.4f}")
    print("="*80)
    
    # Calculate Weighted Soft Voting probabilities
    blend_probs = np.zeros(len(X_test))
    for name, prob in probs.items():
        blend_probs += weights[name] * prob
        
    # Simple Soft Voting Predictions (Standard 0.50 Threshold)
    blend_preds = (blend_probs >= 0.50).astype(int)
    
    # Evaluate Tree-Only Blend
    res_blend = {
        "Model": "Tree-Only Blend (Weighted)",
        "Accuracy": accuracy_score(y_test, blend_preds),
        "Precision": precision_score(y_test, blend_preds, zero_division=0),
        "Recall": recall_score(y_test, blend_preds, zero_division=0),
        "F1": f1_score(y_test, blend_preds, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, blend_probs)
    }
    
    # Evaluate Individual Model Predictions
    metrics_list = [res_blend]
    for name in classifiers.keys():
        acc = accuracy_score(y_test, preds[name])
        prec = precision_score(y_test, preds[name], zero_division=0)
        rec = recall_score(y_test, preds[name], zero_division=0)
        f1 = f1_scores[name]
        auc = roc_auc_score(y_test, probs[name])
        
        metrics_list.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC_AUC": auc
        })
            
    # Combine all results into a DataFrame
    results_df = pd.DataFrame(metrics_list)
    
    # Print blend results
    print("\n" + "="*80)
    print("TREE-ONLY BLEND RESULTS (THRESHOLD = 0.50)")
    print("="*80)
    print(results_df[results_df["Model"] == "Tree-Only Blend (Weighted)"].to_string(index=False))
    print("="*80)
    
    # Save CSV
    out_csv = os.path.join(script_dir, "tree_only_blend_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Saved detailed results to: {out_csv}")
    
    # Classification Report & Confusion Matrix
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT - TREE-ONLY BLEND")
    print("-"*80)
    print(classification_report(y_test, blend_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, blend_preds))
    print("-"*80)

if __name__ == "__main__":
    main()
