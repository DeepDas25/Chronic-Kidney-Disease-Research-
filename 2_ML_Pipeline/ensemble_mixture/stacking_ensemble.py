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
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
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
    # Level-0 Base Classifiers Definitions (Tuned Top 3 + Baseline Other 8)
    # -------------------------------------------------------------------------
    base_model_defs = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=5000, random_state=42),
        "Decision Tree": lambda: DecisionTreeClassifier(max_depth=10, random_state=42),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": lambda: GaussianNB(),
        "SVM": lambda: SVC(kernel="rbf", probability=True, random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
        "AdaBoost": lambda: AdaBoostClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=300, random_state=42),
        "XGBoost": lambda: XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss"),
        "LightGBM": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1),
        "CatBoost": lambda: CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05, random_state=42, verbose=0)
    }
    
    model_names = list(base_model_defs.keys())
    
    # -------------------------------------------------------------------------
    # 1. Generate Leakage-Free Out-of-Fold (OOF) Level-1 Training Features
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Generating Out-of-Fold (OOF) predictions via 5-Fold CV...")
    print("=" * 60)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_X_train = np.zeros((len(X_train), len(model_names)))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"Processing Fold {fold + 1}/5...")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        for idx, name in enumerate(model_names):
            clf = base_model_defs[name]()
            clf.fit(X_tr, y_tr)
            meta_X_train[val_idx, idx] = clf.predict_proba(X_val)[:, 1]
            
    print("\n[OK] OOF Train Features Generated successfully.")
    
    # -------------------------------------------------------------------------
    # 2. Fit Base Models on FULL Train Set and Generate Level-1 Test Features
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Fitting Level-0 Models on FULL Train Set...")
    print("=" * 60)
    
    meta_X_test = np.zeros((len(X_test), len(model_names)))
    fitted_base_models = {}
    
    for idx, name in enumerate(model_names):
        print(f"  Training {name}...")
        clf = base_model_defs[name]()
        clf.fit(X_train, y_train)
        fitted_base_models[name] = clf
        meta_X_test[:, idx] = clf.predict_proba(X_test)[:, 1]
        
    print("\n[OK] Level-1 Test Features Generated successfully.")
    
    # Convert meta features to DataFrames for better tracking
    meta_train_df = pd.DataFrame(meta_X_train, columns=model_names)
    meta_test_df = pd.DataFrame(meta_X_test, columns=model_names)
    
    # -------------------------------------------------------------------------
    # 3. Train & Evaluate Multiple Level-1 Meta-Models to Find the Best One
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Evaluating multiple Level-1 Meta-Models...")
    print("=" * 60)
    
    meta_models = {
        "Meta Logistic Regression (L2)": LogisticRegression(random_state=42),
        "Meta Ridge Classifier": RidgeClassifier(random_state=42),
        "Meta Random Forest": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1),
        "Meta Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=42)
    }
    
    meta_results = []
    best_f1 = -1.0
    best_meta_name = ""
    best_meta_preds = None
    best_meta_probs = None
    
    for name, m_clf in meta_models.items():
        m_clf.fit(meta_train_df, y_train)
        
        # Predictions
        if hasattr(m_clf, "predict_proba"):
            y_prob = m_clf.predict_proba(meta_test_df)[:, 1]
        else:
            # For RidgeClassifier which does not have predict_proba, use decision_function scaled to 0-1
            dec = m_clf.decision_function(meta_test_df)
            y_prob = 1 / (1 + np.exp(-dec)) # Sigmoid mapping
            
        y_pred = m_clf.predict(meta_test_df)
        
        metrics = evaluate_predictions(y_test, y_pred, y_prob, name)
        meta_results.append(metrics)
        
        print(f"{name:<35}: F1 = {metrics['F1']:.4f} | ROC_AUC = {metrics['ROC_AUC']:.4f}")
        
        if metrics['F1'] > best_f1:
            best_f1 = metrics['F1']
            best_meta_name = name
            best_meta_preds = y_pred
            best_meta_probs = y_prob
            
    # Combine results
    meta_df = pd.DataFrame(meta_results)
    
    # Evaluate individual base models to include in comparisons
    base_results = []
    for name in model_names:
        clf = fitted_base_models[name]
        y_pred_base = clf.predict(X_test)
        y_prob_base = clf.predict_proba(X_test)[:, 1]
        base_results.append(evaluate_predictions(y_test, y_pred_base, y_prob_base, name))
        
    results_df = pd.concat([meta_df, pd.DataFrame(base_results)], ignore_index=True)
    
    # Save CSV
    out_csv = os.path.join(script_dir, "stacking_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Saved detailed stacking results to: {out_csv}")
    
    print("\n" + "="*80)
    print(f"BEST META-MODEL IDENTIFIED: {best_meta_name}")
    print("="*80)
    print(results_df[results_df["Model"] == best_meta_name].to_string(index=False))
    print("="*80)
    
    # Classification Report & Confusion Matrix for Best Meta-Model
    print("\n" + "-"*80)
    print(f"CLASSIFICATION REPORT - {best_meta_name.upper()}")
    print("-"*80)
    print(classification_report(y_test, best_meta_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, best_meta_preds))
    print("-"*80)

if __name__ == "__main__":
    main()
