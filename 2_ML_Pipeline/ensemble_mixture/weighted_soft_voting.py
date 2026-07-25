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
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import cross_val_score, StratifiedKFold

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
    # Train 11 Classifiers (Tuned Top 3 + Baseline Other 8)
    # -------------------------------------------------------------------------
    print("\nTraining 11 Classifiers (Tuned Top 3: RF, XGBoost, CatBoost)...")
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": GaussianNB(),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05, random_state=42, verbose=0)
    }
    
    probs = {}
    preds = {}
    f1_scores_cv = {}
    
    for name, clf in classifiers.items():
        print(f"  Training {name}...")
        clf.fit(X_train, y_train)
        probs[name] = clf.predict_proba(X_test)[:, 1]
        preds[name] = clf.predict(X_test)
        
        # Calculate cross‑validated F1‑score on training data to determine performance weights
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_train, y_train, scoring='f1', cv=cv)
        f1_scores_cv[name] = cv_scores.mean()

    # Calculate performance weights (proportional to cross‑validated F1‑Score)
    sum_f1 = sum(f1_scores_cv.values())
    weights = {name: (score / sum_f1) if sum_f1 > 0 else (1.0 / len(classifiers)) for name, score in f1_scores_cv.items()}
    
    print("\n" + "="*80)
    print("CALCULATED MODEL WEIGHTS (PROPORTIONAL TO TRAINING CV F1-SCORE)")
    print("="*80)
    for name, w in weights.items():
        print(f"  {name:<25}: F1 = {f1_scores_cv[name]:.4f} | Weight = {w:.4f}")
    print("="*80)
    
    # Calculate Weighted Soft Voting probabilities
    weighted_probs = np.zeros(len(X_test))
    for name, prob in probs.items():
        weighted_probs += weights[name] * prob
        
    # Simple Soft Voting Predictions (Standard 0.50 Threshold)
    voting_preds = (weighted_probs >= 0.50).astype(int)
    
    # Evaluate Weighted Soft Voting Ensemble
    res_voting = {
        "Model": "Weighted Soft Voting Ensemble",
        "Accuracy": accuracy_score(y_test, voting_preds),
        "Precision": precision_score(y_test, voting_preds, zero_division=0),
        "Recall": recall_score(y_test, voting_preds, zero_division=0),
        "F1": f1_score(y_test, voting_preds, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, weighted_probs)
    }
    
    # Evaluate Individual Model Predictions
    metrics_list = [res_voting]
    for name in classifiers.keys():
        acc = accuracy_score(y_test, preds[name])
        prec = precision_score(y_test, preds[name], zero_division=0)
        rec = recall_score(y_test, preds[name], zero_division=0)
        f1 = f1_score(y_test, preds[name], zero_division=0)
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
    
    # Print voting ensemble results
    print("\n" + "="*80)
    print("WEIGHTED SOFT VOTING ENSEMBLE RESULTS (THRESHOLD = 0.50)")
    print("="*80)
    print(results_df[results_df["Model"] == "Weighted Soft Voting Ensemble"].to_string(index=False))
    print("="*80)
    
    # Save CSV
    out_csv = os.path.join(script_dir, "weighted_soft_voting_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Saved detailed results to: {out_csv}")
    
    # Classification Report & Confusion Matrix
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT - WEIGHTED SOFT VOTING ENSEMBLE (THRESHOLD = 0.50)")
    print("-"*80)
    print(classification_report(y_test, voting_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, voting_preds))
    print("-"*80)

if __name__ == "__main__":
    main()
