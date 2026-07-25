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

def evaluate_predictions(y_true, y_pred, name="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred)
    
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
    # Hard Voting Classifier Ensemble (Tuned Top 3 + Baseline Other 8)
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
    
    preds = {}
    for name, clf in classifiers.items():
        print(f"  Training {name}...")
        clf.fit(X_train, y_train)
        preds[name] = clf.predict(X_test)
        
    # Hard Voting Prediction (Majority Vote: >= 6 models)
    preds_df = pd.DataFrame(preds)
    voting_sum = preds_df.sum(axis=1)
    voting_preds = (voting_sum >= 6).astype(int)
    
    # Evaluate Hard Voting Ensemble
    res_voting = evaluate_predictions(y_test, voting_preds, "Hard Voting Ensemble")
    
    # Evaluate Individual Model Predictions
    metrics_list = [res_voting]
    for name in classifiers.keys():
        m = evaluate_predictions(y_test, preds[name], name)
        metrics_list.append(m)
            
    # Combine all results into a DataFrame
    results_df = pd.DataFrame(metrics_list)
    
    # Print voting ensemble results
    print("\n" + "="*80)
    print("HARD VOTING ENSEMBLE RESULTS")
    print("="*80)
    print(results_df[results_df["Model"] == "Hard Voting Ensemble"].to_string(index=False))
    print("="*80)
    
    # Save CSV
    out_csv = os.path.join(script_dir, "hard_voting_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Saved detailed results to: {out_csv}")
    
    # Classification Report & Confusion Matrix
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT - HARD VOTING ENSEMBLE")
    print("-"*80)
    print(classification_report(y_test, voting_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, voting_preds))
    print("-"*80)

if __name__ == "__main__":
    main()
