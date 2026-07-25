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
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load clean datasets
X_train = pd.read_csv(_find_data_file("X_train_experiment.csv"))
X_test = pd.read_csv(_find_data_file("X_test_experiment.csv"))
y_train = pd.read_csv(_find_data_file("y_train_experiment.csv")).squeeze()
y_test = pd.read_csv(_find_data_file("y_test_experiment.csv")).squeeze()

print("="*60)
print("[OK] Running GridSearchCV hyperparameter tuning on top 3 models...")
print("="*60)

results = []

# 1. RandomForest Tuning
print("Tuning RandomForest...")
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [8, 12, None]
}
rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='roc_auc', verbose=1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
rf_preds = best_rf.predict(X_test)
rf_probs = best_rf.predict_proba(X_test)[:, 1]

# 2. XGBoost Tuning
print("\nTuning XGBoost...")
xgb = XGBClassifier(random_state=42, eval_metric="logloss")
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1]
}
xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=3, scoring='roc_auc', verbose=1)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
xgb_preds = best_xgb.predict(X_test)
xgb_probs = best_xgb.predict_proba(X_test)[:, 1]

# 3. CatBoost Tuning
print("\nTuning CatBoost...")
cb = CatBoostClassifier(random_state=42, verbose=0)
cb_param_grid = {
    'iterations': [100, 200],
    'depth': [4, 6],
    'learning_rate': [0.05, 0.1]
}
cb_grid = GridSearchCV(cb, cb_param_grid, cv=3, scoring='roc_auc', verbose=1)
cb_grid.fit(X_train, y_train)
best_cb = cb_grid.best_estimator_
cb_preds = best_cb.predict(X_test)
cb_probs = best_cb.predict_proba(X_test)[:, 1]

# Print summary
summary_lines = [
    "GridSearchCV Tuning Results Summary",
    "=============================",
    f"Random Forest Best Params: {rf_grid.best_params_}",
    f"Random Forest Best CV ROC-AUC: {rf_grid.best_score_:.4f}",
    f"Random Forest Test ROC-AUC: {roc_auc_score(y_test, rf_probs):.4f}",
    "",
    f"XGBoost Best Params: {xgb_grid.best_params_}",
    f"XGBoost Best CV ROC-AUC: {xgb_grid.best_score_:.4f}",
    f"XGBoost Test ROC-AUC: {roc_auc_score(y_test, xgb_probs):.4f}",
    "",
    f"CatBoost Best Params: {cb_grid.best_params_}",
    f"CatBoost Best CV ROC-AUC: {cb_grid.best_score_:.4f}",
    f"CatBoost Test ROC-AUC: {roc_auc_score(y_test, cb_probs):.4f}",
]

tuning_results_path = os.path.join(script_dir, "tuning_results_summary.txt")
with open(tuning_results_path, "w") as f:
    f.write("\n".join(summary_lines))

print("\n" + "\n".join(summary_lines))
print(f"\n[SUCCESS] Saved tuning results summary to {tuning_results_path}")
