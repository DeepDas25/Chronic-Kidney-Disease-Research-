# ============================================================
# CATBOOST HYPERPARAMETER TUNING (GRID SEARCH)
# ============================================================

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
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load datasets
X_train = pd.read_csv(_find_data_file("X_train_experiment.csv"))
y_train = pd.read_csv(_find_data_file("y_train_experiment.csv")).squeeze()

print("="*60)
print("[OK] Loaded datasets for Grid Search.")
print("="*60)

# Set up parameter grid
param_grid = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.03, 0.05, 0.1],
    'depth': [4, 6, 8]
}

model = CatBoostClassifier(random_state=42, verbose=0)

# Perform GridSearchCV with stratified 3-fold cross-validation to save time
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring='f1', # Optimize for F1 score due to class imbalance
    n_jobs=-1,
    verbose=1
)

print("[OK] Starting GridSearchCV...")
grid_search.fit(X_train, y_train)

print("\n" + "="*60)
print("GRID SEARCH COMPLETE")
print("="*60)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best F1 Score  : {grid_search.best_score_:.4f}")
print("="*60)

# Write results to a log file
log_path = os.path.join(script_dir, "catboost_tuning_results.txt")
with open(log_path, "w") as f:
    f.write("CatBoost Hyperparameter Tuning Results\n")
    f.write("======================================\n")
    f.write(f"Best Parameters: {grid_search.best_params_}\n")
    f.write(f"Best F1 Score: {grid_search.best_score_:.4f}\n")

print(f"[OK] Saved best parameters to: {log_path}")
