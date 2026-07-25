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
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_curve, roc_auc_score

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load clean datasets
X_train = pd.read_csv(_find_data_file("X_train_experiment.csv"))
X_test = pd.read_csv(_find_data_file("X_test_experiment.csv"))
y_train = pd.read_csv(_find_data_file("y_train_experiment.csv")).squeeze()
y_test = pd.read_csv(_find_data_file("y_test_experiment.csv")).squeeze()

# Define top 3 models with best parameters
models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss"),
    "CatBoost": CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05, random_state=42, verbose=0)
}

plt.figure(figsize=(8, 6))

for name, model in models.items():
    print(f"Training {name} with best parameters...")
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Tuned Top 3 Classifiers")
plt.legend()
plt.grid(True)
plt.tight_layout()

roc_path = os.path.join(script_dir, "ROC_Curves_Top3.png")
plt.savefig(roc_path, dpi=300)
plt.close()

print(f"\n[SUCCESS] ROC Curves saved to {roc_path}")
