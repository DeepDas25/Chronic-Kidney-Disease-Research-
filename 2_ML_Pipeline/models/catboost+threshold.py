# ============================================================
# CATBOOST THRESHOLD TUNING
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

from sklearn.metrics import *

script_dir = os.path.dirname(os.path.abspath(__file__))

X_train = pd.read_csv(_find_data_file("X_train_experiment.csv"))
X_test = pd.read_csv(_find_data_file("X_test_experiment.csv"))

y_train = pd.read_csv(_find_data_file("y_train_experiment.csv")).squeeze()
y_test = pd.read_csv(_find_data_file("y_test_experiment.csv")).squeeze()

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    random_state=42,
    verbose=0
)

model.fit(X_train,y_train)

y_prob = model.predict_proba(X_test)[:,1]

thresholds = [0.50,0.45,0.40,0.35,0.30]

results=[]

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    results.append([
        threshold,
        accuracy_score(y_test,y_pred),
        precision_score(y_test,y_pred),
        recall_score(y_test,y_pred),
        f1_score(y_test,y_pred)
    ])

results_df = pd.DataFrame(
    results,
    columns=[
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
)

print("\n")
print("="*70)
print("CATBOOST THRESHOLD TUNING")
print("="*70)
print(results_df)