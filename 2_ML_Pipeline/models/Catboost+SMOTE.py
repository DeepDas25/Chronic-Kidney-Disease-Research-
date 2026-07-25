# ============================================================
# CATBOOST + SMOTE
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

from imblearn.over_sampling import SMOTE

from sklearn.metrics import *

script_dir = os.path.dirname(os.path.abspath(__file__))

X_train = pd.read_csv(_find_data_file("X_train_experiment.csv"))
X_test = pd.read_csv(_find_data_file("X_test_experiment.csv"))

y_train = pd.read_csv(_find_data_file("y_train_experiment.csv")).squeeze()
y_test = pd.read_csv(_find_data_file("y_test_experiment.csv")).squeeze()

print("="*60)
print("ORIGINAL TRAIN DISTRIBUTION")
print("="*60)
print(y_train.value_counts())

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\n")
print("="*60)
print("AFTER SMOTE")
print("="*60)
print(pd.Series(y_train_smote).value_counts())

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    random_state=42,
    verbose=0
)

model.fit(X_train_smote,y_train_smote)

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

print("\n")
print("="*60)
print("CATBOOST + SMOTE RESULTS")
print("="*60)

print(f"Accuracy  : {accuracy_score(y_test,y_pred):.4f}")
print(f"Precision : {precision_score(y_test,y_pred):.4f}")
print(f"Recall    : {recall_score(y_test,y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test,y_pred):.4f}")
print(f"ROC AUC   : {roc_auc_score(y_test,y_prob):.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test,y_pred))