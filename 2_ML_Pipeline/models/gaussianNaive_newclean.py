# ============================================================
# MODEL 4 : GAUSSIAN NAIVE BAYES (FINAL 75-FEATURE DATASET)
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

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------
# Get Script Directory
# ------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# Load Final Experiment Dataset
# ------------------------------------------------------------

X_train = pd.read_csv(
    _find_data_file("X_train_experiment.csv")
)

X_test = pd.read_csv(
    _find_data_file("X_test_experiment.csv")
)

y_train = pd.read_csv(
    _find_data_file("y_train_experiment.csv")
).squeeze()

y_test = pd.read_csv(
    _find_data_file("y_test_experiment.csv")
).squeeze()

print("="*60)
print("TRAIN SHAPE :", X_train.shape)
print("TEST SHAPE  :", X_test.shape)
print("="*60)

# ------------------------------------------------------------
# Model Training
# ------------------------------------------------------------

model = GaussianNB()

model.fit(X_train, y_train)

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n")
print("="*60)
print("GAUSSIAN NAIVE BAYES RESULTS (75 FEATURES)")
print("="*60)

print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC   : {roc_auc_score(y_test, y_prob):.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))