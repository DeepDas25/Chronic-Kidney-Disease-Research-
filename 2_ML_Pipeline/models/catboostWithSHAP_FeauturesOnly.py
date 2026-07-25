# ============================================================
# CATBOOST + TOP 20 SHAP FEATURES
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
# Load Dataset
# ------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))

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

# ------------------------------------------------------------
# TOP 20 SHAP FEATURES
# ------------------------------------------------------------

top20_features = [
    "RIDAGEYR",
    "LBXSBU",
    "LBXSUA",
    "BPX_SYS_MEAN",
    "DIQ010",
    "LBXSIR",
    "LBXSCH",
    "INDFMPIR",
    "DMDEDUC2",
    "LBXSTR",
    "LBXMAGN",
    "LBXSLDSI",
    "LBXSGL",
    "LBXSGB",
    "RIDRETH1",
    "LBXSTP",
    "LBXSOSSI",
    "LBXSATSI",
    "BMXHT",
    "LBXSCA"
]

X_train = X_train[top20_features]
X_test = X_test[top20_features]

print("="*60)
print("TOP 20 FEATURE DATASET")
print("="*60)
print("TRAIN :", X_train.shape)
print("TEST  :", X_test.shape)

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    random_state=42,
    verbose=0
)

model.fit(X_train, y_train)

# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n")
print("="*60)
print("CATBOOST TOP-20 SHAP FEATURES RESULTS")
print("="*60)

print(f"Accuracy  : {accuracy_score(y_test,y_pred):.4f}")
print(f"Precision : {precision_score(y_test,y_pred):.4f}")
print(f"Recall    : {recall_score(y_test,y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test,y_pred):.4f}")
print(f"ROC AUC   : {roc_auc_score(y_test,y_prob):.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test,y_pred))

print("\nClassification Report")
print(classification_report(y_test,y_pred))