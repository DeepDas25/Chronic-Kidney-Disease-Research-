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
    roc_auc_score
)

# ============================================================
# FINAL RESULTS (DYNAMICALLY COMPUTED FROM ALL 11 MODELS)
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "Final_Model_Comparison_43Features.csv")

# Load final splits
X_train = pd.read_csv(_find_data_file("X_train_experiment.csv"))
X_test = pd.read_csv(_find_data_file("X_test_experiment.csv"))
y_train = pd.read_csv(_find_data_file("y_train_experiment.csv")).squeeze()
y_test = pd.read_csv(_find_data_file("y_test_experiment.csv")).squeeze()

# Define all 11 classifiers with matching baseline hyperparameters
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "GaussianNB": GaussianNB(),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, random_state=42, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_state=42, verbose=0)
}

print("="*60)
print("[OK] Loaded datasets. Training 11 classifiers dynamically...")
print("="*60)

results_list = []

for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    results_list.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC_AUC": round(auc, 4)
    })

df = pd.DataFrame(results_list)
print("\n[OK] Dynamic Model Training and Evaluation Complete.")

# ============================================================
# DATAFRAME PRINT AND EXPORT
# ============================================================

print("\n")
print("="*100)
print("FINAL MODEL COMPARISON TABLE")
print("="*100)
print(df)

df.to_csv(
    csv_path,
    index=False
)

# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(12,6))
plt.bar(df["Model"], df["Accuracy"])
plt.xticks(rotation=45, ha="right")
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "Accuracy_Comparison_43Features.png"), dpi=300)
# plt.show()

# ============================================================
# F1 GRAPH
# ============================================================

plt.figure(figsize=(12,6))
plt.bar(df["Model"], df["F1"])
plt.xticks(rotation=45, ha="right")
plt.title("F1 Score Comparison")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "F1_Comparison_43Features.png"), dpi=300)
# plt.show()

# ============================================================
# RECALL GRAPH
# ============================================================

plt.figure(figsize=(12,6))
plt.bar(df["Model"], df["Recall"])
plt.xticks(rotation=45, ha="right")
plt.title("Recall Comparison")
plt.ylabel("Recall")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "Recall_Comparison_43Features.png"), dpi=300)
# plt.show()

# ============================================================
# ROC AUC GRAPH
# ============================================================

plt.figure(figsize=(12,6))
plt.bar(df["Model"], df["ROC_AUC"])
plt.xticks(rotation=45, ha="right")
plt.title("ROC-AUC Comparison")
plt.ylabel("ROC-AUC")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "ROC_AUC_Comparison_43Features.png"), dpi=300)
# plt.show()

# ============================================================
# COMBINED GRAPH
# ============================================================

plt.figure(figsize=(14,7))

plt.plot(df["Model"], df["Accuracy"], marker="o", label="Accuracy")
plt.plot(df["Model"], df["Precision"], marker="o", label="Precision")
plt.plot(df["Model"], df["Recall"], marker="o", label="Recall")
plt.plot(df["Model"], df["F1"], marker="o", label="F1")
plt.plot(df["Model"], df["ROC_AUC"], marker="o", label="ROC-AUC")

plt.xticks(rotation=45, ha="right")
plt.legend()
plt.title("All Metrics Comparison")
plt.tight_layout()

plt.savefig(
    os.path.join(script_dir, "Combined_Model_Comparison_43Features.png"),
    dpi=300
)
# plt.show()

# ============================================================
# BEST MODEL ANALYSIS
# ============================================================

print("\n")
print("="*60)
print("BEST MODEL ANALYSIS")
print("="*60)

print("\nHighest Accuracy")
print(df.loc[df["Accuracy"].idxmax()])

print("\nHighest Precision")
print(df.loc[df["Precision"].idxmax()])

print("\nHighest Recall")
print(df.loc[df["Recall"].idxmax()])

print("\nHighest F1")
print(df.loc[df["F1"].idxmax()])

print("\nHighest ROC-AUC")
print(df.loc[df["ROC_AUC"].idxmax()])