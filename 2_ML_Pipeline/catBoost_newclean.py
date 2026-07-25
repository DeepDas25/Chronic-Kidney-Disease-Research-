# ============================================================
# MODEL 12 : CATBOOST (FINAL 43-FEATURE DATASET)
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
import shap
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

print("="*60)
print("TRAIN SHAPE :", X_train.shape)
print("TEST SHAPE  :", X_test.shape)
print("="*60)

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    random_state=42,
    verbose=0
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

print("\n")
print("="*60)
print("CATBOOST RESULTS (43 FEATURES)")
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

import matplotlib.pyplot as plt
# ============================================================
# SHAP ANALYSIS
# ============================================================

print("\n[OK] CatBoost Trained Successfully")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

print("[OK] SHAP Values Generated")

# ------------------------------------------------------------
# Feature Importance Table
# ------------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": abs(shap_values).mean(axis=0)
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("="*60)
print("TOP 20 SHAP FEATURES")
print("="*60)

print(feature_importance.head(20))

feature_importance.to_csv(
    os.path.join(
        script_dir,
        "SHAP_Top20_Features_Final43.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# SHAP BAR PLOT
# ------------------------------------------------------------

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        script_dir,
        "SHAP_BarPlot_Final43.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("[OK] Saved : SHAP_BarPlot_Final43.png")

# ------------------------------------------------------------
# SHAP SUMMARY PLOT
# ------------------------------------------------------------

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        script_dir,
        "SHAP_SummaryPlot_Final43.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("[OK] Saved : SHAP_SummaryPlot_Final43.png")

# ------------------------------------------------------------
# Top 10 SHAP Plot
# ------------------------------------------------------------

top10 = feature_importance.head(10)

plt.figure(figsize=(10,6))

plt.barh(
    top10["Feature"][::-1],
    top10["Importance"][::-1]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")

plt.title(
    "Top 10 SHAP Features (Final 43 Features Dataset)"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        script_dir,
        "Top10_SHAP_Features_Final43.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("[OK] Saved : Top10_SHAP_Features_Final43.png")

print("\n")
print("="*60)
print("SHAP ANALYSIS COMPLETED")
print("="*60)