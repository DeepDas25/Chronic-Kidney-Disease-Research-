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

import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# NHANES Chronic Kidney Disease (CKD) Classification Pipeline\n",
    "\n",
    "This Jupyter Notebook consolidates all **11 machine learning models** implemented for predicting CKD on the refined **43-feature dataset**, and outputs the publication-ready figures for the research paper.\n",
    "\n",
    "### Models Included:\n",
    "1. **Logistic Regression**\n",
    "2. **Decision Tree**\n",
    "3. **K-Nearest Neighbors (KNN)**\n",
    "4. **Gaussian Naive Bayes**\n",
    "5. **Support Vector Machine (SVM)**\n",
    "6. **Random Forest**\n",
    "7. **AdaBoost**\n",
    "8. **Gradient Boosting**\n",
    "9. **XGBoost**\n",
    "10. **LightGBM**\n",
    "11. **CatBoost**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Imports and Data Loading"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import shap\n",
    "\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score,\n",
    "    precision_score,\n",
    "    recall_score,\n",
    "    f1_score,\n",
    "    roc_auc_score,\n",
    "    confusion_matrix,\n",
    "    classification_report\n",
    ")\n",
    "\n",
    "# Robust path detection for 'ZIP creation nhanes' home directory\n",
    "current_path = os.path.abspath(os.getcwd())\n",
    "home_dir = current_path\n",
    "while home_dir and os.path.basename(home_dir) != \"ZIP creation nhanes\":\n",
    "    parent = os.path.dirname(home_dir)\n",
    "    if parent == home_dir:\n",
    "        break\n",
    "    home_dir = parent\n",
    "\n",
    "print(f\"Detected Home Directory: {home_dir}\")\n",
    "\n",
    "# Load target train/test dataset splits\n",
    "X_train = pd.read_csv(\"X_train_experiment.csv\")\n",
    "X_test = pd.read_csv(\"X_test_experiment.csv\")\n",
    "y_train = pd.read_csv(\"y_train_experiment.csv\").squeeze()\n",
    "y_test = pd.read_csv(\"y_test_experiment.csv\").squeeze()\n",
    "\n",
    "print(\"=\" * 50)\n",
    "print(f\"Training set shape: {X_train.shape}\")\n",
    "print(f\"Testing set shape:  {X_test.shape}\")\n",
    "print(\"=\" * 50)\n",
    "\n",
    "# Dictionary to store compiled performance results\n",
    "all_results = {}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Model Training and Evaluation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model 1: Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "model_lr = LogisticRegression(max_iter=5000, random_state=42)\n",
    "model_lr.fit(X_train, y_train)\n",
    "\n",
    "y_pred_lr = model_lr.predict(X_test)\n",
    "y_prob_lr = model_lr.predict_proba(X_test)[:, 1]\n",
    "\n",
    "all_results[\"Logistic Regression\"] = {\n",
    "    \"Accuracy\": accuracy_score(y_test, y_pred_lr),\n",
    "    \"Precision\": precision_score(y_test, y_pred_lr),\n",
    "    \"Recall\": recall_score(y_test, y_pred_lr),\n",
    "    \"F1\": f1_score(y_test, y_pred_lr),\n",
    "    \"ROC_AUC\": roc_auc_score(y_test, y_prob_lr)\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model 2: Decision Tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "model_dt = DecisionTreeClassifier(max_depth=10, random_state=42)\n",
    "model_dt.fit(X_train, y_train)\n",
    "\n",
    "y_pred_dt = model_dt.predict(X_test)\n",
    "y_prob_dt = model_dt.predict_proba(X_test)[:, 1]\n",
    "\n",
    "all_results[\"Decision Tree\"] = {\n",
    "    \"Accuracy\": accuracy_score(y_test, y_pred_dt),\n",
    "    \"Precision\": precision_score(y_test, y_pred_dt),\n",
    "    \"Recall\": recall_score(y_test, y_pred_dt),\n",
    "    \"F1\": f1_score(y_test, y_pred_dt),\n",
    "    \"ROC_AUC\": roc_auc_score(y_test, y_prob_dt)\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model 3: K-Nearest Neighbors (KNN)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "model_knn = KNeighborsClassifier(n_neighbors=5)\n",
    "model_knn.fit(X_train, y_train)\n",
    "\n",
    "y_pred_knn = model_knn.predict(X_test)\n",
    "y_prob_knn = model_knn.predict_proba(X_test)[:, 1]\n",
    "\n",
    "all_results[\"KNN\"] = {\n",
    "    \"Accuracy\": accuracy_score(y_test, y_pred_knn),\n",
    "    \"Precision\": precision_score(y_test, y_pred_knn),\n",
    "    \"Recall\": recall_score(y_test, y_pred_knn),\n",
    "    \"F1\": f1_score(y_test, y_pred_knn),\n",
    "    \"ROC_AUC\": roc_auc_score(y_test, y_prob_knn)\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model 4: Gaussian Naive Bayes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.naive_bayes import GaussianNB\n",
    "\n",
    "model_gnb = GaussianNB()\n",
    "model_gnb.fit(X_train, y_train)\n",
    "\n",
    "y_pred_gnb = model_gnb.predict(X_test)\n",
    "y_prob_gnb = model_gnb.predict_proba(X_test)[:, 1]\n",
    "\n",
    "all_results[\"GaussianNB\"] = {\n",
    "    \"Accuracy\": accuracy_score(y_test, y_pred_gnb),\n",
    "    \"Precision\": precision_score(y_test, y_pred_gnb),\n",
    "    \"Recall\": recall_score(y_test, y_pred_gnb),\n",
    "    \"F1\": f1_score(y_test, y_pred_gnb),\n",
    "    \"ROC_AUC\": roc_auc_score(y_test, y_prob_gnb)\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model 5: Support Vector Machine (SVM)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.svm import SVC\n",
    "\n",
    "model_svm = SVC(kernel=\"rbf\", probability=True, random_state=42)\n",
    "model_svm.fit(X_train, y_train)\n",
    "\n",
    "y_pred_svm = model_svm.predict(X_test)\n",
    "y_prob_svm = model_svm.predict_proba(X_test)[:, 1]\n",
    "\n",
    "all_results[\"SVM\"] = {\n",
    "    \"Accuracy\": accuracy_score(y_test, y_pred_svm),\n",
    "    \"Precision\": precision_score(y_test, y_pred_svm),\n",
    "    \"Recall\": recall_score(y_test, y_pred_svm),\n",
    "    \"F1\": f1_score(y_test, y_pred_svm),\n",
    "    \"ROC_AUC\": roc_auc_score(y_test, y_prob_svm)\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model 6: Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "model_rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)\n",
    "model_rf.fit(X_train, y_train)\n",
    "\n",
    "y_pred_rf = model_rf.predict(X_test)\n",
    "y_prob_rf = model_rf.predict_proba(X_test)[:, 1]\n",
    "\n",
    "all_results[\"Random Forest\"] = {\n",
    "    \"Accuracy\": accuracy_score(y_test, y_pred_rf),\n",
    "    \"Precision\": precision_score(y_test, y_pred_rf),\n",
    "    \"Recall\": recall_score(y_test, y_pred_rf),\n",
    "    \"F1\": f1_score(y_test, y_pred_rf),\n",
    "    \"ROC_AUC\": roc_auc_score(y_test, y_prob_rf)\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model 7: AdaBoost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import AdaBoostClassifier\n",
    "\n",
    "model_ada = AdaBoostClassifier(n_estimators=200, random_state=42)\n",
    "model_ada.fit(X_train, y_train)\n",
    "\n",
    "y_pred_ada = model_ada.predict(X_test)\n",
    "y_prob_ada = model_ada.predict_proba(X_test)[:, 1]\n",
    "\n",
    "all_results[\"AdaBoost\"] = {\n",
    "    \"Accuracy\": accuracy_score(y_test, y_pred_ada),\n",
    "    \"Precision\": precision_score(y_test, y_pred_ada),\n",
    "    \"Recall\": recall_score(y_test, y_pred_ada),\n",
    "    \"F1\": f1_score(y_test, y_pred_ada),\n",
    "    \"ROC_AUC\": roc_auc_score(y_test, y_prob_ada)\n}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model 8: Gradient Boosting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "\n",
    "model_gb = GradientBoostingClassifier(n_estimators=300, random_state=42)\n",
    "model_gb.fit(X_train, y_train)\n",
    "\n",
    "y_pred_gb = model_gb.predict(X_test)\n",
    "y_prob_gb = model_gb.predict_proba(X_test)[:, 1]\n",
    "\n",
    "all_results[\"Gradient Boosting\"] = {\n",
    "    \"Accuracy\": accuracy_score(y_test, y_pred_gb),\n",
    "    \"Precision\": precision_score(y_test, y_pred_gb),\n",
    "    \"Recall\": recall_score(y_test, y_pred_gb),\n",
    "    \"F1\": f1_score(y_test, y_pred_gb),\n",
    "    \"ROC_AUC\": roc_auc_score(y_test, y_prob_gb)\n}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model 9: XGBoost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from xgboost import XGBClassifier\n",
    "\n",
    "model_xgb = XGBClassifier(n_estimators=300, random_state=42, eval_metric=\"logloss\")\n",
    "model_xgb.fit(X_train, y_train)\n",
    "\n",
    "y_pred_xgb = model_xgb.predict(X_test)\n",
    "y_prob_xgb = model_xgb.predict_proba(X_test)[:, 1]\n",
    "\n",
    "all_results[\"XGBoost\"] = {\n",
    "    \"Accuracy\": accuracy_score(y_test, y_pred_xgb),\n",
    "    \"Precision\": precision_score(y_test, y_pred_xgb),\n",
    "    \"Recall\": recall_score(y_test, y_pred_xgb),\n",
    "    \"F1\": f1_score(y_test, y_pred_xgb),\n",
    "    \"ROC_AUC\": roc_auc_score(y_test, y_prob_xgb)\n}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model 10: LightGBM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from lightgbm import LGBMClassifier\n",
    "\n",
    "model_lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)\n",
    "model_lgb.fit(X_train, y_train)\n",
    "\n",
    "y_pred_lgb = model_lgb.predict(X_test)\n",
    "y_prob_lgb = model_lgb.predict_proba(X_test)[:, 1]\n",
    "\n",
    "all_results[\"LightGBM\"] = {\n",
    "    \"Accuracy\": accuracy_score(y_test, y_pred_lgb),\n",
    "    \"Precision\": precision_score(y_test, y_pred_lgb),\n",
    "    \"Recall\": recall_score(y_test, y_pred_lgb),\n",
    "    \"F1\": f1_score(y_test, y_pred_lgb),\n",
    "    \"ROC_AUC\": roc_auc_score(y_test, y_prob_lgb)\n}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model 11: CatBoost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from catboost import CatBoostClassifier\n",
    "\n",
    "model_cat = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_state=42, verbose=0)\n",
    "model_cat.fit(X_train, y_train)\n",
    "\n",
    "y_pred_cat = model_cat.predict(X_test)\n",
    "y_prob_cat = model_cat.predict_proba(X_test)[:, 1]\n",
    "\n",
    "all_results[\"CatBoost\"] = {\n",
    "    \"Accuracy\": accuracy_score(y_test, y_pred_cat),\n",
    "    \"Precision\": precision_score(y_test, y_pred_cat),\n",
    "    \"Recall\": recall_score(y_test, y_pred_cat),\n",
    "    \"F1\": f1_score(y_test, y_pred_cat),\n",
    "    \"ROC_AUC\": roc_auc_score(y_test, y_prob_cat)\n}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Results Synthesis and Comparison (Figure 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_results = pd.DataFrame(all_results).T.reset_index().rename(columns={\"index\": \"Model\"})\n",
    "\n",
    "print(\"=\" * 80)\n",
    "print(\"FINAL MODEL COMPARISON TABLE\")\n",
    "print(\"=\" * 80)\n",
    "print(df_results.to_string(index=False))\n",
    "print(\"=\" * 80)\n",
    "\n",
    "# Generate Figure 1\n",
    "metrics = [\"Accuracy\", \"Precision\", \"Recall\", \"F1\", \"ROC_AUC\"]\n",
    "fig, axes = plt.subplots(3, 2, figsize=(16, 18))\n",
    "axes = axes.ravel()\n",
    "\n",
    "colors = sns.color_palette(\"muted\", len(df_results))\n",
    "\n",
    "for i, metric in enumerate(metrics):\n",
    "    sns.barplot(data=df_results, x=\"Model\", y=metric, ax=axes[i], palette=colors)\n",
    "    axes[i].set_title(f\"{metric} Comparison Across Models\", fontsize=14, fontweight=\"bold\")\n",
    "    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha=\"right\")\n",
    "    axes[i].set_ylabel(metric, fontsize=12)\n",
    "    axes[i].set_xlabel(\"\")\n",
    "    axes[i].grid(True, linestyle=\"--\", alpha=0.5)\n",
    "\n",
    "# Hide the unused 6th subplot\n",
    "fig.delaxes(axes[5])\n",
    "\n",
    "plt.tight_layout()\n",
    "fig_path_1 = os.path.join(home_dir, \"Figure1_Classifier_Comparison.png\")\n",
    "plt.savefig(fig_path_1, dpi=300, bbox_inches=\"tight\")\n",
    "plt.show()\n",
    "\n",
    "print(f\"[OK] Figure 1 saved to: {fig_path_1}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Decision Threshold Tuning (Figure 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate Figure 2\n",
    "thresholds = [0.50, 0.45, 0.40, 0.35, 0.30]\n",
    "recalls = []\n",
    "f1s = []\n",
    "\n",
    "for t in thresholds:\n",
    "    y_pred_t = (y_prob_cat >= t).astype(int)\n",
    "    recalls.append(recall_score(y_test, y_pred_t))\n",
    "    f1s.append(f1_score(y_test, y_pred_t))\n",
    "\n",
    "plt.figure(figsize=(8, 5))\n",
    "plt.plot(thresholds, recalls, marker=\"o\", linewidth=2, color=\"#1f77b4\", label=\"Recall\")\n",
    "plt.plot(thresholds, f1s, marker=\"s\", linewidth=2, color=\"#ff7f0e\", label=\"F1-Score\")\n",
    "\n",
    "plt.xlabel(\"Decision Threshold\", fontsize=12)\n",
    "plt.ylabel(\"Metric Score\", fontsize=12)\n",
    "plt.title(\"Performance Metrics Variation Across Decision Thresholds (CatBoost)\", fontsize=13, fontweight=\"bold\")\n",
    "plt.legend(fontsize=11)\n",
    "plt.grid(True, linestyle=\"--\", alpha=0.6)\n",
    "plt.gca().invert_xaxis()  # Show threshold going from 0.50 down to 0.30\n",
    "\n",
    "plt.tight_layout()\n",
    "fig_path_2 = os.path.join(home_dir, \"Figure2_Threshold_Tuning.png\")\n",
    "plt.savefig(fig_path_2, dpi=300, bbox_inches=\"tight\")\n",
    "plt.show()\n",
    "\n",
    "print(f\"[OK] Figure 2 saved to: {fig_path_2}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. SHAP Explainability Analysis (Figure 3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate Figure 3\n",
    "explainer = shap.TreeExplainer(model_cat)\n",
    "shap_values = explainer.shap_values(X_test)\n",
    "\n",
    "plt.figure(figsize=(10, 8))\n",
    "shap.summary_plot(shap_values, X_test, show=False)\n",
    "\n",
    "plt.title(\"Top 20 Features Ranked by Mean Absolute SHAP Values (CatBoost)\", fontsize=14, fontweight=\"bold\", pad=20)\n",
    "plt.tight_layout()\n",
    "\n",
    "fig_path_3 = os.path.join(home_dir, \"Figure3_SHAP_Summary.png\")\n",
    "plt.savefig(fig_path_3, dpi=300, bbox_inches=\"tight\")\n",
    "plt.show()\n",
    "\n",
    "print(f\"[OK] Figure 3 saved to: {fig_path_3}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

out_path = os.path.join(os.getcwd(), "NHANES_CKD_43Features_ML_Pipeline.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"[OK] Created and updated jupyter notebook at: {out_path}")
