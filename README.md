# Machine Learning for Early Detection of Chronic Kidney Disease (CKD)

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.60.0-FF4B4B.svg)](https://streamlit.io/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-success?logo=streamlit)](https://ckd-diagnosis-ai.streamlit.app/)
[![Dataset](https://img.shields.io/badge/Data-CDC%20NHANES-green.svg)](https://www.cdc.gov/nchs/nhanes/)
[![ML Framework](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20CatBoost%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed%20%26%20Verified-success.svg)](#)

An end-to-end Machine Learning research system and interactive Streamlit web dashboard for early detection of **Chronic Kidney Disease (CKD)** using high-dimensional epidemiological data from the **CDC NHANES** survey (5,552 patients, 43 clinical features).

---

## 📌 Project Highlights

* **Epidemiological Gold-Standard Dataset**: Extracted and merged from CDC's National Health and Nutrition Examination Survey (NHANES) adult cohort (`RIDAGEYR >= 18`).
* **Strict Data Leakage Prevention**: Derived markers (**eGFR**, **ACR**, **Serum Creatinine**, **Urinary Albumin**) were systematically removed during feature selection to prevent data leakage.
* **11+ Machine Learning Models Evaluated**: Comprehensive evaluation across Decision Trees, Random Forests, XGBoost, LightGBM, CatBoost, SVM (RBF), Naive Bayes, K-Nearest Neighbors, Logistic Regression, AdaBoost, and a **2-Level Stacking Meta-Ensemble**.
* **Clinical Performance**: The Stacking Meta-Learner achieved the top performance with **87.13% Accuracy** and **0.833 ROC-AUC**.
* **Explainable AI (SHAP & Counterfactuals)**: Integrated SHAP beeswarm interpretability plots and actionable patient-level clinical counterfactual recommendations to open the "black box" in healthcare AI.
* **Interactive Smart Diagnostic Tool**: A 10-page Streamlit web dashboard featuring interactive diagnostic screening, missing data guidance, and CDC population median auto-filling.

---

## 📁 Repository Directory Structure

```text
CKD_Project/
├── README.md                           # Comprehensive project documentation & user guide
├── .gitignore                          # Git rules excluding temp logs, __pycache__, & catboost_info
│
├── 1_Web_Application/                  # Interactive Streamlit Web Interface & Diagnostic Tool
│   ├── .streamlit/                     # Streamlit theme & server configuration
│   ├── requirements.txt                # Python package dependencies
│   └── src/
│       ├── app.py                      # Main Streamlit application entry point
│       ├── style.css                   # Custom modern UI styling & CSS theme
│       ├── train.py                    # Runtime CatBoost training helper
│       ├── pages/                      # 10 Multi-page Streamlit views
│       │   ├── 01_home.py              # Project hero banner & statistics at a glance
│       │   ├── 02_ckd_info.py          # Clinical background on CKD & risk factors
│       │   ├── 03_dataset.py           # NHANES survey details & train/test splits
│       │   ├── 04_preprocessing.py     # Leakage removal & preprocessing pipeline
│       │   ├── 05_eda.py               # Exploratory Data Analysis & statistical charts
│       │   ├── 06_model_comparison.py  # 11-Model comparison matrix & ROC curves
│       │   ├── 07_explainable_ai.py    # SHAP global feature attributions & beeswarm plots
│       │   ├── 08_research_findings.py # Key clinical research insights & counterfactuals
│       │   ├── 09_team.py              # Research team, guide, & institution details
│       │   └── 10_prediction.py        # Smart Diagnostic Tool with 100% input validation
│       └── assets/                     # 25 static UI charts, ROC curves, CSV metrics & logo
│
├── 2_ML_Pipeline/                      # Machine Learning Pipeline, Models & Experiments
│   ├── run_pipeline.py                 # Master orchestrator running the full ML pipeline
│   ├── clean_data.py                   # Data preprocessing, leakage removal & split generator
│   ├── generate_eda_and_analysis_final.py # EDA & statistical figures generator
│   ├── catBoost_newclean.py            # CatBoost training & SHAP interpretability analysis
│   ├── tune_top_models.py              # Hyperparameter tuning module
│   ├── counterfactual_explain.py       # Clinical counterfactual recommendation generator
│   ├── result_clean.py                 # Baseline model evaluation matrix generator
│   ├── plot_roc_curves.py              # ROC curve plotting script
│   ├── build_ieee_docx.py              # Research paper draft generator script
│   ├── create_ipynb_v2.py              # Jupyter notebook generator script
│   ├── NHANES_CKD_43Features_ML_Pipeline.ipynb # Complete Jupyter Notebook
│   ├── models/                         # Individual classifier scripts (SVM, RF, XGBoost, etc.)
│   ├── ensemble_mixture/               # Stacking, Voting & Cascade Ensemble scripts
│   └── outputs/                        # 114 generated figures, SHAP plots, ROC charts & CSVs
│
├── 3_Datasets/                         # Datasets & Preprocessing Scripts
│   ├── raw/                            # Original CDC NHANES .xpt files (DEMO, BIOPRO, ALB_CR, etc.)
│   ├── processed/                      # Merged datasets (ckd_nhanes.csv & cleaned variant)
│   ├── experiment_splits/              # Preprocessed train/test CSV splits (X_train, y_train, etc.)
│   └── preprocessing_scripts/          # Data extraction scripts (create_ckd_dataset.py)
│
└── 4_Research_and_Reports/             # Documentation & Academic Research
    ├── Literature_Review.pdf           # Comprehensive Literature Review document
    ├── All_Paper_Summary.csv           # Comparative literature summary matrix
    ├── Research_Papers/                # Collection of 19 reference research PDFs
    ├── Weekly_Reports/                 # Weekly progress reports
    └── Project_Doc.docx                # Complete project documentation DOCX
```

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/Chronic-Kidney-Disease-Research-.git
cd Chronic-Kidney-Disease-Research-
pip install -r 1_Web_Application/requirements.txt
```

### 2. Launch the Streamlit Web Dashboard

To run the interactive web application on your local machine:

```bash
python -m streamlit run 1_Web_Application/src/app.py
```

The app will open automatically in your default browser at **`http://localhost:8501`**.

### 3. Re-run the Full Machine Learning Pipeline

To re-execute all data processing, model training, hyperparameter tuning, SHAP generation, and ensemble evaluation:

```bash
python 2_ML_Pipeline/run_pipeline.py
```

All regenerated figures, ROC curves, and evaluation CSVs will automatically sync into `2_ML_Pipeline/outputs/`.

### 4. Re-merge Raw CDC NHANES Files

To re-extract raw `.xpt` files from `3_Datasets/raw/` and regenerate `ckd_nhanes.csv`:

```bash
python 3_Datasets/preprocessing_scripts/create_ckd_dataset.py
```
---
## 🌐 Access the Live Application

You can use the deployed web application directly without installing anything:

**🔗 https://ckd-diagnosis-ai.streamlit.app/**

---

## 📊 Performance Comparison Matrix

The table below reflects the exact empirical evaluation metrics on the test dataset split (`N = 1,111` test cases) generated by `2_ML_Pipeline/run_pipeline.py`:

| Model Architecture | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Meta Logistic Regression (Stacking L2)** | **87.13%** | **0.8333** | **73.74%** | **38.42%** | **50.52%** |
| **Meta Ridge Classifier (Stacking L2)** | 87.04% | 0.8309 | 74.47% | 36.84% | 49.30% |
| **Meta Gradient Boosting (Stacking L2)** | 87.04% | 0.8265 | 75.56% | 35.79% | 48.57% |
| **XGBoost Classifier** | 87.04% | 0.8341 | 75.56% | 35.79% | 48.57% |
| **CatBoost Classifier** | 86.77% | 0.8326 | 73.63% | 35.26% | 47.69% |
| **Random Forest** | 86.68% | 0.8327 | 81.82% | 28.42% | 42.19% |
| **Gradient Boosting** | 86.50% | 0.8285 | 66.95% | 41.58% | 51.30% |
| **LightGBM** | 86.41% | 0.8096 | 69.70% | 36.32% | 47.75% |
| **Logistic Regression** | 86.32% | 0.8173 | 71.59% | 33.16% | 45.32% |
| **AdaBoost** | 85.78% | 0.8122 | 66.00% | 34.74% | 45.52% |
| **Support Vector Machine (SVM)** | 85.69% | 0.7926 | 78.18% | 22.63% | 35.10% |
| **K-Nearest Neighbors (KNN)** | 83.44% | 0.6879 | 56.25% | 14.21% | 22.69% |
| **Gaussian Naive Bayes** | 82.90% | 0.8129 | 50.00% | 49.47% | 49.74% |
| **Decision Tree** | 82.72% | 0.6479 | 49.21% | 32.63% | 39.24% |

---

## 🔬 Top Key Biomarkers (SHAP Feature Importance)

1. **Age (`RIDAGEYR`)**: Strongest global predictor; risk skews significantly higher with increasing age.
2. **Blood Urea Nitrogen (`LBXSBU`)**: Elevated BUN concentration strongly indicates declining renal filtration.
3. **Blood Pressure (`BPX_SYS_MEAN`, `BPX_DIA_MEAN`)**: Hypertension is directly correlated with glomerular damage.
4. **Diabetes Status (`DIQ010`)**: Clinical diabetic diagnosis increases CKD probability by over 35%.
5. **Serum Uric Acid (`LBXSUA`)**: Hyperuricemia serves as a secondary marker for impaired kidney excretion.

---

## 👨‍💻 Authors

1. Deepdas Somani
2. Viraj Kumar Sahu
3. Ayan Bhandari

**Supervisor : Mr. Divanshu Jain**

---

## ⚠️ Medical Disclaimer

This software and web application are designed strictly for **academic research, scientific demonstration, and educational purposes**. It is **not** a certified medical diagnostic device and should **not** replace clinical evaluation, professional medical advice, or physician laboratory diagnosis.
