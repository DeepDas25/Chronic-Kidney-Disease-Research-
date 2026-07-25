# src/pages/10_prediction.py
"""
CKD Smart Diagnostic Tool — Premium AI Medical Dashboard
Trains a CatBoost classifier at runtime using pre-processed experiment CSVs.
Displays prediction, probability, risk meter, category and recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

#  Paths 
# Walk upward from this file to find the dataset directory,
# checking 3_Datasets/experiment_splits/ first, then MAIN/ — robust against Streamlit's runtime __file__ quirks.
def _find_main_dir():
    current = os.path.abspath(__file__)
    for _ in range(10):  # Walk up at most 10 levels
        current = os.path.dirname(current)
        # Check new structured dataset location first
        candidate1 = os.path.join(current, "3_Datasets", "experiment_splits", "X_train_experiment.csv")
        if os.path.exists(candidate1):
            return os.path.join(current, "3_Datasets", "experiment_splits")
        # Check MAIN folder
        candidate2 = os.path.join(current, "MAIN", "X_train_experiment.csv")
        if os.path.exists(candidate2):
            return os.path.join(current, "MAIN")
        # Check extracted_project folder
        candidate3 = os.path.join(current, "extracted_project", "MAIN", "X_train_experiment.csv")
        if os.path.exists(candidate3):
            return os.path.join(current, "extracted_project", "MAIN")
    return None

main_dir     = _find_main_dir()
X_TRAIN_PATH = os.path.join(main_dir, "X_train_experiment.csv") if main_dir else ""
Y_TRAIN_PATH = os.path.join(main_dir, "y_train_experiment.csv") if main_dir else ""


class StackingEnsembleWrapper:
    """Wrapper class that combines Level-0 Base Classifiers and Level-1 Meta Logistic Regression."""
    def __init__(self, base_models, meta_model, model_names):
        self.base_models = base_models
        self.meta_model = meta_model
        self.model_names = model_names

    def predict_proba(self, X):
        meta_features = []
        for name in self.model_names:
            clf = self.base_models[name]
            proba = clf.predict_proba(X)[:, 1]
            meta_features.append(proba)
        meta_X = np.column_stack(meta_features)
        return self.meta_model.predict_proba(meta_X)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


#  Model Training (cached so it only runs once per session) 
@st.cache_resource(show_spinner=" Loading Stacking Ensemble diagnostic model (11 Base Models + Meta-LR) — please wait (~10 sec)…")
def load_model():
    """Train Stacking Ensemble and fit input scaler/imputer on raw training data."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        if not main_dir or not os.path.exists(X_TRAIN_PATH):
            return None, "Training split CSV not found.", None, None

        RAW_DATA_PATH = os.path.join(main_dir, "ckd_nhanes.csv")
        if not os.path.exists(RAW_DATA_PATH):
            RAW_DATA_PATH = os.path.join(os.path.dirname(main_dir), "processed", "ckd_nhanes.csv")
        if not os.path.exists(RAW_DATA_PATH):
            return None, f"Raw data ckd_nhanes.csv not found.", None, None

        # Load raw dataset to fit scaling pipeline
        df_raw = pd.read_csv(RAW_DATA_PATH)

        # Replicate repeat exam consolidation
        systolic_cols = [c for c in df_raw.columns if c.startswith("BPXOSY")]
        diastolic_cols = [c for c in df_raw.columns if c.startswith("BPXODI")]
        pulse_cols = [c for c in df_raw.columns if c.startswith("BPXOPLS")]
        if systolic_cols:
            df_raw["BPX_SYS_MEAN"] = df_raw[systolic_cols].mean(axis=1)
            df_raw.drop(columns=systolic_cols, inplace=True)
        if diastolic_cols:
            df_raw["BPX_DIA_MEAN"] = df_raw[diastolic_cols].mean(axis=1)
            df_raw.drop(columns=diastolic_cols, inplace=True)
        if pulse_cols:
            df_raw["BPX_PLS_MEAN"] = df_raw[pulse_cols].mean(axis=1)
            df_raw.drop(columns=pulse_cols, inplace=True)

        # Drop leakages & IDs
        leakage_and_ids = ["SEQN", "eGFR", "ACR", "LBXSCR", "LBDSCRSI", "URXUMA", "URXUMS", "URXUCR", "URDACT", "URXCRS",
                           "URDUMALC", "URDUCRLC", "WTMEC2YR", "WTINT2YR", "WTPH2YR", "SDMVPSU", "SDMVSTRA", "SDDSRVYR",
                           "LBDSALSI", "LBDSBUSI", "LBDSGBSI", "LBDSGLSI", "LBDSIRSI", "LBDSPHSI", "LBDSTBSI", "LBDSCHSI",
                           "LBDSTPSI", "LBDSTRSI", "LBDSUASI", "RIDSTATR", "RIDEXMON", "BPAOARM", "LBDSGTLC", "LBDSTBLC", "BMDSTATS"]
        df_raw.drop(columns=leakage_and_ids, inplace=True, errors="ignore")

        # Drop high missing
        missing_percent = (df_raw.isnull().sum() / len(df_raw)) * 100
        high_missing_cols = missing_percent[missing_percent > 40].index.tolist()
        df_raw.drop(columns=high_missing_cols, inplace=True, errors="ignore")

        X_raw = df_raw.drop(columns=["CKD"])
        y_raw = df_raw["CKD"]

        # Train/Test Split
        X_train_raw, _, _, _ = train_test_split(
            X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
        )

        # Load scaled splits
        X_train = pd.read_csv(X_TRAIN_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

        # Categorical feature names
        categorical_features = ['RIAGENDR', 'DMQMILIZ', 'DMDBORN4', 'DMDEDUC2', 'DMDMARTZ',
                                 'DIQ010', 'DIQ160', 'DIQ180']

        # Cast cat columns to int
        for col in categorical_features:
            if col in X_train.columns:
                X_train[col] = X_train[col].fillna(0).astype(int)

        # Numerical columns list that remain in the final 43 features
        num_cols_final = [col for col in X_train.columns if col not in categorical_features]

        # Fit numerical Imputer and Scaler on raw training data
        num_imputer = SimpleImputer(strategy='median')
        num_scaler = StandardScaler()

        X_train_num_imputed = num_imputer.fit_transform(X_train_raw[num_cols_final])
        num_scaler.fit(X_train_num_imputed)

        # Train 11 Level-0 Base Classifiers
        base_model_defs = {
            "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "GaussianNB": GaussianNB(),
            "SVM": SVC(kernel="rbf", probability=True, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
            "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss"),
            "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1),
            "CatBoost": CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05, random_state=42, verbose=0)
        }

        model_names = list(base_model_defs.keys())
        meta_X_list = []
        fitted_models = {}

        for name in model_names:
            clf = base_model_defs[name]
            clf.fit(X_train, y_train)
            fitted_models[name] = clf
            meta_X_list.append(clf.predict_proba(X_train)[:, 1])

        meta_X_train = np.column_stack(meta_X_list)
        meta_model = LogisticRegression(random_state=42)
        meta_model.fit(meta_X_train, y_train)

        ensemble_model = StackingEnsembleWrapper(fitted_models, meta_model, model_names)
        feature_cols = list(X_train.columns)
        return ensemble_model, feature_cols, num_imputer, num_scaler
    except Exception as e:
        return None, str(e), None, None


#  Feature Metadata 
FEATURE_META = {
    # Demographics
    "RIDAGEYR":   {"label": "Age (years)",              "type": "num", "min": 18.0, "max": 85.0, "default": 50.0, "step": 1.0, "group": "Demographics"},
    "RIDRETH3":   {"label": "Race / Ethnicity",         "type": "cat", "options": {"Mexican American": 1, "Other Hispanic": 2, "Non-Hispanic White": 3, "Non-Hispanic Black": 4, "Non-Hispanic Asian": 6, "Other / Multi-Racial": 7}, "default": "Non-Hispanic White", "group": "Demographics"},
    "DMDHHSIZ":   {"label": "Household Size",           "type": "num", "min": 1.0,  "max": 7.0,  "default": 3.0,  "step": 1.0, "group": "Demographics"},
    "INDFMPIR":   {"label": "Poverty Income Ratio",     "type": "num", "min": 0.0,  "max": 5.0,  "default": 2.5,  "step": 0.1, "group": "Demographics"},
    "RIAGENDR":   {"label": "Gender",                   "type": "cat", "options": {"Male": 1, "Female": 2},    "default": "Male",       "group": "Demographics"},
    "DMDEDUC2":   {"label": "Education Level",          "type": "cat", "options": {"< 9th Grade": 1, "9-11th Grade": 2, "High School / GED": 3, "Some College": 4, "College Graduate": 5}, "default": "Some College", "group": "Demographics"},
    "DMDMARTZ":   {"label": "Marital Status",           "type": "cat", "options": {"Married / Living w/ Partner": 1, "Widowed": 2, "Divorced / Separated": 3, "Refused / Unknown": 99}, "default": "Married / Living w/ Partner", "group": "Demographics"},
    "DMQMILIZ":   {"label": "Military Service",         "type": "cat", "options": {"Yes": 1, "No": 2},          "default": "No",         "group": "Demographics"},
    "DMDBORN4":   {"label": "Born in USA?",             "type": "cat", "options": {"Yes (USA)": 1, "No (Other)": 2}, "default": "Yes (USA)", "group": "Demographics"},

    # Diabetes
    "DIQ010":     {"label": "Diagnosed with Diabetes?", "type": "cat", "options": {"Yes": 1, "No": 2, "Borderline": 3}, "default": "No", "group": "Clinical History"},
    "DIQ160":     {"label": "Told Prediabetes?",        "type": "cat", "options": {"Yes": 1, "No": 2},          "default": "No",         "group": "Clinical History"},
    "DIQ180":     {"label": "Blood Sugar Test (past yr)?","type": "cat","options": {"Yes": 1, "No": 2},          "default": "Yes",        "group": "Clinical History"},

    # Blood Pressure
    "BPX_SYS_MEAN": {"label": "Systolic BP (mmHg)",   "type": "num", "min": 80.0, "max": 220.0, "default": 125.0, "step": 1.0, "group": "Cardiovascular"},
    "BPX_DIA_MEAN": {"label": "Diastolic BP (mmHg)",  "type": "num", "min": 40.0, "max": 130.0, "default": 80.0,  "step": 1.0, "group": "Cardiovascular"},
    "BPX_PLS_MEAN": {"label": "Pulse Rate (bpm)",     "type": "num", "min": 40.0, "max": 120.0, "default": 72.0,  "step": 1.0, "group": "Cardiovascular"},

    # Kidney / Metabolic Labs
    "LBXSBU":     {"label": "Blood Urea Nitrogen — BUN (mg/dL)",   "type": "num", "min": 2.0,  "max": 100.0, "default": 16.0, "step": 0.5, "group": "Laboratory — Kidney & Metabolic"},
    "LBXSUA":     {"label": "Uric Acid (mg/dL)",                   "type": "num", "min": 1.0,  "max": 15.0,  "default": 5.5,  "step": 0.1, "group": "Laboratory — Kidney & Metabolic"},
    "LBXSIR":     {"label": "Serum Iron (ug/dL)",                  "type": "num", "min": 10.0, "max": 280.0, "default": 85.0, "step": 1.0, "group": "Laboratory — Kidney & Metabolic"},
    "LBXMAGN":    {"label": "Serum Magnesium (mg/dL)",             "type": "num", "min": 0.5,  "max": 4.0,   "default": 2.0,  "step": 0.1, "group": "Laboratory — Kidney & Metabolic"},
    "LBXSOSSI":   {"label": "Serum Osmolality (mmol/kg)",          "type": "num", "min": 260.0,"max": 330.0, "default": 290.0,"step": 1.0, "group": "Laboratory — Kidney & Metabolic"},
    "LBXSKSI":    {"label": "Potassium (mmol/L)",                  "type": "num", "min": 2.5,  "max": 7.0,   "default": 4.1,  "step": 0.1, "group": "Laboratory — Kidney & Metabolic"},
    "LBXSNASI":   {"label": "Sodium (mmol/L)",                     "type": "num", "min": 125.0,"max": 155.0, "default": 140.0,"step": 0.5, "group": "Laboratory — Kidney & Metabolic"},
    "LBXSPH":     {"label": "Phosphorus (mg/dL)",                  "type": "num", "min": 1.5,  "max": 7.0,   "default": 3.6,  "step": 0.1, "group": "Laboratory — Kidney & Metabolic"},
    "LBXSCLSI":   {"label": "Chloride (mmol/L)",                   "type": "num", "min": 85.0, "max": 115.0, "default": 103.0,"step": 0.5, "group": "Laboratory — Kidney & Metabolic"},
    "LBXSC3SI":   {"label": "Bicarbonate / CO2 (mmol/L)",          "type": "num", "min": 14.0, "max": 40.0,  "default": 25.0, "step": 0.5, "group": "Laboratory — Kidney & Metabolic"},

    # Liver / Metabolic
    "LBXSCH":     {"label": "Total Cholesterol (mg/dL)",           "type": "num", "min": 80.0, "max": 400.0, "default": 195.0,"step": 1.0, "group": "Laboratory — Lipids & Liver"},
    "LBXSTR":     {"label": "Triglycerides (mg/dL)",               "type": "num", "min": 20.0, "max": 600.0, "default": 120.0,"step": 1.0, "group": "Laboratory — Lipids & Liver"},
    "LBXSGL":     {"label": "Blood Glucose (mg/dL)",               "type": "num", "min": 40.0, "max": 450.0, "default": 100.0,"step": 1.0, "group": "Laboratory — Lipids & Liver"},
    "LBXSTP":     {"label": "Total Protein (g/dL)",                "type": "num", "min": 4.0,  "max": 10.0,  "default": 7.0,  "step": 0.1, "group": "Laboratory — Lipids & Liver"},
    "LBXSAL":     {"label": "Albumin (g/dL)",                      "type": "num", "min": 2.0,  "max": 6.0,   "default": 4.1,  "step": 0.1, "group": "Laboratory — Lipids & Liver"},
    "LBXSGB":     {"label": "Globulin (g/dL)",                     "type": "num", "min": 1.0,  "max": 6.0,   "default": 2.8,  "step": 0.1, "group": "Laboratory — Lipids & Liver"},
    "LBXSGTSI":   {"label": "GGT — Liver Enzyme (U/L)",            "type": "num", "min": 5.0,  "max": 500.0, "default": 28.0, "step": 1.0, "group": "Laboratory — Lipids & Liver"},
    "LBXSATSI":   {"label": "ALT — Liver Enzyme (U/L)",            "type": "num", "min": 5.0,  "max": 300.0, "default": 22.0, "step": 1.0, "group": "Laboratory — Lipids & Liver"},
    "LBXSAPSI":   {"label": "Alkaline Phosphatase (U/L)",          "type": "num", "min": 20.0, "max": 400.0, "default": 72.0, "step": 1.0, "group": "Laboratory — Lipids & Liver"},
    "LBXSASSI":   {"label": "AST — Liver Enzyme (U/L)",            "type": "num", "min": 5.0,  "max": 300.0, "default": 24.0, "step": 1.0, "group": "Laboratory — Lipids & Liver"},
    "LBXSLDSI":   {"label": "LDH — Lactate Dehydrogenase (U/L)",   "type": "num", "min": 80.0, "max": 600.0, "default": 168.0,"step": 1.0, "group": "Laboratory — Lipids & Liver"},
    "LBXSCK":     {"label": "Creatine Kinase (U/L)",               "type": "num", "min": 20.0, "max": 1000.0,"default": 110.0,"step": 1.0, "group": "Laboratory — Lipids & Liver"},
    "LBXSTB":     {"label": "Total Bilirubin (mg/dL)",             "type": "num", "min": 0.1,  "max": 5.0,   "default": 0.7,  "step": 0.05,"group": "Laboratory — Lipids & Liver"},
    "LBXSCA":     {"label": "Serum Calcium (mg/dL)",               "type": "num", "min": 6.0,  "max": 12.0,  "default": 9.2,  "step": 0.1, "group": "Laboratory — Lipids & Liver"},

    # Anthropometrics
    "BMXWT":      {"label": "Body Weight (kg)",                    "type": "num", "min": 30.0, "max": 180.0, "default": 72.0, "step": 0.5, "group": "Body Measurements"},
    "BMXHT":      {"label": "Height (cm)",                         "type": "num", "min": 140.0,"max": 200.0, "default": 168.0,"step": 0.5, "group": "Body Measurements"},
    "BMXLEG":     {"label": "Upper Leg Length (cm)",               "type": "num", "min": 25.0, "max": 55.0,  "default": 38.0, "step": 0.5, "group": "Body Measurements"},
    "BMXARML":    {"label": "Upper Arm Length (cm)",               "type": "num", "min": 25.0, "max": 45.0,  "default": 36.0, "step": 0.5, "group": "Body Measurements"},
}

#  Page Layout 
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0A1931 0%, #15305B 60%, #003d52 100%);
    padding: 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: #fff;
    position: relative;
    overflow: hidden;
">
    <div style="
        position:absolute; top:-30px; right:-30px;
        width:180px; height:180px;
        background:rgba(0,210,196,0.07); border-radius:50%;
    "></div>
    <div style="
        display:inline-block; background:rgba(0,210,196,0.15);
        border:1px solid rgba(0,210,196,0.4);
        padding:4px 14px; border-radius:50px;
        font-size:0.76rem; font-weight:600; letter-spacing:1px;
        color:#00D2C4; text-transform:uppercase; margin-bottom:1rem;
    "> AI-Powered Diagnostic</div>
    <h1 style="color:#fff !important; font-size:2rem; font-weight:800; margin:0 0 0.6rem 0;">
        CKD Smart Diagnostic Tool
    </h1>
    <p style="color:#b8cfe8; font-size:1rem; margin:0; max-width:620px; line-height:1.6;">
        Enter patient clinical values to receive an AI-generated Chronic Kidney Disease
        risk assessment using our validated Stacking Ensemble model
        (Accuracy: <strong style="color:#00D2C4;">87.13%</strong> · AUC: <strong style="color:#00D2C4;">0.833</strong>).
    </p>
</div>
""", unsafe_allow_html=True)

#  Load model 
model_result = load_model()
model, feature_cols, imputer, scaler = model_result

if model is None:
    st.error(f" Could not load model: {feature_cols}")
    st.stop()


#  Input Form 
st.subheader(" Patient Clinical Information")
st.caption("Fill in the patient's laboratory and clinical values. All values are from routine check-up panels.")

# Group features by section
groups = {}
for feat in feature_cols:
    meta = FEATURE_META.get(feat, {"group": "Other"})
    grp = meta.get("group", "Other")
    groups.setdefault(grp, []).append(feat)

def clear_inputs():
    """Reset all input fields in st.session_state to None (blank)."""
    for feat in feature_cols:
        key = f"inp_{feat}"
        st.session_state[key] = None

def autofill_medians():
    """Populate all unentered fields with CDC population medians."""
    for feat in feature_cols:
        key = f"inp_{feat}"
        if st.session_state.get(key) is None:
            meta = FEATURE_META.get(feat, {})
            ftype = meta.get("type", "num")
            if ftype == "cat":
                opts = meta.get("options", {})
                default_key = meta.get("default", list(opts.keys())[0])
                st.session_state[key] = default_key
            else:
                default_num = meta.get("default", 0.0)
                st.session_state[key] = float(default_num)

# Collect user inputs across tabs
tab_labels = list(groups.keys())
tabs = st.tabs([f" {g}" for g in tab_labels])

raw_inputs = {}
user_inputs = {}

for tab, (grp_name, feats) in zip(tabs, groups.items()):
    with tab:
        n_cols = 3 if len(feats) > 4 else 2
        cols = st.columns(n_cols)
        for i, feat in enumerate(feats):
            meta = FEATURE_META.get(feat, {})
            col = cols[i % n_cols]
            with col:
                ftype = meta.get("type", "num")
                label = meta.get("label", feat)
                if ftype == "cat":
                    opts = meta.get("options", {"Yes": 1, "No": 2})
                    selected = st.selectbox(
                        label,
                        list(opts.keys()),
                        index=None,
                        placeholder="Select option...",
                        key=f"inp_{feat}"
                    )
                    raw_inputs[feat] = selected
                    if selected is not None and selected in opts:
                        user_inputs[feat] = opts[selected]
                else:
                    default_num = meta.get("default", 0.0)
                    val = st.number_input(
                        label,
                        min_value=float(meta.get("min", 0.0)),
                        max_value=float(meta.get("max", 999.0)),
                        value=None,
                        step=float(meta.get("step", 0.1)),
                        format="%.1f",
                        placeholder=f"e.g. {default_num}",
                        key=f"inp_{feat}"
                    )
                    raw_inputs[feat] = val
                    if val is not None:
                        user_inputs[feat] = float(val)

#  Predict & Clear Buttons 
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3, col_info = st.columns([1.5, 1.5, 1.0, 3.0])
with col_btn1:
    run_pred = st.button(" Run CKD Screening", type="primary", use_container_width=True)
with col_btn2:
    st.button(" Auto-Fill CDC Medians", on_click=autofill_medians, use_container_width=True)
with col_btn3:
    st.button(" Clear Form", on_click=clear_inputs, use_container_width=True)
with col_info:
    st.info(" This tool is for research demonstration only. Not a substitute for clinical diagnosis.")

#  Prediction Results 
if run_pred:
    missing_by_group = {}
    for feat, val in raw_inputs.items():
        if val is None:
            meta = FEATURE_META.get(feat, {})
            grp = meta.get("group", "Other")
            label = meta.get("label", feat)
            missing_by_group.setdefault(grp, []).append(label)

    if missing_by_group:
        num_missing = sum(len(v) for v in missing_by_group.values())
        num_filled = len(feature_cols) - num_missing
        st.markdown(f"""
        <div style="
            background: rgba(239, 68, 68, 0.08);
            border: 1px solid rgba(239, 68, 68, 0.4);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-top: 1.2rem;
            margin-bottom: 1rem;
        ">
            <h4 style="color: #ef4444; margin: 0 0 0.5rem 0; font-size: 1.1rem; font-weight: 700;">
                 Action Required: Missing Clinical Data ({num_filled}/{len(feature_cols)} Entered)
            </h4>
            <p style="margin: 0; font-size: 0.95rem; color: #b8cfe8; line-height: 1.5;">
                Please navigate to the specific tabs below and enter the missing clinical values, or click 
                <strong style="color: #00D2C4;">"Auto-Fill CDC Medians"</strong> to populate empty fields automatically.
            </p>
        </div>
        """, unsafe_allow_html=True)

        for grp_name, missing_labels in missing_by_group.items():
            formatted_fields = ", ".join([f"**{lbl}**" for lbl in missing_labels])
            st.warning(f"📌 **Go to `{grp_name}` Tab**: Missing {formatted_fields}")

        st.stop()

    # Build input DataFrame in correct column order
    input_row = {feat: [user_inputs.get(feat, 0.0)] for feat in feature_cols}
    input_df = pd.DataFrame(input_row)

    # Scale numerical columns
    _cat_feats = ['RIAGENDR', 'DMQMILIZ', 'DMDBORN4', 'DMDEDUC2', 'DMDMARTZ', 'DIQ010', 'DIQ160', 'DIQ180']
    num_cols_final = [col for col in feature_cols if col not in _cat_feats]

    input_num_imputed = imputer.transform(input_df[num_cols_final])
    input_df[num_cols_final] = scaler.transform(input_num_imputed)

    # Cast categoricals to int to match training dtype
    for col in _cat_feats:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(int)

    # Reorder columns to match feature_cols
    input_df = input_df[feature_cols]

    # Get probability and prediction
    prob = model.predict_proba(input_df)[0][1]   # P(CKD)
    pred = int(model.predict(input_df)[0])



    # Risk category
    if prob >= 0.65:
        risk_label = "HIGH RISK"
        risk_color = "#ef4444"
        risk_bg    = "rgba(239,68,68,0.1)"
        risk_class = "risk-card-high"
        risk_icon  = ""
        risk_desc  = "Strong probability of Chronic Kidney Disease. Immediate nephrology referral is advised."
    elif prob >= 0.40:
        risk_label = "MODERATE RISK"
        risk_color = "#f97316"
        risk_bg    = "rgba(249,115,22,0.1)"
        risk_class = "risk-card-medium"
        risk_icon  = ""
        risk_desc  = "Moderate CKD risk detected. Additional renal function tests and lifestyle monitoring recommended."
    else:
        risk_label = "LOW RISK"
        risk_color = "#22c55e"
        risk_bg    = "rgba(34,197,94,0.1)"
        risk_class = "risk-card-low"
        risk_icon  = ""
        risk_desc  = "Low probability of CKD detected. Routine annual screening continues to be advised."

    pct = int(prob * 100)

    st.markdown("---")
    st.subheader(" Screening Report")

    #  Top result cards
    rc1, rc2, rc3 = st.columns(3)

    with rc1:
        dx_label = "CKD Detected" if pred == 1 else "No CKD Detected"
        dx_color = "#ef4444" if pred == 1 else "#22c55e"
        st.markdown(f"""
        <div style="
            background:{risk_bg};
            border:2px solid {risk_color};
            border-radius:14px; padding:1.6rem; text-align:center;
        ">
            <div style="font-size:2.5rem;">{risk_icon}</div>
            <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:1.2px;
                        color:{risk_color}; font-weight:700; margin:8px 0 4px;">
                Screening Outcome
            </div>
            <div style="font-size:1.5rem; font-weight:800; color:{dx_color};">
                {dx_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with rc2:
        st.markdown(f"""
        <div style="
            background:var(--bg-card);
            border:1px solid var(--border-card);
            border-radius:14px; padding:1.6rem; text-align:center;
            box-shadow:var(--card-shadow);
        ">
            <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:1.2px;
                        color:var(--text-muted); font-weight:700; margin-bottom:10px;">
                CKD Probability Score
            </div>
            <div style="font-size:3rem; font-weight:800; color:{risk_color}; line-height:1;">
                {pct}%
            </div>
            <div style="font-size:0.78rem; color:var(--text-muted); margin-top:6px;">
                Model confidence (P = {prob:.4f})
            </div>
        </div>
        """, unsafe_allow_html=True)

    with rc3:
        st.markdown(f"""
        <div style="
            background:{risk_bg};
            border:2px solid {risk_color};
            border-radius:14px; padding:1.6rem; text-align:center;
        ">
            <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:1.2px;
                        color:{risk_color}; font-weight:700; margin-bottom:10px;">
                Risk Category
            </div>
            <div style="font-size:1.6rem; font-weight:800; color:{risk_color};">
                {risk_label}
            </div>
            <div style="font-size:0.78rem; color:var(--text-muted); margin-top:6px;">
                Threshold: ≥65% = High · 40-64% = Moderate · &lt;40% = Low
            </div>
        </div>
        """, unsafe_allow_html=True)

    #  Risk Meter (Gauge via HTML/CSS)
    st.markdown("<br>", unsafe_allow_html=True)

    meter_left = min(max(pct, 2), 98)   # clamp for visual accuracy
    st.markdown(f"""
    <div style="
        background:var(--bg-card);
        border:1px solid var(--border-card);
        border-radius:14px; padding:1.5rem;
        box-shadow:var(--card-shadow); margin-bottom:1.5rem;
    ">
        <div style="font-weight:600; color:var(--text-heading); margin-bottom:1rem; font-size:0.95rem;">
             CKD Risk Probability Meter
        </div>
        <!-- Track -->
        <div style="
            height: 28px; border-radius: 14px;
            background: linear-gradient(to right, #22c55e 0%, #f97316 45%, #ef4444 75%, #991b1b 100%);
            position: relative; overflow: visible; margin-bottom: 8px;
        ">
            <!-- Indicator needle -->
            <div style="
                position: absolute;
                left: {meter_left}%;
                top: -6px;
                transform: translateX(-50%);
                width: 4px;
                height: 40px;
                background: #ffffff;
                border-radius: 2px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.5);
                border: 1px solid rgba(0,0,0,0.3);
            "></div>
        </div>
        <!-- Labels -->
        <div style="display:flex; justify-content:space-between; font-size:0.72rem;
                    color:var(--text-muted); font-weight:600; letter-spacing:0.5px;">
            <span style="color:#22c55e;">0% — LOW</span>
            <span style="color:#f97316;">40% — MODERATE</span>
            <span style="color:#ef4444;">65% — HIGH</span>
            <span style="color:#991b1b;">100%</span>
        </div>
        <div style="text-align:center; margin-top:10px;
                    font-size:1.1rem; font-weight:700; color:{risk_color};">
            Patient Score: {pct}% — {risk_label}
        </div>
    </div>
    """, unsafe_allow_html=True)

    #  Clinical Interpretation & Recommendations 
    col_interp, col_recs = st.columns([1, 1])

    with col_interp:
        st.markdown(f"""
        <div class="custom-alert">
            <strong> Clinical Interpretation:</strong><br><br>
            {risk_desc}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("** Key Input Summary**")
        age_val = user_inputs.get("RIDAGEYR", "N/A")
        bun_val = user_inputs.get("LBXSBU", "N/A")
        ua_val  = user_inputs.get("LBXSUA", "N/A")
        sbp_val = user_inputs.get("BPX_SYS_MEAN", "N/A")
        diab_opts = {1: "Yes", 2: "No", 3: "Borderline"}
        diab_val  = diab_opts.get(int(user_inputs.get("DIQ010", 2)), "N/A")

        c_a, c_b = st.columns(2)
        with c_a:
            st.metric("Age", f"{int(age_val)} yrs")
            st.metric("BUN", f"{bun_val} mg/dL")
            st.metric("Uric Acid", f"{ua_val} mg/dL")
        with c_b:
            st.metric("Systolic BP", f"{int(sbp_val)} mmHg")
            st.metric("Diabetes", diab_val)
            st.metric("CKD Risk", f"{pct}%", delta=f"{risk_label}")

    with col_recs:
        st.markdown("** Clinical Recommendations**")

        if pred == 1:
            recs = [
                ("", "Nephrology Referral", "Schedule an immediate appointment with a nephrologist for further renal function tests."),
                ("", "eGFR & ACR Testing", "Measure estimated Glomerular Filtration Rate and Albumin-to-Creatinine Ratio for CKD staging."),
                ("", "Medication Review", "Evaluate current medications for nephrotoxic agents. Consider ACE inhibitors for BP control."),
                ("", "Dietary Guidance", "Reduce sodium, potassium, phosphorus, and protein intake under dietitian supervision."),
                ("", "Follow-up Monitoring", "Schedule monthly renal panels and blood pressure monitoring."),
            ]
        elif prob >= 0.40:
            recs = [
                ("", "Renal Panel Testing", "Request comprehensive metabolic panel with BUN and uric acid re-evaluation in 3 months."),
                ("", "BP Monitoring", "Monitor blood pressure weekly. Target systolic < 130 mmHg."),
                ("", "Lifestyle Changes", "Adopt DASH diet, reduce processed food intake, and maintain adequate hydration."),
                ("", "Weight Management", "If BMI > 25, initiate structured weight reduction program."),
                ("", "Annual Re-screening", "Repeat CKD screening in 6-12 months using this tool or clinical panels."),
            ]
        else:
            recs = [
                ("", "Routine Monitoring", "Continue annual health check-ups and blood panel monitoring."),
                ("", "Healthy Lifestyle", "Maintain a balanced diet, regular exercise, and adequate hydration."),
                ("", "Blood Pressure Control", "Keep blood pressure below 130/80 mmHg through lifestyle measures."),
                ("", "Blood Sugar Control", "If pre-diabetic or diabetic, maintain tight glycemic control (HbA1c < 7%)."),
                ("", "Avoid Nephrotoxins", "Limit NSAID use and avoid prolonged dehydration or nephrotoxic substances."),
            ]

        for icon, title, desc in recs:
            st.markdown(f"""
            <div style="
                display:flex; gap:12px; align-items:flex-start;
                margin-bottom:0.9rem; padding:0.75rem;
                background:var(--bg-card); border:1px solid var(--border-card);
                border-radius:10px; box-shadow:var(--card-shadow);
            ">
                <div style="font-size:1.3rem; min-width:32px; text-align:center;">{icon}</div>
                <div>
                    <div style="font-weight:600; font-size:0.88rem; color:var(--text-heading);">{title}</div>
                    <div style="font-size:0.8rem; color:var(--text-muted); line-height:1.4;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    #  Disclaimer
    st.markdown("""
    <div style="
        margin-top:1.5rem; padding:1rem 1.25rem;
        background:rgba(239,68,68,0.05);
        border:1px solid rgba(239,68,68,0.2);
        border-radius:10px; font-size:0.78rem; color:var(--text-muted);
        line-height:1.5;
    ">
         <strong>Disclaimer:</strong> This AI tool is developed for <em>research and educational demonstration</em> purposes only.
        It is based on the NHANES epidemiological dataset and is NOT validated for individual clinical decisions.
        Always consult a qualified nephrologist or physician for medical advice.
    </div>
    """, unsafe_allow_html=True)
