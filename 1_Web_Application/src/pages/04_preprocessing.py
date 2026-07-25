# src/pages/04_preprocessing.py
import streamlit as st

st.title(" Data Preprocessing & Leakage Prevention Pipeline")

# Flowchart representation using styled HTML cards and arrows
st.subheader(" The Clean Preprocessing Pipeline Flowchart")
st.markdown("""
<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; margin-bottom: 2rem;">
    <div style="background-color: #FFFFFF; color: #0A1931; border: 1px solid #CBD5E0; border-left: 5px solid #0A1931; padding: 12px 20px; border-radius: 10px; font-weight: bold; width: 340px; text-align: center; box-shadow: 0 4px 15px rgba(10,25,49,0.06);">
        1. RAW NHANES DATASETS<br><span style="font-size:0.82rem; font-weight:normal; color:#64748B;">5,552 Patients × 118 Features</span>
    </div>
    <div style="color: #00a8a1; font-weight: bold; font-size: 1.3rem;">↓</div>
    <div style="background-color: #FFFFFF; color: #0A1931; border: 1px solid #CBD5E0; border-left: 5px solid #15305B; padding: 12px 20px; border-radius: 10px; font-weight: bold; width: 340px; text-align: center; box-shadow: 0 4px 15px rgba(10,25,49,0.06);">
        2. DATA LEAKAGE CLEANUP<br><span style="font-size:0.82rem; font-weight:normal; color:#64748B;">Dropped eGFR, ACR, Creatinine, & IDs</span>
    </div>
    <div style="color: #00a8a1; font-weight: bold; font-size: 1.3rem;">↓</div>
    <div style="background-color: #FFFFFF; color: #0A1931; border: 1px solid #CBD5E0; border-left: 5px solid #15305B; padding: 12px 20px; border-radius: 10px; font-weight: bold; width: 340px; text-align: center; box-shadow: 0 4px 15px rgba(10,25,49,0.06);">
        3. HIGH-MISSING VALUE PRUNING<br><span style="font-size:0.82rem; font-weight:normal; color:#64748B;">Dropped columns with >40% NaNs</span>
    </div>
    <div style="color: #00a8a1; font-weight: bold; font-size: 1.3rem;">↓</div>
    <div style="background-color: #EBF2FA; color: #0A1931; border: 2px solid #00a8a1; padding: 12px 20px; border-radius: 10px; font-weight: bold; width: 350px; text-align: center; box-shadow: 0 4px 15px rgba(0,168,161,0.12);">
         STRATIFIED TRAIN-TEST SPLIT<br><span style="font-size:0.82rem; font-weight:600; color:#00a8a1;">Isolated Test Split before Imputing</span>
    </div>
    <div style="color: #00a8a1; font-weight: bold; font-size: 1.3rem;">↓</div>
    <div style="background-color: #FFFFFF; color: #0A1931; border: 1px solid #CBD5E0; border-left: 5px solid #15305B; padding: 12px 20px; border-radius: 10px; font-weight: bold; width: 340px; text-align: center; box-shadow: 0 4px 15px rgba(10,25,49,0.06);">
        4. IMPUTATION & SCALING<br><span style="font-size:0.82rem; font-weight:normal; color:#64748B;">Median lab numbers / Mode categories</span>
    </div>
    <div style="color: #00a8a1; font-weight: bold; font-size: 1.3rem;">↓</div>
    <div style="background-color: #FFFFFF; color: #0A1931; border: 1px solid #CBD5E0; border-left: 5px solid #15305B; padding: 12px 20px; border-radius: 10px; font-weight: bold; width: 340px; text-align: center; box-shadow: 0 4px 15px rgba(10,25,49,0.06);">
        5. PAIRWISE CORRELATION FILTER<br><span style="font-size:0.82rem; font-weight:normal; color:#64748B;">Removed collinear features > 0.85</span>
    </div>
    <div style="color: #00a8a1; font-weight: bold; font-size: 1.3rem;">↓</div>
    <div style="background-color: #FFFFFF; color: #0A1931; border: 1px solid #CBD5E0; border-left: 5px solid #15305B; padding: 12px 20px; border-radius: 10px; font-weight: bold; width: 340px; text-align: center; box-shadow: 0 4px 15px rgba(10,25,49,0.06);">
        6. SMOTE BALANCING (TRAIN)<br><span style="font-size:0.82rem; font-weight:normal; color:#64748B;">Simulated minority risk cases</span>
    </div>
    <div style="color: #00a8a1; font-weight: bold; font-size: 1.3rem;">↓</div>
    <div style="background-color: #FFFFFF; color: #0A1931; border: 1px solid #CBD5E0; border-left: 5px solid #0A1931; padding: 12px 20px; border-radius: 10px; font-weight: bold; width: 340px; text-align: center; box-shadow: 0 4px 15px rgba(10,25,49,0.06);">
        7. FINAL MODELING SPLIT<br><span style="font-size:0.82rem; font-weight:normal; color:#64748B;">4,441 Train (Balanced: 7,364) / 1,111 Test</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Step-by-Step Pipeline Details
st.subheader(" Detailed Step Explanations")

exp1 = st.expander(" Step 1 & 2: Dataset Merging & Leakage Removal", expanded=False)
with exp1:
    st.markdown("""
    *   **Merging:** CDC releases NHANES as individual tables (`DEMO_J.xpt`, `BIOPRO_J.xpt`, `BPX_J.xpt` etc.) containing specific files. We did a clean inner join using the sequence number `SEQN`.
    *   **Leakage Variables Dropped:** eGFR (estimated Glomerular Filtration Rate), ACR (Albumin-to-Creatinine Ratio), standard creatinine indicators (`LBXSCR`), and urinary albumin (`URXUMA`).
    *   **Why this matters:** eGFR and ACR are the actual mathematical rules clinicians use to define CKD. If they are kept in features, any ML algorithm can achieve 100% classification by backing out those formulas. In clinical screening, we want to flag CKD *before* renal specialists calculate complex ratios/eGFR, using simpler biochemical blood panels.
    """)

exp2 = st.expander(" Step 3: Column Pruning (>40% Missing)", expanded=False)
with exp2:
    st.markdown("""
    *   *Rule:* We remove columns that have more than 40% missing entries.
    *   *Aesthetic Rationale:* Imputing values for variables where ~half of the patients didn't have tests introduces heavy artificial bias. It's better to focus on data features that are dense enough to display reliable clinical patterns.
    """)

exp3 = st.expander(" Train-Test Split Separation (Zero-Leakage Guard)", expanded=True)
with exp3:
    st.markdown("""
    *   *Sequence:* We performed the **Train-Test split FIRST** before scaling, imputing, or feature selection.
    *   *Standard Pitfall:* Frequently, developers impute missing values or scale the dataset globally, and then split. This leaks the mean/median/distribution profiles of the validation split into training data, skewing accuracy metrics on paper while failing in staging. 
    *   *Solution:* Preprocessing rules were exclusively fitted on `X_train_raw` and then applied to `X_test_raw` in transform-only mode.
    """)

exp4 = st.expander(" Step 4 & 5: Numerical Imputation, Scaling, and Collinearity Filter", expanded=False)
with exp4:
    st.markdown("""
    *   **Numerical Imputation:** Uses `SimpleImputer` configured with `median` calculations. Median is chosen over mean to immunize imputation against clinical outliers.
    *   **Standardization:** StandardScaler is applied to ensure distance-based models (e.g. KNN, SVM) and neural networks are not biased by unit scale differences (e.g. Cholesterol in mg/dL vs age in years).
    *   **Pairwise Correlation Filter:** Standard Pearson correlation matrix is fit on X_train. For any two variables sharing correlation $> 0.85$, the one showing lower correlation with target CKD is dropped. This removes redundant indicators like duplicate blood measurements.
    """)

exp5 = st.expander(" Step 6: SMOTE Balancing (Minority Risk Oversampling)", expanded=True)
with exp5:
    st.markdown("""
    *   **Methodology:** **Synthetic Minority Over-sampling Technique (SMOTE)** analyzes minority instances (CKD patients) and draws synthetic line segments connecting their nearest neighbors to create new representative patient records.
    *   *Clinical Rationale:* It prevents the classifier from building a lazy "always predict healthy" bias, ensuring that the model has high sensitivity towards recognizing CKD markers.
    """)
