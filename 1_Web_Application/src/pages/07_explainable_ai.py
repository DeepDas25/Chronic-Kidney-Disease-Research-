# src/pages/07_explainable_ai.py
import streamlit as st
import os
import pandas as pd

st.title(" Explainable AI (SHAP Interpretability)")

# Dynamic path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(os.path.dirname(script_dir), "assets")

st.markdown("""
###  Opening the Black Box in Healthcare
In medicine, accuracy is not enough. Clinicians must understand **why** an AI model flags a patient for CKD. 
We integrate **SHAP (SHapley Additive exPlanations)**, built on cooperative game theory, which assigns each clinical feature an attribution value (SHAP value) representing its contribution to a particular diagnostic decision.
""")

col1, col2 = st.columns([2.0, 1.5])

with col1:
    st.subheader(" Global SHAP Summary Plot")
    img_summary = os.path.join(assets_dir, "SHAP_SummaryPlot_Final43.png")
    if os.path.exists(img_summary):
        st.image(img_summary, use_container_width=True, caption="SHAP Beeswarm distribution for the Top 20 features")
    else:
        st.warning("SHAP Summary plot missing in assets.")

with col2:
    st.subheader(" How to Interpret a SHAP Beeswarm Plot")
    st.markdown("""
    Reading the SHAP beeswarm plot correctly reveals which features drive the model's decisions and in which direction:
    
    1.  **Feature Ranking:** Features are listed along the Y-axis in decreasing order of global importance. The top feature represents the strongest overall predictor.
    2.  **SHAP Value (X-Axis):** If the SHAP value is positive ($>0$), this feature **increases** the likelihood of a CKD diagnosis. If negative ($<0$), it represents a **preventative** effect (reducing CKD prediction risk).
    3.  **Color Scale (Feature Value):** **Red** represents a high value of the patient variable; **Blue** represents a low value.
    
    ** Examples to cite:**
    -   **Age (`RIDAGEYR`):** High values (Red points) skew heavily to the right. This means **older age increases CKD risk**.
    -   **Blood Urea Nitrogen (`LBXSBU`):** High concentration (Red) shifts to the right, showing **elevated urea levels indicate renal failure**.
    -   **Doctor told you have Diabetes (`DIQ010`):** High values (indicating yes, coded values) push predicted probability higher.
    """)

st.markdown("---")

col_bar, col_table = st.columns([2.0, 1.5])

with col_bar:
    st.subheader(" Feature Importance Bar Plot")
    img_bar = os.path.join(assets_dir, "SHAP_BarPlot_Final43.png")
    if os.path.exists(img_bar):
        st.image(img_bar, use_container_width=True, caption="Mean absolute SHAP value representing feature importance weights")
    else:
        # Check fallback names
        img_top10 = os.path.join(assets_dir, "Top10_SHAP_Features_Final43.png")
        if os.path.exists(img_top10):
            st.image(img_top10, use_container_width=True, caption="Top SHAP feature weights")
        else:
            st.warning("SHAP bar plot missing in assets.")

with col_table:
    st.subheader(" Top 20 SHAP Feature Weights Table")
    csv_shap = os.path.join(assets_dir, "SHAP_Top20_Features_Final43.csv")
    if os.path.exists(csv_shap):
        df_shap = pd.read_csv(csv_shap)
        df_shap["Importance"] = df_shap["Importance"].round(4)
        st.dataframe(df_shap, height=950, use_container_width=True)
    else:
        st.warning("SHAP Top 20 CSV not found.")
        st.write("""
        1. **RIDAGEYR** (Age) — *Weight: 0.470*
        2. **LBXSBU** (Blood Urea Nitrogen) — *Weight: 0.232*
        3. **LBXSUA** (Uric Acid) — *Weight: 0.178*
        4. **DIQ010** (Diabetes Indicator) — *Weight: 0.171*
        5. **BPX_SYS_MEAN** (Systolic BP Mean) — *Weight: 0.165*
        6. **LBXSIR** (Iron) — *Weight: 0.137*
        7. **LBXSCH** (Cholesterol) — *Weight: 0.123*
        8. **INDFMPIR** (Poverty Income Ratio) — *Weight: 0.111*
        """)
