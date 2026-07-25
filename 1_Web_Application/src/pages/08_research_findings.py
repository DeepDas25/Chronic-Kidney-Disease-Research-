# src/pages/08_research_findings.py
import streamlit as st
import os
import pandas as pd

st.title(" Major Research Findings & Clinical Insights")

# Dynamic path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(os.path.dirname(script_dir), "assets")

st.markdown("""
###  Key Clinical Takeaways from the Research
This research bridges the gap between machine learning and nephrology, proving that standard clinic measurements can be used for early-stage screening of Chronic Kidney Disease.
""")

col_main, col_cont = st.columns([2.0, 1.5])

with col_main:
    st.subheader(" Major Scientific Conclusions")
    
    st.markdown("""
    *   **The Power of Laboratory Diagnostics:** The model highlights biochemical markers like **Blood Urea Nitrogen (BUN)**, **Uric Acid**, and **Iron (Serum)** as leading predictors of CKD, outperforming standard subjective evaluations (e.g. self-reported family histories).
    *   **Interactive Pathways:** While diabetes code (`DIQ010`) and hypertension variables (`BPX_SYS_MEAN`) are well-established conditions, the model shows that their predictive power surges when they co-occur with minor age-related filtration declines.
    *   **Ensuring Clinician Trust:** By integrating SHAP, we demonstrate that the model operates on authentic physiological relationships. It does not map predictions to statistical noise, but maps them to biological indicators (elevated BUN, increased systolic blood pressure, high blood sugar).
    """)
    
    # Check if counterfactuals table exists
    st.subheader(" Actionable Patient Counterfactual Recommendations")
    st.write("""
    Counterfactual explanations provide actionable pathways. They answer: *"What must a patient change (e.g. blood pressure, cholesterol, BMI) to decrease their predicted CKD risk classification from High Risk to Low Risk?"*
    """)
    
    csv_cf = os.path.join(assets_dir, "counterfactual_recommendations.csv")
    if os.path.exists(csv_cf):
        try:
            df_cf = pd.read_csv(csv_cf)
            st.dataframe(df_cf, height=220, use_container_width=True)
            st.markdown("""
            *Interpretation:* The table above (generated from our research pipeline) showcases mock counterfactual cases indicating exact target levels (e.g. reducing Systolic BP by $x$ mmHg or controlling blood glucose values) required to transition patient classifications to safety.
            """)
        except Exception:
            st.warning("Failed to render counterfactual recommendations table. Asset file format mismatch.")
    else:
        st.warning("Counterfactual recommendations file not found.")

with col_cont:
    st.subheader(" Key Contributions of this Project")
    
    st.markdown("""
    <div class="contrib-card" style="border-left: 5px solid #0A1931;">
        <h4 style="color: var(--text-heading) !important;">1. Strict Data Leakage Elimination</h4>
        <p>Unlike many published studies that include eGFR or serum creatinine directly in models, our model was strictly evaluated on non-leakage precursors, validating its early-stage screening utility.</p>
    </div>
    <div class="contrib-card" style="border-left: 5px solid #00D2C4;">
        <h4 style="color: var(--blue-accent) !important;">2. Dual Balancing for Model Sensitivity</h4>
        <p>Using SMOTE on train sets, we forced the decision trees to identify silent clinical anomalies, boosting recall significantly over native, imbalanced baselines.</p>
    </div>
    <div class="contrib-card" style="border-left: 5px solid #15305B;">
        <h4 style="color: var(--text-subheading) !important;">3. Clinician-Centric Explainable Framework</h4>
        <p>Incorporating local &amp; global SHAP values ensures the system acts as a transparent, clinical assistant tool, boosting adoption confidence.</p>
    </div>
    """, unsafe_allow_html=True)

