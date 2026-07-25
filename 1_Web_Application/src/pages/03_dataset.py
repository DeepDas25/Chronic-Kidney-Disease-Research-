# src/pages/03_dataset.py
import streamlit as st
import os
import pandas as pd

st.title(" CDC NHANES Dataset & Class Distribution")

# Dynamic paths resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(os.path.dirname(script_dir), "assets")

st.markdown("""
###  National Health and Nutrition Examination Survey (NHANES)
The database for this study is sourced from the **National Health and Nutrition Examination Survey (NHANES)**, conducted by the CDC's National Center for Health Statistics. 
NHANES combines interviews (demographics, dietary survey, socioeconomic variables), physical examinations, and chemical laboratory measurements of blood/urine.
""")

col_desc, col_plot = st.columns([1.5, 2.0])

with col_desc:
    st.subheader(" Dataset Dimension History")
    
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">RAW RECORDS (NHANES)</div>
        <div class="metric-val">5,552 × 118</div>
        <div style="font-size:0.8rem; color:#666;">Original size before administrative variables, identifiers, and data leakage channels were removed.</div>
    </div>
    <div class="metric-card accent">
        <div class="metric-label">PROCESSED SUBSET (CLEAN)</div>
        <div class="metric-val">4,441 × 43</div>
        <div style="font-size:0.8rem; color:#666;">Training set dimensions after dropping high-missing columns, correlation filters, and rows containing missing targets.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader(" Class Distribution (Raw)")
    # Raw value counts table
    st.markdown("""
    *   **Class 0 (Non-CKD):** 4,603 patients (**82.9%**)
    *   **Class 1 (CKD):** 949 patients (**17.1%**)
    *   *Observation:* This represents a **severe class imbalance**, which is typical of epidemiology surveys where the general population healthy subset dominates the target class. If left unmanaged, the machine learning models would favor the majority class, leading to terrible recall (failing to flag true CKD cases).
    """)

with col_plot:
    st.subheader(" Target Variable Distribution Chart")
    dist_img = os.path.join(assets_dir, "EDA_01_CKD_Class_Distribution_43(new).png")
    if os.path.exists(dist_img):
        st.image(dist_img, caption="CKD Target Class Distribution (Pie & Bar Graph showing the ~83% vs 17% Imbalance)", use_container_width=True)
    else:
        st.warning("Class distribution image missing. Check asset paths.")

st.markdown("---")

# Data Splits Table
st.subheader(" Dataset Splits & Training Configuration")
split_data = {
    "Split Name": ["Raw (NHANES)", "Train Split (Raw Imbalanced)", "Test Split (Holdout)", "Train Split (SMOTE Balanced)"],
    "Records Count (Rows)": [5552, 4441, 1111, 7364],
    "CKD Cases (Class 1)": [949, 759, 190, 3682],
    "Non-CKD (Class 0)": [4603, 3682, 921, 3682],
    "Balanced Ratio": ["Imbalanced (17:83)", "Imbalanced (17:83)", "Imbalanced (Holdout Benchmark)", "Balanced (50:50)"]
}
st.table(split_data)

st.info("""
**Why we balanced the Training Split but left the Testing Split imbalanced:**
To train the model to successfully recognize minority features, the training split must be balanced (achieved using **SMOTE**). However, the **testing/evaluation holdout split must remain imbalanced** (matching original population distribution) to verify how the model performs in a real-world, clinical setting.
""")
