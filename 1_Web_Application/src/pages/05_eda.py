# src/pages/05_eda.py
import streamlit as st
import os

st.title(" Exploratory Data Analysis (EDA)")

# Dynamic path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(os.path.dirname(script_dir), "assets")

st.markdown("""
###  Investigating Clinical Feature Associations
Before fitting machine learning models, exploratory data analysis is used to discover how demographic attributes, examination charts, and biochemical blood values correlate with CKD.
""")

tab_plots, tab_correlations = st.tabs([" Diagnostic Density Plots", " Feature Correlations & Heatmap"])

with tab_plots:
    # Age & Diabetes Columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Age Distribution in CKD vs Non-CKD")
        img_age = os.path.join(assets_dir, "EDA_02_Age_Distribution_CKD_43(new).png")
        if os.path.exists(img_age):
            st.image(img_age, use_container_width=True)
        else:
            st.warning("Age distribution plot missing.")
        st.markdown("""
        **Clinical Interpretation:**
        - Healthy/Non-CKD distributions peak at much younger demographics.
        - CKD-positive counts drastically skew towards the elderly ($>60$ years). This occurs because kidneys undergo cumulative nephron scarring and natural GFR decay over life.
        """)
        
    with col2:
        st.subheader(" Diabetes Prevalence vs CKD Status")
        img_diab = os.path.join(assets_dir, "EDA_03_Diabetes_vs_CKD_43(new).png")
        if os.path.exists(img_diab):
            st.image(img_diab, use_container_width=True)
        else:
            st.warning("Diabetes comparison plot missing.")
        st.markdown("""
        **Clinical Interpretation:**
        - Diabetic patients (Status 1/2) show a significantly higher proportion of CKD.
        - Diabetic nephropathy is a well-established disease pathway where chronic hyper-glycemia thickens glomerular membranes, causing protein leaks and filtration failure.
        """)

    st.markdown("---")

    # BUN & Uric Acid Columns
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader(" Blood Urea Nitrogen (BUN) Distributions")
        img_bun = os.path.join(assets_dir, "EDA_04_BUN_Boxplot_43(new).png")
        if os.path.exists(img_bun):
            st.image(img_bun, use_container_width=True)
        else:
            st.warning("BUN boxplot missing.")
        st.markdown("""
        **Clinical Interpretation:**
        - BUN measures the amount of nitrogen in blood coming from urea (protein waste).
        - The boxplot displays elevated ranges and significant outliers for CKD patients, indicating that damaged kidneys fail to clear nitrogenous waste, leaving urea circulating.
        """)
        
    with col4:
        st.subheader(" Uric Acid Violin Distribution")
        img_uric = os.path.join(assets_dir, "EDA_05_UricAcid_Violin_43(new).png")
        if os.path.exists(img_uric):
            st.image(img_uric, use_container_width=True)
        else:
            st.warning("Uric Acid violin plot missing.")
        st.markdown("""
        **Clinical Interpretation:**
        - Excessive uric acid (hyperuricemia) is both a cause and biomarker of CKD.
        - The violin plot confirms that the CKD density is wider at higher uric acid values, indicating systemic metabolic kidney burden.
        """)

    st.markdown("---")
    
    # Blood Pressure Column
    col5, col5_text = st.columns([2.0, 1.5])
    with col5:
        st.subheader(" Mean Systolic Blood Pressure Boxplot")
        img_bp = os.path.join(assets_dir, "EDA_06_BP_Boxplot_43(new).png")
        if os.path.exists(img_bp):
            st.image(img_bp, use_container_width=True)
        else:
            st.warning("Systolic BP boxplot missing.")
    with col5_text:
        st.markdown("""
        <br><br>
        **Clinical Interpretation:**
        - Displays higher average values for the CKD class.
        - High blood pressure causes arteriolosclerosis in renal vessels, restricting blood flow. In turn, damaged kidneys release hormone triggers (renin) that further raise blood pressure, creating a dangerous feedback loop.
        """, unsafe_allow_html=True)

with tab_correlations:
    col_heat_plot, col_heat_desc = st.columns([2.2, 1.5])
    
    with col_heat_plot:
        st.subheader(" Features Correlation Heatmap")
        img_corr = os.path.join(assets_dir, "EDA_07_Correlation_Heatmap_43(new).png")
        if os.path.exists(img_corr):
            st.image(img_corr, use_container_width=True)
        else:
            st.warning("Correlation heatmap missing.")
            
    with col_heat_desc:
        st.markdown("""
        **Linear Association Assessment:**
        - The heatmap displays pairwise Pearson correlations between the clinical features and target CKD.
        - Laboratory biomarkers (BUN, Uric Acid, blood panels) and Age express the strongest positive correlations.
        - Physical indicators (BMI, arm length, height) possess weaker linear associations, confirming that lab panels are the primary predictive signals for early-stage screening.
        - Pairwise collinear features have been actively pruned in preprocessing (filtered above 0.85 threshold) to prevent model parameter redundancy and over-fitting.
        """)
