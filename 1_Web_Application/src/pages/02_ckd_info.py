# src/pages/02_ckd_info.py
import streamlit as st

st.title(" Chronic Kidney Disease (CKD) Clinical Education")


# Main layout split
col_info, col_space, col_stages = st.columns([1.8, 0.1, 1.8])

with col_info:
    st.subheader(" What is Chronic Kidney Disease?")
    st.write("""
    **Chronic Kidney Disease (CKD)** refers to the progressive and irreversible loss of renal function over a period of months or years. The kidneys filter waste products and excess water from the blood, regulate blood pressure, aid red blood cell synthesis, and maintain electrolyte balance. When damaged, metabolic wastes build up in the blood, leading to systemic complications.
    """)
    
    st.subheader(" Common Causes & Risk Factors")
    st.markdown("""
    *   **Diabetes (Type 1 & 2):** High blood glucose damages the tiny filtering units (nephrons) inside the kidneys. This accounts for ~44% of new CKD cases.
    *   **Hypertension (High Blood Pressure):** Uncontrolled pressure damages blood vessels in the kidneys, limiting filtration capacity. Accounts for ~28% of cases.
    *   **Cardiovascular Disease:** Shared metabolic risks link heart and kidney disease.
    *   **Ageing & Family History:** Kidneys lose function naturally after age 40, and genetics play a major hereditary role.
    """)
    
    st.subheader(" Silent Symptoms of Progression")
    st.write("""
    In early stages (1 to 3), CKD is usually **asymptomatic**. As kidney capacity degrades to less than 25%, critical symptoms appear:
    """)
    col_sym1, col_sym2 = st.columns(2)
    with col_sym1:
        st.markdown("""
        -  **Edema:** Swelling in feet, ankles, hands due to fluid retention.
        -  **Chronic Fatigue:** Decreased EPO hormones leading to anemia.
        -  **Urination Changes:** Bubbly, frothy, or dark red urine.
        """)
    with col_sym2:
        st.markdown("""
        -  **Uremia:** Nausea, metallic taste, and lack of appetite.
        -  **Dyspnea:** Shortness of breath from fluid accumulation in lungs.
        -  **Hypertension Spike:** Fluid volume increase raising BP.
        """)

with col_stages:
    st.subheader(" Clinical Classification: The 5 Stages of CKD")
    st.write("""
    CKD is clinically staged based on the **estimated Glomerular Filtration Rate (eGFR)**, representing the volume of blood filtered per minute by the glomeruli.
    """)
    
    # Render table
    stages_data = {
        "Stage": ["Stage 1", "Stage 2", "Stage 3A", "Stage 3B", "Stage 4", "Stage 5"],
        "eGFR Range (mL/min/1.73m²)" : [">= 90", "60 - 89", "45 - 59", "30 - 44", "15 - 29", "< 15 (or Dialysis)"],
        "Diagnosis": ["Mild kidney damage, normal/high filtration", "Mild loss of kidney function", "Mild-to-moderate loss of function", "Moderate-to-severe loss of function", "Severe loss of kidney function", "End-Stage Renal Disease (ESRD) / Kidney Failure"]
    }
    st.table(stages_data)
    
    st.markdown("""
     **Why Early Detection Matters:**
    - Delaying kidney function decline is possible **only if diagnosed at Stages 1-3** via dietary changes, blood sugar control, and ACE inhibitor medications.
    - Pathways leading to Stage 5 (ESRD) result in a critical reliance on **dialysis or organ transplants**, leading to astronomical healthcare costs and heavily shortened life expectancies.
    """)
    
    st.subheader(" Global Prevalence Statistics")
    st.markdown("""
    - **Prevalence:** CKD affects over **850 million people** worldwide.
    - **Ignorance Gap:** Up to **90% of individuals** with early-stage CKD are completely unaware they have the condition because of the lack of early symptoms.
    - **Mortality:** CKD is currently the 12th leading cause of death globally and is projected to become the **5th leading cause of death by 2040** if left unmanaged.
    """)
