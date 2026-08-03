# src/pages/01_home.py
import streamlit as st

#  Hero Banner 
st.markdown("""
<div style="background: linear-gradient(135deg, #0A1931 0%, #15305B 55%, #1a4a7a 100%); padding: 3rem 2.5rem 2.5rem 2.5rem; border-radius: 16px; margin-bottom: 2rem; color: #ffffff; position: relative; overflow: hidden;">
    <div style="position: absolute; top: -40px; right: -40px; width: 200px; height: 200px; background: rgba(0,210,196,0.08); border-radius: 50%;"></div>
    <div style="position: absolute; bottom: -60px; right: 80px; width: 150px; height: 150px; background: rgba(0,210,196,0.05); border-radius: 50%;"></div>
    <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(0,210,196,0.15); border: 1px solid rgba(0,210,196,0.4); padding: 5px 14px; border-radius: 50px; font-size: 0.78rem; font-weight: 600; letter-spacing: 1px; color: #00D2C4; text-transform: uppercase; margin-bottom: 1.2rem;"> Research Project Showcase</div>
    <h1 style="color: #ffffff !important; font-size: 2.4rem; font-weight: 800; margin: 0 0 0.75rem 0; font-family: Outfit, sans-serif; line-height: 1.2;">
        Machine Learning for Early Detection of<br>
        <span style="color: #00D2C4;">Chronic Kidney Disease</span>
    </h1>
    <p style="font-size: 1.05rem; color: #b8cfe8; margin-bottom: 1.75rem; font-weight: 300; max-width: 680px; line-height: 1.6;">
        Leveraging high-dimensional NHANES epidemiological data and a 11-model
        Stacking Ensemble to build an explainable, clinical-grade CKD screening system.
    </p>
    <div style="display: flex; gap: 12px; flex-wrap: wrap;">
        <div style="background: rgba(0,210,196,0.12); border: 1px solid rgba(0,210,196,0.35); padding: 8px 18px; border-radius: 8px; font-size: 0.88rem; font-weight: 500; color: #ffffff;"> NHANES Dataset (CDC)</div>
        <div style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.2); padding: 8px 18px; border-radius: 8px; font-size: 0.88rem; font-weight: 500; color: #ffffff;"> 11+ ML Models Compared</div>
        <div style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.2); padding: 8px 18px; border-radius: 8px; font-size: 0.88rem; font-weight: 500; color: #ffffff;"> SHAP Explainability</div>
        <div style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.2); padding: 8px 18px; border-radius: 8px; font-size: 0.88rem; font-weight: 500; color: #ffffff;"> Smart Diagnostic Tool</div>
    </div>
</div>
""", unsafe_allow_html=True)


#  Statistics Cards 
st.subheader(" Project at a Glance")
c1, c2, c3, c4 = st.columns(4)

stats = [
    ("5,552", "NHANES Patients", "Raw clinical records from CDC's gold-standard survey"),
    ("43", "Final Features", "Post leakage-removal and correlation filtering"),
    ("11+", "Models Evaluated", "Tree, boosting, SVM, Bayesian & stacking ensemble"),
    ("87.13%", "Best Accuracy", "Stacking Meta-LR · ROC-AUC: 0.833"),
]

for col, (num, label, desc) in zip([c1, c2, c3, c4], stats):
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-num">{num}</div>
            <div class="stat-label">{label}</div>
            <div class="metric-desc" style="margin-top:6px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

#  Project Summary + Key Objectives 
col_summary, col_obj = st.columns([1.6, 1.4])

with col_summary:
    st.subheader(" Project Summary")
    st.markdown("""
    <div class="custom-alert" style="margin-bottom: 1rem;">
        <strong> CKD Awareness:</strong> Chronic Kidney Disease is often a
        <em>"silent killer"</em> — showing no symptoms until advanced stages. Early
        ML-powered screening can prevent irreversible end-stage renal disease (ESRD).
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    This research project demonstrates a **complete, production-ready ML pipeline** for
    early CKD risk stratification using the **CDC NHANES dataset** — a nationally
    representative survey of 5,552 Americans including lab tests, physical exams, and
    demographic questionnaires.

    The pipeline rigorously addresses **data leakage** (by excluding eGFR/ACR/Creatinine),
    handles **class imbalance** (via SMOTE), and delivers transparent results through
    **SHAP-based Explainable AI** — making it suitable for clinical trust and adoption.
    """)

with col_obj:
    st.subheader(" Research Objectives")
    objectives = [
        ("", "Clinical Need", "CKD affects 850M+ people globally; 90% are undiagnosed in early stages"),
        ("", "ML Pipeline", "Build a leakage-free, validated screening system on real-world epidemiological data"),
        ("", "Explainability", "Ensure clinicians can trust model decisions using SHAP attribution"),
        ("", "Deployment Ready", "Create an interactive prediction portal for clinical-grade risk screening"),
    ]
    for icon, title, desc in objectives:
        st.markdown(f"""
        <div style="display:flex; gap:12px; align-items:flex-start; margin-bottom:0.9rem;">
            <div style="
                font-size:1.5rem; min-width:40px; height:40px;
                display:flex; align-items:center; justify-content:center;
                background: rgba(0,210,196,0.12);
                border-radius:8px;
            ">{icon}</div>
            <div>
                <div style="font-weight:600; color:var(--text-heading); font-size:0.9rem;">{title}</div>
                <div style="color:var(--text-muted); font-size:0.82rem; line-height:1.4;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

#  Key Contributions 
st.subheader(" Key Contributions")

kc1, kc2, kc3, kc4 = st.columns(4)

contributions = [
    (kc1, "", "#0A1931", "Leakage-Free Pipeline",
     "Strictly excluded eGFR, ACR, and Serum Creatinine — the variables clinicians use to define CKD — preventing artificial inflation of accuracy."),
    (kc2, "", "#00D2C4", "Ensemble Learning",
     "Meta-Logistic Regression stacking across 11 base classifiers produces the most balanced precision-recall trade-off for clinical screening."),
    (kc3, "", "#15305B", "Explainable AI (SHAP)",
     "Global & local SHAP beeswarm plots map model decisions to real biomarkers (BUN, Uric Acid, Age), enabling clinician trust and adoption."),
    (kc4, "", "#00a8a1", "Early CKD Screening",
     "Achieves 87.13% accuracy and 0.833 ROC-AUC using only pre-diagnostic lab panels available at routine check-ups — no specialist tests required."),
]

for col, icon, color, title, desc in contributions:
    with col:
        st.markdown(f"""
        <div class="contrib-card" style="border-left: 5px solid {color};">
            <div style="font-size:1.8rem; margin-bottom:0.6rem;">{icon}</div>
            <h4 style="color:{color} !important; margin:0 0 8px 0; font-size:0.95rem; font-weight:700;">{title}</h4>
            <p style="color:var(--text-card-muted); font-size:0.82rem; line-height:1.5; margin:0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

#  Team & Guide 
st.subheader(" Research Team & Guidance")

col_team1, col_team2, col_team3, col_guide = st.columns(4)

members = [
    (col_team1, "Deepdas Somani"),
    (col_team2, "Viraj Kumar Sahu"),
    (col_team3, "Ayan Bhandari"),
]

for col, name in members:
    with col:
        st.markdown(f"""
        <div class="team-card">
            <h4>{name}</h4>
        </div>
        """, unsafe_allow_html=True)

with col_guide:
    st.markdown("""
    <div class="guide-card">
        <h4>Divanshu Jain</h4>
        <p>Project Guide &amp; Research Mentor</p>
    </div>
    """, unsafe_allow_html=True)
