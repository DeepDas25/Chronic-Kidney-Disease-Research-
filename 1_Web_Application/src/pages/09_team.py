# src/pages/09_team.py
import streamlit as st

st.title(" Research Team & Implementation Timeline")

col1, col2 = st.columns([1.8, 2.0])

with col1:
    st.subheader(" Team Members & Roles")
    
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 12px; margin-bottom: 1.5rem;">
        <div class="team-card" style="text-align: left;">
            <h4 style="color: #0A1931 !important;">Deepdas Somani</h4>
            # <p style="color: #00a8a1; font-weight: 600; margin-top: 2px;">Data Engineering &amp; Pipeline Specialist</p>
            # <p style="color: #4A5568; font-size: 0.8rem; margin-top: 4px;">Led CDC NHANES dataset merging, zero-leakage split checkpoints, and SMOTE resampling.</p>
        </div>
        <div class="team-card" style="text-align: left;">
            <h4 style="color: #0A1931 !important;">Viraj Kumar Sahu</h4>
            # <p style="color: #00a8a1; font-weight: 600; margin-top: 2px;">ML Scientist &amp; Stacking Architect</p>
            # <p style="color: #4A5568; font-size: 0.8rem; margin-top: 4px;">Engineered the Meta-Logistic Regression Stacking framework, cross-validations, and evaluation pipelines.</p>
        </div>
        <div class="team-card" style="text-align: left;">
            <h4 style="color: #0A1931 !important;">Ayan Bhandari</h4>
            # <p style="color: #00a8a1; font-weight: 600; margin-top: 2px;">Research Developer &amp; Dashboard Architect</p>
            # <p style="color: #4A5568; font-size: 0.8rem; margin-top: 4px;">Designed clinical timeline layouts, literature review synthesis drafts, and Streamlit showcase web hub.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader(" Project Guidance")
    st.markdown("""
    <div class="guide-card" style="text-align: left; margin-bottom: 1.5rem;">
        <h4 style="color: #0A1931 !important;">Divanshu Jain</h4>
        # <p style="color: #00a8a1; font-weight: 600; margin-top: 2px;">Project Guide &amp; Research Mentor</p>
        # <p style="color: #4A5568; font-size: 0.8rem; margin-top: 4px;">Mentored study scope, validated early-stage clinical relevance, and reviewed strict model data leakage guardrails.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader(" Technologies and Libraries Used")
    st.info(" **Core:** Python 3.10+ & Streamlit Framework")
    st.info(" **Data Science:** Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE)")
    st.info(" **Models:** CatBoost, XGBoost, LightGBM, AdaBoost, Gradient Boosting, SVM, Random Forest")
    st.info(" **Explainability:** SHAP (SHapley Additive exPlanations)")
    st.info(" **Graphics:** Matplotlib, Seaborn")

with col2:
    st.subheader(" Project Implementation Timeline")
    
    st.markdown("""
    <div style="margin-left: 10px; border-left: 3px solid #00a8a1; padding-left: 20px; position: relative;">
        <div style="margin-bottom: 22px;">
            <span class="timeline-phase-tag">Weeks 1-2</span>
            <h4 class="timeline-phase-h">Phase 1: Scoping &amp; Lit Review</h4>
            <p class="timeline-phase-desc">Studied medical papers, kidney failure triggers, and diagnosed missing-values thresholds in public clinical surveys.</p>
        </div>
        <div style="margin-bottom: 22px;">
            <span class="timeline-phase-tag">Weeks 3-5</span>
            <h4 class="timeline-phase-h">Phase 2: NHANES Data Ingestion</h4>
            <p class="timeline-phase-desc">Merged CDC examinations, questionnaires, and lab reports using unique identifier sequences.</p>
        </div>
        <div style="margin-bottom: 22px;">
            <span class="timeline-phase-tag">Weeks 6-7</span>
            <h4 class="timeline-phase-h">Phase 3: Pipeline Sanitation</h4>
            <p class="timeline-phase-desc">Eliminated GFR/ACR leakage features, split train/test states, imputed missing cells, and executed correlation filters.</p>
        </div>
        <div style="margin-bottom: 22px;">
            <span class="timeline-phase-tag accent">Weeks 8-11</span>
            <h4 class="timeline-phase-h">Phase 4: ML Exploration &amp; Tuning</h4>
            <p class="timeline-phase-desc">Trained 11 standalone classifiers; optimized hyper-parameters with Cross-Validation; built level-1 Meta Stacking.</p>
        </div>
        <div style="margin-bottom: 10px;">
            <span class="timeline-phase-tag accent">Weeks 12-14</span>
            <h4 class="timeline-phase-h">Phase 5: Interpretability &amp; Web Hub</h4>
            <p class="timeline-phase-desc">Conducted SHAP Beeswarm attribution scans and engineered this interactive multi-page dashboard model presentation.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader(" Future Scope & Upgrades")
    st.markdown("""
    1.  **Deployment Integration:** Activating the prediction interface using `stacking_model.pkl`.

    2.  **API Integration:** Bundling the pipeline into a FastAPI microservice for electronic health records (EHR) integration.
    3.  **Real-time wearable streams:** Incorporating continuous patient vitals (diurnal blood pressure, smartwatch heart-rate metrics) for dynamic predictions.
    4.  **Local Clinic Testing:** Partnering with regional hospital databases to validate the model's out-of-distribution generalizability across global demographics.
    """)
