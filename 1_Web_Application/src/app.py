# src/app.py
"""Main entry point for CKD Smart Diagnostics Web App.

Orchestrates multi-page navigation across 10 research-oriented sections.
"""

import streamlit as st
import os

# Page Config
st.set_page_config(
    page_title="CKD Smart Diagnostics Hub",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS Helper
def local_css(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(script_dir, file_name)
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Inject Theme and UI Styling
local_css("style.css")

# Define Pages
home_page              = st.Page("pages/01_home.py",              title="Project Home",                icon=":material/home:")
ckd_info_page          = st.Page("pages/02_ckd_info.py",          title="About CKD",                   icon=":material/info:")
dataset_page           = st.Page("pages/03_dataset.py",           title="NHANES Dataset & Splits",     icon=":material/dataset:")
preprocessing_page     = st.Page("pages/04_preprocessing.py",     title="Data Preprocessing Pipeline", icon=":material/tune:")
eda_page               = st.Page("pages/05_eda.py",               title="Exploratory Data Analysis",   icon=":material/bar_chart:")
model_comparison_page  = st.Page("pages/06_model_comparison.py",  title="Ensemble & ML Comparison",    icon=":material/model_training:")
explainable_ai_page    = st.Page("pages/07_explainable_ai.py",    title="Explainable AI (SHAP)",        icon=":material/psychology:")
research_findings_page = st.Page("pages/08_research_findings.py", title="Clinical & Key Findings",     icon=":material/lab_research:")
team_page              = st.Page("pages/09_team.py",              title="Research Team & Guide",        icon=":material/group:")
prediction_page        = st.Page("pages/10_prediction.py",        title="Smart Diagnostic Tool",        icon=":material/stethoscope:")

# Multi-Page Navigation — Grouped Sidebar
pg = st.navigation({
    "Project Overview":  [home_page, ckd_info_page],
    "Research Analysis": [dataset_page, preprocessing_page, eda_page,
                          model_comparison_page, explainable_ai_page, research_findings_page],
    "About":             [team_page],
    "Prediction":        [prediction_page],
})

import base64

# Base64 helper for centered sidebar logo
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
logo_img_tag = ""
if os.path.exists(logo_path):
    b64_logo = get_base64_image(logo_path)
    logo_img_tag = f'<div style="text-align: center; margin-bottom: 0.8rem;"><img src="data:image/png;base64,{b64_logo}" style="width: 90px; border-radius: 50%; box-shadow: 0 4px 14px rgba(0,210,196,0.3); border: 2px solid rgba(0,210,196,0.4);"></div>'

# Sidebar Branding & Footer
st.sidebar.markdown(f"""
<div style="text-align: center; padding: 1rem 0.5rem 1rem 0.5rem;">
    {logo_img_tag}
    <h3 style="color: #ffffff !important; margin: 0; font-size: 1.2rem; font-family: 'Outfit', sans-serif; font-weight: 700;">
        NephroAI Hub
    </h3>
    <p style="color: #8bb3e9 !important; font-size: 0.78rem; margin: 4px 0 0 0; letter-spacing: 0.5px;">
        CKD Research Showcase
    </p>
    <hr style="border: none; border-top: 1px solid #1a3c6d; margin: 14px 0 8px 0;">
    <div style="
        background: rgba(0,210,196,0.12);
        border: 1px solid rgba(0,210,196,0.3);
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 0.72rem;
        color: #00D2C4 !important;
        letter-spacing: 0.5px;
    ">
         NHANES · 5,552 Patients<br>
         43 Features · 11+ Models<br>
         Accuracy: 87.13%
    </div>
</div>
""", unsafe_allow_html=True)

# Run navigation
pg.run()
