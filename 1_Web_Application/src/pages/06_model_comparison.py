# src/pages/06_model_comparison.py
import streamlit as st
import os
import pandas as pd

st.title(" Classifier Comparison & Stacking Ensemble")

# Dynamic path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(os.path.dirname(script_dir), "assets")

st.markdown("""
###  Head-to-Head Algorithm Evaluation
To check which model handles CKD screening best, we evaluated 11 standard classifiers spanning probability, distance, kernel, and tree-based formulations, compared against our **Stacking Ensemble**.
""")

# Show main metrics table
st.subheader(" Consolidated Evaluation Metrics (On Holdout Split)")

metrics_data = {
    "Model Name": [
        "Logistic Regression", "Decision Tree", "K-Nearest Neighbors (KNN)", 
        "Gaussian Naive Bayes", "Support Vector Machine (SVM)", "Random Forest", 
        "AdaBoost", "Gradient Boosting (GBM)", "XGBoost", "LightGBM", "CatBoost", 
        " Stacking Ensemble (Meta-LR)"
    ],
    "Accuracy": [0.8632, 0.8272, 0.8344, 0.8290, 0.8569, 0.8668, 0.8578, 0.8650, 0.8515, 0.8641, 0.8677, 0.8713],
    "Precision": [0.7159, 0.4921, 0.5625, 0.5000, 0.7818, 0.8182, 0.6600, 0.6695, 0.7556, 0.6970, 0.7363, 0.7374],
    "Recall (Sensitivity)": [0.3316, 0.3263, 0.1421, 0.4947, 0.2263, 0.2842, 0.3474, 0.4158, 0.3579, 0.3632, 0.3526, 0.3842],
    "F1-Score": [0.4532, 0.3924, 0.2269, 0.4974, 0.3510, 0.4219, 0.4552, 0.5130, 0.4857, 0.4775, 0.4769, 0.5052],
    "ROC AUC": [0.8173, 0.6479, 0.6879, 0.8129, 0.7926, 0.8327, 0.8122, 0.8285, 0.8341, 0.8096, 0.8326, 0.8333]
}
df_metrics = pd.DataFrame(metrics_data)
st.dataframe(df_metrics.style.highlight_max(subset=["Accuracy", "Recall (Sensitivity)", "F1-Score", "ROC AUC"], color="#cdebf9"))

st.markdown("""
<div class="custom-alert">
    <strong> Selecting the Stacking Ensemble:</strong> 
    While Gradient Boosting has a slightly higher F1-score due to recall bias, the <strong>Stacking Ensemble (Meta-Logistic Regression)</strong> provides the most balanced performance, achieving the highest overall Accuracy (<strong>87.13%</strong>), high Precision (<strong>73.74%</strong>), and a solid ROC-AUC of <strong>0.8333</strong>. It meta-generalizes the predictions of diverse base models, reducing prediction variance in clinical screening.
</div>
""", unsafe_allow_html=True)

# Performance Plots tabs
tab_charts, tab_theory = st.tabs([" Visual Comparisons & ROC Curves", " Classifier Explanations & Theory"])

with tab_charts:
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader(" Bar Chart Comparison")
        img_comp = os.path.join(assets_dir, "Combined_Model_Comparison_43Features.png")
        if os.path.exists(img_comp):
            st.image(img_comp, use_container_width=True, caption="Accuracy, F1, Recall & Precision comparisons across model subsets")
        else:
            img_acc = os.path.join(assets_dir, "Accuracy_Comparison_43Features.png")
            if os.path.exists(img_acc):
                st.image(img_acc, use_container_width=True, caption="Model Accuracy Metrics comparison")
            else:
                st.warning("Comparison chart missing.")
                
    with col_chart2:
        st.subheader(" Top Models ROC Curves")
        img_roc = os.path.join(assets_dir, "ROC_Curves_Top3.png")
        if os.path.exists(img_roc):
            st.image(img_roc, use_container_width=True, caption="Receiver Operating Characteristic (ROC) curves of top classifiers")
        else:
            st.warning("ROC curves plot missing.")

with tab_theory:
    st.markdown("""
    ###  Working Rationale Behind the Models Tested
    
    1. **Logistic Regression (LR):** Fits a linear sigmoid boundary. *Advantage:* Highly interpretable, low variance. *Disadvantage:* Fails on complex non-linear combinations.
    2. **Decision Tree (DT):** Splits space using entropy/Gini index. *Advantage:* Simple to view. *Disadvantage:* Overfits heavily, poor generalizing capability.
    3. **K-Nearest Neighbors (KNN):** Distance-based vote ($L_2$ distance). *Advantage:* No training assumptions. *Disadvantage:* Degrades in high dimensions ($D=43$) due to the 'curse of dimensionality'.
    4. **Gaussian Naive Bayes:** Joint probability assuming independent features. *Advantage:* Very fast. *Disadvantage:* Independence assumption violates clinical realities (e.g. Blood Pressure and Diabetes are heavily dependent).
    5. **Support Vector Machine (SVM):** Fits maximum-margin hyperplane in kernel space. *Advantage:* Effective in high dimensional spaces. *Disadvantage:* High computation cost, sensitive to scaling.
    6. **Ensembles (Random Forest, Extra Trees, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost):**
       - **Bagging (RF):** Fits parallel trees on bootstrapped splits. Lowers model variance.
       - **Boosting (XGB, LGBM, CatBoost):** Sequentially fits trees on residual errors. CatBoost handles categorical splits exceptionally well.
    7. **Stacking Ensemble:** Combines predictions from all the above base classifiers, funneling their probability outputs into a **Meta-Logistic Regressor** acting as the final level-1 decision-maker. It leverages the strengths of all models, enhancing predictive safety.
    """)
