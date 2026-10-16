# AIPI 590 - Project 1: Why Was I Rejected?
### Auditing Algorithmic Recourse, Actionability, and Burden Disparity in Consumer Credit Risk

## 📌 Overview
Automated credit scoring models frequently deny loan applicants without providing actionable recourse. While explainability frameworks like SHAP highlight retrospective feature attributions (explaining *why* an applicant was denied), they fail to provide prescriptive guidance on *how to achieve approval*.

This project investigates **Algorithmic Recourse** through Counterfactual Explanations using Microsoft's `DiCE` framework, auditing whether credit risk models impose an unequal **Algorithmic Burden** across demographic groups (e.g., gender, age).

---

## 🔬 Research Questions
1. **Explainability Paradigm Gap:** How do retrospective feature attributions (SHAP) compare against prescriptive counterfactuals (DiCE) in providing realistic consumer recourse?
2. **Actionability Constraints:** Can recourse recommendations be constrained to strictly actionable features (e.g., credit card debt, payment streak) while locking immutable traits (age, gender, education)?
3. **Burden Disparity Audit:** Do counterfactual recourse models require a systematically higher financial effort ($L_1$ recourse distance) from historically marginalized demographic groups?

---

## 🛠️ Project Architecture
- `data/`: Cached datasets (UCI Default of Credit Card Clients). Data ingestion and cleaning is handled by `src/data_loader.py`.
- `models/`: Serialized model weights (`credit_model.joblib`), metadata, and audit results (`audit_disparity_results.json`, `audit_disparity_plot.png`).
- `src/`: Core machine learning, SHAP, DiCE counterfactual recourse, and fairness auditing modules:
  - `data_loader.py`: Ingestion, caching, and domain-grounded feature categorization.
  - `train_model.py`: LightGBM classifier training and calibration.
  - `xai_shap.py`: Global and local TreeSHAP attribution.
  - `xai_recourse.py`: Actionable counterfactual recourse engine via Microsoft DiCE.
  - `audit.py`: Out-of-sample demographic recourse disparity audit with hypothesis testing.
- `app/`: Interactive Streamlit dashboard for applicants, risk analysts, and regulators.
- `notebooks/`: Exploratory analysis, baseline model evaluation, and recourse experiments.

---

## 🚀 Setup & Execution (Fresh Clone)
```bash
git clone git@github.com:tonantzin-real/AIPI_590_Project_1.git
cd AIPI_590_Project_1
pip install -r requirements.txt

# Step 1: Ingest data and train champion LightGBM model (generates models/credit_model.joblib)
python src/train_model.py

# Step 2: Run multi-dimensional fairness audit with statistical testing (generates models/audit_disparity_results.json & plot)
python src/audit.py

# Step 3: Launch interactive Streamlit web application
streamlit run app/main.py
```
