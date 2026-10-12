# AIPI 590 - Project 1: ¿Por qué me rechazaron?
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
- `data/`: Dataset scripts and preprocessing pipeline (UCI Default of Credit Card Clients).
- `src/`: Core machine learning, SHAP, and DiCE counterfactual recourse engines.
- `app/`: Interactive Streamlit dashboard for applicants, loan officers, and regulators.
- `notebooks/`: Exploratory data analysis, baseline model evaluation, and fairness audit experiments.

---

## 🚀 Setup & Execution
```bash
git clone git@github.com:tonantzin-real/AIPI_590_Project_1.git
cd AIPI_590_Project_1
pip install -r requirements.txt
streamlit run app/main.py
```
