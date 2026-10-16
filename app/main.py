"""
=============================================================================
APPLICATION: main.py (Streamlit Web Dashboard)
PROJECT: AIPI 590 - Project 1: Why Was I Rejected?
AUTHOR: Tonantzin Real Rojas
DESCRIPTION:
    Interactive portfolio artifact demonstrating Explainable AI, Algorithmic
    Recourse (DiCE), and Multi-Dimensional Fairness Auditing with Statistical
    Significance Testing for credit scoring underwriting.
    Designed for loan applicants, risk officers, and financial regulators.
=============================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Relative imports with robust base directory resolution
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from src.data_loader import CATEGORICAL_LABELS, DEFAULT_DATA_PATH
from src.xai_shap import CreditSHAPExplainer
from src.xai_recourse import CreditRecourseEngine

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Why Was I Rejected? | XAI Credit Recourse",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for professional fintech UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 1.2rem;
        border-left: 4px solid #2563eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-approved {
        background-color: #dcfce7;
        color: #166534;
        font-weight: bold;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
    }
    .badge-rejected {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: bold;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# RESOURCE CACHING
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models_and_engines():
    """Loads serialized model, SHAP explainer, and DiCE recourse engine."""
    model_path = os.path.join(BASE_DIR, "models", "credit_model.joblib")
    meta_path = os.path.join(BASE_DIR, "models", "model_metadata.json")
    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    shap_explainer = CreditSHAPExplainer(model_path=model_path)
    recourse_engine = CreditRecourseEngine(
        model_path=model_path,
        data_path=os.path.join(BASE_DIR, "data", "credit_card_clean.csv"),
        sample_size=1500
    )
    return model, metadata, shap_explainer, recourse_engine


model, metadata, shap_explainer, recourse_engine = load_models_and_engines()


# ---------------------------------------------------------------------------
# HUMAN-READABLE FEATURE LABELS
# ---------------------------------------------------------------------------
MONTH_AGO = {1: "most recent month", 2: "2 months ago", 3: "3 months ago", 4: "4 months ago", 5: "5 months ago", 6: "6 months ago"}

FEATURE_LABELS = {
    "LIMIT_BAL": "Credit limit",
    "SEX": "Sex",
    "EDUCATION": "Education level",
    "MARRIAGE": "Marital status",
    "AGE": "Age",
    **{f"PAY_{i}": f"Payment status, {MONTH_AGO[i]}" for i in range(1, 7)},
    **{f"BILL_AMT{i}": f"Statement balance, {MONTH_AGO[i]}" for i in range(1, 7)},
    **{f"PAY_AMT{i}": f"Amount paid, {MONTH_AGO[i]}" for i in range(1, 7)},
}

MONEY_FEATURES = {"LIMIT_BAL"} | {f"BILL_AMT{i}" for i in range(1, 7)} | {f"PAY_AMT{i}" for i in range(1, 7)}

# Plain-language action for each actionable lever, depending on the direction of the change
RECOURSE_ACTIONS = {
    "PAY_1": {"down": "Improve your most recent payment status", "up": "Most recent payment status changes"},
    "BILL_AMT1": {"down": "Reduce your current statement balance", "up": "Current statement balance increases"},
    "PAY_AMT1": {"down": "Most recent payment decreases", "up": "Increase your most recent payment"},
}


def feature_label(feature: str) -> str:
    """Returns 'Interpretation (FEATURE)' for display."""
    return f"{FEATURE_LABELS.get(feature, feature)} ({feature})"


def format_money(value: float) -> str:
    return f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


def payment_status_label(code: float) -> str:
    code = int(round(code))
    if code == -2:
        return "No card use / zero balance"
    if code == -1:
        return "Paid in full on time"
    if code == 0:
        return "Revolving credit (minimum payment made)"
    return f"{code}-month payment delay"


def describe_recourse_change(feature: str, old: float, new: float) -> dict:
    """Builds a human-readable description of one recommended change."""
    direction = "up" if new > old else "down"
    action = RECOURSE_ACTIONS.get(feature, {}).get(direction, FEATURE_LABELS.get(feature, feature))
    if feature.startswith("PAY_") and not feature.startswith("PAY_AMT"):
        from_str, to_str = payment_status_label(old), payment_status_label(new)
        steps = int(round(abs(new - old)))
        diff_str = f"{steps} level{'s' if steps != 1 else ''} {'better' if new < old else 'worse'}"
    elif feature in MONEY_FEATURES:
        from_str, to_str = format_money(old), format_money(new)
        diff_str = ("+" if new > old else "-") + f"${abs(new - old):,.0f}"
    else:
        from_str, to_str = f"{old:,.0f}", f"{new:,.0f}"
        diff_str = f"{new - old:+,.0f}"
    return {
        "action": f"{action} ({feature})",
        "from": from_str,
        "to": to_str,
        "difference": diff_str,
    }


def recourse_changes(original: pd.Series, cf_row: pd.Series) -> list:
    """Lists the human-readable changes between the applicant and one counterfactual plan."""
    changes = []
    for feature in ["PAY_1", "BILL_AMT1", "PAY_AMT1"]:
        if feature in cf_row and abs(float(cf_row[feature]) - float(original[feature])) > 0.01:
            changes.append(describe_recourse_change(feature, float(original[feature]), float(cf_row[feature])))
    return changes


# ---------------------------------------------------------------------------
# SIDEBAR CONTROLS & APPLICANT PROFILER
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/bank-card-back-side.png", width=64)
st.sidebar.markdown("### 👤 Applicant Profile")
st.sidebar.markdown("Configure loan applicant financial and demographic details:")

preset = st.sidebar.selectbox(
    "Load a preset profile:",
    ["High-risk applicant (rejected)", "Reliable applicant (approved)", "Custom"]
)

if preset == "High-risk applicant (rejected)":
    init_limit = 20000
    init_sex = 2  # Female
    init_edu = 2  # University
    init_mar = 1  # Married
    init_age = 24
    init_pay1 = 2  # 2 months delayed payment
    init_bill1 = 45000
    init_payamt1 = 500
elif preset == "Reliable applicant (approved)":
    init_limit = 150000
    init_sex = 1  # Male
    init_edu = 1  # Grad school
    init_mar = 2  # Single
    init_age = 35
    init_pay1 = -1 # Paid in full
    init_bill1 = 12000
    init_payamt1 = 12000
else:
    init_limit = 50000
    init_sex = 2
    init_edu = 2
    init_mar = 2
    init_age = 28
    init_pay1 = 1
    init_bill1 = 25000
    init_payamt1 = 2000

# Input Controls
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    sex = st.selectbox("Sex (SEX):", [1, 2], index=0 if init_sex == 1 else 1, format_func=lambda x: CATEGORICAL_LABELS["SEX"][x])
    marriage = st.selectbox("Marital status (MARRIAGE):", [1, 2, 3], index=init_mar - 1, format_func=lambda x: CATEGORICAL_LABELS["MARRIAGE"][x])
with col_sb2:
    age = st.slider("Age (AGE):", 18, 75, init_age)
    education = st.selectbox("Education level (EDUCATION):", [1, 2, 3, 4], index=init_edu - 1, format_func=lambda x: CATEGORICAL_LABELS["EDUCATION"][x])

limit_bal = st.sidebar.number_input("Credit limit in $ (LIMIT_BAL):", min_value=10000, max_value=800000, value=init_limit, step=10000)

st.sidebar.markdown("#### Recent History and Balances")
pay_1 = st.sidebar.selectbox(
    "Most recent payment status (PAY_1):",
    [-2, -1, 0, 1, 2, 3],
    index=[-2, -1, 0, 1, 2, 3].index(init_pay1),
    format_func=lambda x: f"{payment_status_label(x)} ({x:+d})" if x else f"{payment_status_label(x)} (0)"
)
st.sidebar.caption("ℹ️ *Note:* In the UCI credit data, -2 and -1 mean the balance was paid off or paid on time, 0 means revolving credit with the minimum payment made, and positive values are months of payment delay.")

bill_amt1 = st.sidebar.number_input("Current statement balance in $ (BILL_AMT1):", min_value=0, max_value=500000, value=init_bill1, step=2500)
pay_amt1 = st.sidebar.number_input("Most recent amount paid in $ (PAY_AMT1):", min_value=0, max_value=300000, value=init_payamt1, step=1000)

# Build Complete Feature Vector (23 features)
sample_vector = {
    "LIMIT_BAL": limit_bal,
    "SEX": sex,
    "EDUCATION": education,
    "MARRIAGE": marriage,
    "AGE": age,
    "PAY_1": pay_1,
    "PAY_2": 0 if pay_1 <= 0 else pay_1 - 1,
    "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": bill_amt1,
    "BILL_AMT2": int(bill_amt1 * 0.95),
    "BILL_AMT3": int(bill_amt1 * 0.90),
    "BILL_AMT4": int(bill_amt1 * 0.85),
    "BILL_AMT5": int(bill_amt1 * 0.80),
    "BILL_AMT6": int(bill_amt1 * 0.75),
    "PAY_AMT1": pay_amt1,
    "PAY_AMT2": int(pay_amt1 * 0.9),
    "PAY_AMT3": int(pay_amt1 * 0.8),
    "PAY_AMT4": int(pay_amt1 * 0.8),
    "PAY_AMT5": int(pay_amt1 * 0.8),
    "PAY_AMT6": int(pay_amt1 * 0.8)
}
applicant_df = pd.DataFrame([sample_vector])


# ---------------------------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">Why Was I Rejected?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header"><b>Auditing Explainability, Actionable Recourse, and Burden Disparity in Consumer Credit</b> | '
    'AIPI 590: Explainable AI Final Project</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "💳 1. Simulator & Actionable Recourse",
    "⚖️ 2. SHAP vs. DiCE (The XAI Paradox)",
    "📊 3. Algorithmic Burden Audit"
])


# ---------------------------------------------------------------------------
# TAB 1: SIMULATOR & ACTIONABLE RECOURSE
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Automated Credit Model Decision")

    # Model inference
    proba_default = float(model.predict_proba(applicant_df)[:, 1][0])
    threshold = metadata["threshold"]
    is_rejected = proba_default >= threshold

    col_res1, col_res2, col_res3 = st.columns([1.5, 1.5, 3])
    with col_res1:
        st.markdown("**Result:**")
        if is_rejected:
            st.markdown('<span class="badge-rejected">❌ APPLICATION REJECTED (High Risk)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-approved">✅ APPLICATION APPROVED (Reliable)</span>', unsafe_allow_html=True)

    with col_res2:
        st.metric(
            "Probability of Default:",
            f"{proba_default:.1%}",
            delta=f"{(proba_default - threshold):+.1%} vs. threshold",
            delta_color="inverse"
        )

    with col_res3:
        st.markdown(f"""
        **Decision Rule:**
        The lender's risk threshold is **{threshold:.0%}**. 
        {f'This applicant exceeds the risk tolerance by {(proba_default - threshold):.1%}.' if is_rejected else 'This applicant qualifies for the requested credit line.'}
        """)

    st.markdown("---")

    if is_rejected:
        st.markdown("### 🎯 Actionable Recourse Plans (DiCE Counterfactuals)")
        st.markdown("""
        Under consumer financial protection frameworks (**CONDUSEF / GDPR Art. 22**), it is not enough to explain *why* an application was rejected.
        The applicant deserves **minimal, feasible action plans** to reverse the decision.
        """)

        with st.spinner("Searching for actionable plans under strict constraints..."):
            cfs_df, deltas_df, mean_burden = recourse_engine.generate_recourse_plans(applicant_df, total_cfs=2)

        if not deltas_df.empty:
            col_plan1, col_plan2 = st.columns(2)
            for idx, row in deltas_df.iterrows():
                col_target = col_plan1 if idx == 0 else col_plan2
                with col_target:
                    st.markdown(f"#### 📋 {row['Plan_ID']} (Relative effort: {row['Normalized_Burden']:.2f})")
                    for change in recourse_changes(applicant_df.iloc[0], cfs_df.iloc[idx]):
                        st.info(f"**{change['action']}:** {change['from']} ➔ {change['to']} ({change['difference']})")
                    st.caption("🔒 *Personal traits (age, sex, education, marital status), past payment history, and the bank-set credit limit were kept fixed.*")
            
            st.success(f"**Average algorithmic effort (normalized L1 distance):** {mean_burden:.2f}")
        else:
            st.warning("No feasible plan was found within a single step. This applicant would need broader, longer-term debt restructuring.")
    else:
        st.balloons()
        st.success("Congratulations! This profile qualifies for credit approval.")


# ---------------------------------------------------------------------------
# TAB 2: SHAP VS. DICE (THE EXPLAINABILITY PARADOX)
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("The XAI Paradox: Retrospective Attribution vs. Prescriptive Recourse")
    st.markdown("""
    In finance, **SHAP** and **LIME** are commonly used to explain credit denials. 
    This comparison shows why feature attributions alone are not enough for the applicant.
    """)

    col_shap, col_dice = st.columns(2)

    with col_shap:
        st.markdown("### 📉 Traditional Approach: SHAP (Retrospective)")
        st.caption("Answers: *'What did you do wrong in the past?'* (static attribution)")
        shap_vals, base_val, summary_df = shap_explainer.explain_instance(applicant_df)

        fig_shap, ax_shap = plt.subplots(figsize=(7, 4.5))
        top_shap = summary_df.head(6)
        colors = ["#ef4444" if v > 0 else "#22c55e" for v in top_shap["SHAP_Attribution"]]
        ax_shap.barh([feature_label(f) for f in top_shap["Feature"]], top_shap["SHAP_Attribution"], color=colors)
        ax_shap.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax_shap.set_xlabel("SHAP impact on default risk (red = raises risk, green = lowers risk)")
        ax_shap.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_shap)

        st.markdown("""
        **Limitations of SHAP for the Applicant:**
        - It only assigns blame (*"Your balance is high and your history shows delays"*).
        - It does not say how much to pay or which concrete goal leads to approval.
        - It often highlights traits the applicant cannot change.
        """)

    with col_dice:
        st.markdown("### 🚀 Responsible Approach: DiCE (Prescriptive)")
        st.caption("Answers: *'What is the smallest action that gets you approved?'* (actionable recourse)")

        if is_rejected and not deltas_df.empty:
            plan_rows = []
            for idx, row in deltas_df.iterrows():
                for change in recourse_changes(applicant_df.iloc[0], cfs_df.iloc[idx]):
                    plan_rows.append({
                        "Plan": row["Plan_ID"],
                        "Recommended change": change["action"],
                        "From": change["from"],
                        "To": change["to"],
                        "Difference": change["difference"],
                        "Relative effort": f"{row['Normalized_Burden']:.2f}",
                    })
            st.dataframe(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True)
            st.markdown("""
            **Key Advantages of DiCE / Algorithmic Recourse:**
            - **Strict actionability:** Only changes levers the applicant controls today: payment status (`PAY_1`), statement balance (`BILL_AMT1`), and amount paid (`PAY_AMT1`).
            - **Non-discrimination:** Personal traits (sex, age, education, marital status) are locked.
            - **Causal and temporal realism:** Past payment history (`PAY_2..6`) and the bank-set credit limit (`LIMIT_BAL`) are never changed.
            - **Monotonic constraints:** A plan can never worsen payment status or pay more than the statement balance.
            - **Minimal effort:** Picks the lowest-friction path ($L_1$ distance).
            """)
        else:
            st.info("To see the recourse comparison, select a high-risk profile in the sidebar.")


# ---------------------------------------------------------------------------
# TAB 3: ALGORITHMIC BURDEN AUDIT (REGULATORY AUDIT)
# ---------------------------------------------------------------------------
DIMENSION_LABELS = {
    "gender": ("Gender", "Female / Male"),
    "age": ("Age", "Under 30 / 30 and older"),
    "education": ("Education", "High school or other / Graduate or university"),
    "marriage": ("Marital status", "Single / Married"),
}


def burden_sentence(comp: dict) -> str:
    p, p_adj = comp["welch_t_pvalue"], comp["welch_t_pvalue_holm_adjusted"]
    if comp["is_burden_statistically_significant"]:
        return f"Statistically significant difference in algorithmic burden (p = {p:.4f}, p_adj = {p_adj:.4f} < 0.05)."
    return f"No evidence of a difference in algorithmic burden (p = {p:.4f}, p_adj = {p_adj:.4f} ≥ 0.05); we fail to reject H₀."


def feasibility_sentence(comp: dict) -> str:
    p, p_adj = comp["fisher_feasibility_pvalue"], comp["fisher_pvalue_holm_adjusted"]
    if comp["is_feasibility_statistically_significant"]:
        return f"statistically significant feasibility gap (p = {p:.4f}, p_adj = {p_adj:.4f} < 0.05)"
    return f"no statistically significant feasibility gap (p = {p:.4f}, p_adj = {p_adj:.4f} ≥ 0.05)"


with tab3:
    st.subheader("Regulatory Fairness and Algorithmic Burden Audit")
    st.markdown("""
    An empirical evaluation on the **hold-out test set** of whether the credit model imposes a 
    **disproportionate algorithmic burden** across 4 demographic dimensions (sex, age, education, and marital status).
    """)

    audit_json_path = os.path.join(BASE_DIR, "models", "audit_disparity_results.json")
    if os.path.exists(audit_json_path):
        with open(audit_json_path, "r") as f:
            audit_data = json.load(f)

        comps = audit_data["comparisons"]
        subgroups = audit_data["subgroups"]
        overall = audit_data["overall"]

        # 4 Key Demographic Dimensions Cards
        st.markdown("#### 1. Burden Ratios by Demographic Dimension")
        col_g, col_a, col_e, col_m = st.columns(4)

        with col_g:
            g_comp = comps["gender"]
            ratio_g = g_comp["disparity_ratio"]
            st.metric(
                f"{DIMENSION_LABELS['gender'][0]} ({DIMENSION_LABELS['gender'][1]}):",
                f"{ratio_g:.3f}x",
                delta=f"{(ratio_g - 1.0):+.1%} vs. parity (1.00x)",
                delta_color="inverse" if g_comp["is_burden_statistically_significant"] else "off"
            )
            st.caption(f"Welch t-test: p = {g_comp['welch_t_pvalue']:.3f} (p_adj = {g_comp['welch_t_pvalue_holm_adjusted']:.3f}) | {'Significant' if g_comp['is_burden_statistically_significant'] else 'No significant difference'}")

        with col_a:
            a_comp = comps["age"]
            ratio_a = a_comp["disparity_ratio"]
            st.metric(
                f"{DIMENSION_LABELS['age'][0]} ({DIMENSION_LABELS['age'][1]}):",
                f"{ratio_a:.3f}x",
                delta=f"{(ratio_a - 1.0):+.1%} vs. parity (1.00x)",
                delta_color="inverse" if a_comp["is_burden_statistically_significant"] else "off"
            )
            st.caption(f"Welch t-test: p = {a_comp['welch_t_pvalue']:.3f} (p_adj = {a_comp['welch_t_pvalue_holm_adjusted']:.3f}) | {'Significant' if a_comp['is_burden_statistically_significant'] else 'No significant difference'}")

        with col_e:
            e_comp = comps["education"]
            ratio_e = e_comp["disparity_ratio"]
            st.metric(
                f"{DIMENSION_LABELS['education'][0]} ({DIMENSION_LABELS['education'][1]}):",
                f"{ratio_e:.3f}x",
                delta=f"{(ratio_e - 1.0):+.1%} vs. parity (1.00x)",
                delta_color="inverse" if e_comp["is_burden_statistically_significant"] else "off"
            )
            st.caption(f"Welch t-test: p = {e_comp['welch_t_pvalue']:.3f} (p_adj = {e_comp['welch_t_pvalue_holm_adjusted']:.3f}) | {'Significant' if e_comp['is_burden_statistically_significant'] else 'No significant difference'}")

        with col_m:
            m_comp = comps["marriage"]
            ratio_m = m_comp["disparity_ratio"]
            st.metric(
                f"{DIMENSION_LABELS['marriage'][0]} ({DIMENSION_LABELS['marriage'][1]}):",
                f"{ratio_m:.3f}x",
                delta=f"{(ratio_m - 1.0):+.1%} vs. parity (1.00x)",
                delta_color="inverse" if m_comp["is_burden_statistically_significant"] else "off"
            )
            st.caption(f"Welch t-test: p = {m_comp['welch_t_pvalue']:.3f} (p_adj = {m_comp['welch_t_pvalue_holm_adjusted']:.3f}) | {'Significant' if m_comp['is_burden_statistically_significant'] else 'No significant difference'}")

        st.info(
            "⚠️ **Methodological note on survivorship bias:** A conditional burden ratio below 1.00x should **not** be read as "
            "lower effort if it comes with a lower feasibility rate. Effort is only measured for applicants who obtain a feasible plan "
            "within the constraints; the most financially stressed profiles are dropped as infeasible, which can understate the "
            "average burden of those groups."
        )

        st.markdown("---")

        # Overall Metrics Cards
        col_ov1, col_ov2, col_ov3 = st.columns(3)
        with col_ov1:
            st.metric("Overall Recourse Feasibility:", f"{overall['feasibility_rate_pct']:.1f}%", delta="Feasible plans found")
            st.caption(f"{overall['successful_total']} of {overall['evaluated_total']} evaluated test-set applicants.")
        with col_ov2:
            st.metric("Mean Conditional Algorithmic Burden:", f"{overall['mean_burden']:.2f}", delta=f"95% CI: [{overall['ci_95_lower']:.2f}, {overall['ci_95_upper']:.2f}]")
            st.caption("L1 distance normalized by population standard deviations.")
        with col_ov3:
            st.metric("Rejected Applicants in Test Set:", f"{audit_data['metadata']['model_rejected_sample_size']:,}", delta=f"Out of {audit_data['metadata']['test_sample_size']:,} test cases")
            st.caption(f"Applications with P(default) ≥ {audit_data['metadata']['threshold']:.0%}.")

        st.markdown("---")

        col_plot, col_recs = st.columns([1.3, 1])
        with col_plot:
            st.markdown("#### Normalized L1 Effort by Subgroup (with 95% Confidence Intervals)")
            plot_img_path = os.path.join(BASE_DIR, "models", "audit_disparity_plot.png")
            if os.path.exists(plot_img_path):
                st.image(plot_img_path, use_container_width=True)

        with col_recs:
            st.markdown("#### 🏛️ Conclusions and Regulatory Recommendations (CONDUSEF & Mexico's Fintech Law)")
            
            sub_high = subgroups.get("Graduate / University", {})
            sub_sec = subgroups.get("High school / Other", {})
            sub_yng = subgroups.get("Under 30", {})
            sub_old = subgroups.get("30 and older", {})
            
            st.markdown(f"""
            1. **Gender ({DIMENSION_LABELS['gender'][1]}: {g_comp['disparity_ratio']:.3f}x):**  
               {burden_sentence(g_comp)}
            2. **Age ({DIMENSION_LABELS['age'][1]}: {a_comp['disparity_ratio']:.3f}x):**  
               {burden_sentence(a_comp)}  
               *Feasibility:* {sub_yng.get('feasibility_rate_pct', 0.0):.1f}% for applicants under 30 vs. {sub_old.get('feasibility_rate_pct', 0.0):.1f}% for applicants 30 and older ({feasibility_sentence(a_comp)}).
            3. **Education ({DIMENSION_LABELS['education'][1]}: {e_comp['disparity_ratio']:.3f}x):**  
               {burden_sentence(e_comp)}  
               *Feasibility:* {sub_high.get('feasibility_rate_pct', 0.0):.1f}% for graduate/university vs. {sub_sec.get('feasibility_rate_pct', 0.0):.1f}% for high school/other ({feasibility_sentence(e_comp)}).
            4. **Marital status ({DIMENSION_LABELS['marriage'][1]}: {m_comp['disparity_ratio']:.3f}x):**  
               {burden_sentence(m_comp)}
            5. **Survivorship bias and multiple-testing control:**  
               Conditional burden only covers applicants with a feasible plan ({overall['feasibility_rate_pct']:.1f}% overall). Feasibility is therefore audited separately with Fisher's exact test, and Holm-Bonferroni corrections control the family-wise error rate (both per family of 4 tests and globally across all 8 tests).
            """)

        # Statistical Testing Detail Expander
        with st.expander("🔬 Hypothesis Test Details and Holm-Bonferroni Correction"):
            st.markdown("### Statistical Test Comparison Table")
            detail_records = []
            for dim_key, c_info in comps.items():
                detail_records.append({
                    "Dimension": DIMENSION_LABELS.get(dim_key, (c_info["dimension"],))[0],
                    "Comparison": f"{c_info['group_a']} vs. {c_info['group_b']}",
                    "Burden ratio": f"{c_info['disparity_ratio']:.3f}x",
                    "Welch p (raw)": f"{c_info['welch_t_pvalue']:.4f}",
                    "Welch p (Holm 4-tests)": f"{c_info['welch_t_pvalue_holm_adjusted']:.4f}",
                    "Welch p (Holm global 8-tests)": f"{c_info.get('welch_t_pvalue_holm_global_8tests', c_info['welch_t_pvalue_holm_adjusted']):.4f}",
                    "Feasibility gap": f"{c_info['feasibility_gap_pct']:+.1f} pts",
                    "Fisher p (raw)": f"{c_info['fisher_feasibility_pvalue']:.4f}",
                    "Fisher p (Holm 4-tests)": f"{c_info['fisher_pvalue_holm_adjusted']:.4f}",
                    "Fisher p (Holm global 8-tests)": f"{c_info.get('fisher_pvalue_holm_global_8tests', c_info['fisher_pvalue_holm_adjusted']):.4f}",
                    "Significant (p_adj < 0.05)": "Yes (p_adj < 0.05)" if (c_info["is_burden_statistically_significant"] or c_info["is_feasibility_statistically_significant"]) else "No (p_adj ≥ 0.05)"
                })
            st.dataframe(pd.DataFrame(detail_records), use_container_width=True)

    else:
        st.warning("No audit results found. Run `python src/audit.py` to generate them.")
