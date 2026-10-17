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
from src.data_loader import CATEGORICAL_LABELS, DEFAULT_DATA_PATH, download_and_clean_data
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
    # Downloads and caches the dataset from OpenML on first launch (e.g. fresh clone or cloud deployment)
    download_and_clean_data(cache_path=DEFAULT_DATA_PATH)
    recourse_engine = CreditRecourseEngine(
        model_path=model_path,
        data_path=DEFAULT_DATA_PATH,
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
                        # Escape "$" so Streamlit's markdown does not render "$500 ➔ $16,281" as a LaTeX formula
                        plan_text = f"{change['from']} ➔ {change['to']} ({change['difference']})".replace("$", "\\$")
                        st.info(f"**{change['action']}:** {plan_text}")
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
        """)

        shap_audit_path = os.path.join(BASE_DIR, "models", "shap_actionability_results.json")
        if os.path.exists(shap_audit_path):
            with open(shap_audit_path, "r") as f:
                shap_audit = json.load(f)
            shares = shap_audit["share_of_top_k_reasons_pct"]
            not_actionable = round(100.0 - shares.get("actionable", 0.0), 1)
            st.warning(
                f"**Measured across all {shap_audit['metadata']['rejected_applicants']:,} rejected test applicants:** "
                f"{not_actionable:.0f}% of SHAP's top-{shap_audit['metadata']['top_k']} reasons are things the applicant cannot change today "
                f"(past payment history {shares.get('historical', 0.0):.0f}%, bank-set credit limit {shares.get('institutional', 0.0):.0f}%, "
                f"demographics {shares.get('immutable', 0.0):.0f}%). For {shap_audit['applicants_whose_top_reason_is_not_actionable_pct']:.0f}% "
                f"of applicants the #1 reason is not actionable, and {shap_audit['applicants_with_no_actionable_reason_in_top_k_pct']:.0f}% "
                f"get no actionable reason at all."
            )

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


with tab3:
    st.subheader("Fairness Audit: Does Everyone Get a Fair Path to Approval?")

    audit_json_path = os.path.join(BASE_DIR, "models", "audit_disparity_results.json")
    if os.path.exists(audit_json_path):
        with open(audit_json_path, "r") as f:
            audit_data = json.load(f)

        comps = audit_data["comparisons"]
        subgroups = audit_data["subgroups"]
        overall = audit_data["overall"]
        g_comp, a_comp, e_comp, m_comp = comps["gender"], comps["age"], comps["education"], comps["marriage"]

        st.caption(
            f"All {audit_data['metadata']['model_rejected_sample_size']:,} rejected applicants in the hold-out test set. "
            "**Effort** = size of the changes a plan requires (among applicants who got a plan). "
            "**Plan rate** = share of applicants for whom any feasible plan exists. "
            "Significant = unlikely to be due to chance (p < 0.05 after correcting for multiple comparisons)."
        )

        # One-line key finding, generated from the results
        effort_gaps = [DIMENSION_LABELS[k][0].lower() for k, c in comps.items() if c["is_burden_statistically_significant"]]
        plan_gaps = []
        for k, c in comps.items():
            if c["is_feasibility_statistically_significant"]:
                rate_a = subgroups[c["group_a"]]["feasibility_rate_pct"]
                rate_b = subgroups[c["group_b"]]["feasibility_rate_pct"]
                low, low_rate, high_rate = (c["group_a"], rate_a, rate_b) if rate_a < rate_b else (c["group_b"], rate_b, rate_a)
                plan_gaps.append(f"**{low.lower()}** applicants ({low_rate:.1f}% vs. {high_rate:.1f}%)")
        effort_text = (
            "No group needs significantly more effort to get approved."
            if not effort_gaps else f"Effort differs significantly by {', '.join(effort_gaps)}."
        )
        plan_text = (
            "Plan rates do not differ significantly across groups."
            if not plan_gaps else f"But plans are found significantly less often for {' and '.join(plan_gaps)}."
        )
        st.markdown(f"**Key finding:** {effort_text} {plan_text}")

        summary_rows = []
        for dim_key, comp in comps.items():
            rate_a = subgroups[comp["group_a"]]["feasibility_rate_pct"]
            rate_b = subgroups[comp["group_b"]]["feasibility_rate_pct"]
            summary_rows.append({
                "Dimension": DIMENSION_LABELS[dim_key][0],
                "Groups (A vs. B)": f"{comp['group_a']} vs. {comp['group_b']}",
                "Effort ratio (A / B)": f"{comp['disparity_ratio']:.2f}x",
                "Effort gap?": "⚠️ Significant" if comp["is_burden_statistically_significant"] else "No",
                "Plan rate (A vs. B)": f"{rate_a:.1f}% vs. {rate_b:.1f}%",
                "Plan-rate gap?": "⚠️ Significant" if comp["is_feasibility_statistically_significant"] else "No",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

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

            # Interpretation only: the per-group numbers are already shown in the summary table above
            access_gap_dims = [k for k, c in comps.items() if c["is_feasibility_statistically_significant"]]
            effort_gap_dims = [k for k, c in comps.items() if c["is_burden_statistically_significant"]]
            meaning = []
            if access_gap_dims and not effort_gap_dims:
                meaning.append(
                    "**The gap is in access, not effort.** Once a plan exists, it asks about the same of everyone, "
                    "but some groups are more often left with no realistic path to approval."
                )
            elif effort_gap_dims:
                meaning.append("**Some groups must work harder to get approved**, which is a direct fairness concern for the model.")
            else:
                meaning.append("**No significant gaps were found** in either effort or access to a plan.")
            if {"age", "marriage"} <= set(access_gap_dims):
                meaning.append(
                    "Age and marital status overlap (younger applicants are more often single), "
                    "so these two gaps may reflect the same underlying pattern."
                )
            meaning.append(
                "Effort is only measured for applicants who get a plan, so a group that gets fewer plans can look "
                "like it needs less effort."
            )
            st.markdown("**What this means**\n" + "\n".join(f"- {m}" for m in meaning))
            st.markdown("""
            **Recommendations**
            1. Require lenders to give rejected applicants **concrete steps to qualify**, not only the reasons for rejection.
            2. Audit **who gets a path to approval at all**, not only how hard that path is.
            3. Re-run this audit **whenever the model changes**, and check that recommended actions are affordable before showing them.
            """)

    else:
        st.warning("No audit results found. Run `python src/audit.py` to generate them.")
