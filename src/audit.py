"""
=============================================================================
MODULE: audit.py
PROJECT: AIPI 590 - Project 1: ¿Por qué me rechazaron?
AUTHOR: Tonantzin Real Rojas
DESCRIPTION:
    Audits Algorithmic Recourse Disparity, Feasibility, and Burden Inequality
    across four demographic dimensions on the hold-out test set:
      1. Gender (Male vs. Female)
      2. Age (Younger < 30 vs. Older >= 30)
      3. Education (Higher Ed vs. Secondary / Other)
      4. Marital Status (Married vs. Single)

    Methodological & Statistical Rigor:
      - Uses actual model-predicted rejections on the hold-out test set (no data leakage).
      - Enforces strict actionability constraints (monotonic payment bounds, bill-coupled payments).
      - Computes empirical feasibility rates and audits them via Fisher's Exact Test.
      - Evaluates conditional algorithmic burden using Welch's t-test and Mann-Whitney U test.
      - Applies Holm-Bonferroni step-down correction for multiple hypothesis testing.
      - Accounts for survivorship bias by analyzing feasibility and burden in tandem.
      - Reports 95% Confidence Intervals for Algorithmic Burden.
=============================================================================
"""

import os
import json
import joblib
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Relative import with robust base directory resolution
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from src.data_loader import (
    get_train_test_data,
    DEFAULT_DATA_PATH,
    IMMUTABLE_FEATURES,
    HISTORICAL_FEATURES,
    INSTITUTIONAL_FEATURES,
    ACTIONABLE_FEATURES,
)
from src.xai_recourse import CreditRecourseEngine

DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_model.joblib")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "models")


def compute_ci(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes mean and 95% confidence interval using Student-t distribution.
    """
    n = len(data)
    if n < 2:
        val = float(data[0]) if n == 1 else 0.0
        return val, val, val
    mean = float(np.mean(data))
    se = float(stats.sem(data))
    h = float(se * stats.t.ppf((1 + confidence) / 2., n - 1))
    return round(mean, 2), round(max(0.0, mean - h), 2), round(mean + h, 2)


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Applies Holm-Bonferroni step-down correction to control family-wise error rate.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        step = m - rank
        adj_p = min(1.0, p * step)
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = round(running_max, 4)
    return adjusted


def run_demographic_recourse_audit(
    sample_per_group: int = 80,
    model_path: str = DEFAULT_MODEL_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs an empirical audit of Algorithmic Recourse Burden and Feasibility
    on out-of-sample model-rejected applicants.

    Parameters:
        sample_per_group (int): Number of rejected instances to evaluate per cohort.
        model_path (str): Path to trained credit classification model.
        output_dir (str): Directory where plots and JSON audit results are saved.
        random_state (int): Reproducibility seed.

    Returns:
        dict: Full audit statistics, statistical hypothesis tests, and disparity ratios.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load model metadata to retrieve official threshold
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    threshold = 0.50
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as mf:
                meta = json.load(mf)
                threshold = meta.get("threshold", 0.50)
        except Exception:
            threshold = 0.50

    # 2. Load out-of-sample test set (prevents training data leakage)
    _, X_test, _, y_test = get_train_test_data(random_state=random_state)
    model = joblib.load(model_path)

    # 3. Identify applicants ACTUALLY rejected by the model decision (P(default) >= threshold)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    model_rejected_mask = (y_pred_proba >= threshold)
    rejected_test_df = X_test[model_rejected_mask].copy()

    print("\n" + "=" * 65)
    print("STARTING RESPONSIBLE AI DEMOGRAPHIC RECOURSE AUDIT")
    print(f"Total Test Set Size:               {len(X_test)} applicants")
    print(f"Model Risk Cutoff Threshold:       {threshold:.2f}")
    print(f"Model-Predicted Denials in Test:   {len(rejected_test_df)} ({len(rejected_test_df)/len(X_test):.1%})")
    print(f"Sample Evaluated per Subgroup:     {sample_per_group} applicants")
    print("=" * 65)

    engine = CreditRecourseEngine(model_path=model_path, random_state=random_state)

    # 4. Define Subgroups across 4 Demographic Dimensions
    demographic_groups = {
        "Male (Hombre)": rejected_test_df[rejected_test_df["SEX"] == 1],
        "Female (Mujer)": rejected_test_df[rejected_test_df["SEX"] == 2],
        "Young (< 30)": rejected_test_df[rejected_test_df["AGE"] < 30],
        "Adultos (≥ 30)": rejected_test_df[rejected_test_df["AGE"] >= 30],
        "Higher Ed (Grad/Univ)": rejected_test_df[rejected_test_df["EDUCATION"].isin([1, 2])],
        "Secondary / Other": rejected_test_df[rejected_test_df["EDUCATION"].isin([3, 4])],
        "Married (Casado/a)": rejected_test_df[rejected_test_df["MARRIAGE"] == 1],
        "Single (Soltero/a)": rejected_test_df[rejected_test_df["MARRIAGE"] == 2],
    }

    group_results = {}
    group_burdens = {}

    for group_name, group_data in demographic_groups.items():
        print(f"\n[audit] Evaluating: {group_name} (Pool: {len(group_data)} rejected applicants)...")
        sample = group_data.sample(min(sample_per_group, len(group_data)), random_state=random_state)

        burdens = []
        cfs_found_count = 0
        unexpected_errors_count = 0

        for idx, row in sample.iterrows():
            instance_df = pd.DataFrame([row])
            try:
                _, deltas, mean_b = engine.generate_recourse_plans(instance_df, total_cfs=1, random_seed=random_state)
                if not deltas.empty and mean_b > 0:
                    burdens.append(mean_b)
                    cfs_found_count += 1
            except Exception as e:
                unexpected_errors_count += 1

        feasibility_rate = (cfs_found_count / len(sample)) * 100.0
        mean_b, ci_lower, ci_upper = compute_ci(burdens) if burdens else (0.0, 0.0, 0.0)
        median_b = float(np.median(burdens)) if burdens else 0.0
        std_b = float(np.std(burdens, ddof=1)) if len(burdens) > 1 else 0.0

        group_burdens[group_name] = burdens
        group_results[group_name] = {
            "group": group_name,
            "pool_size": int(len(group_data)),
            "evaluated_n": int(len(sample)),
            "successful_recourse_n": int(cfs_found_count),
            "infeasible_recourse_n": int(len(sample) - cfs_found_count),
            "unexpected_error_n": int(unexpected_errors_count),
            "feasibility_rate_pct": round(feasibility_rate, 1),
            "mean_burden": mean_b,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "median_burden": round(median_b, 2),
            "std_burden": round(std_b, 2),
        }
        print(f"  -> Feasibility Rate: {feasibility_rate:.1f}% ({cfs_found_count}/{len(sample)})")
        print(f"  -> Conditional Burden: {mean_b:.2f} (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])")

    # 5. Perform Formal Statistical Hypothesis Testing between Pairs
    pairs = [
        ("Gender", "Female (Mujer)", "Male (Hombre)", "Mujer / Hombre"),
        ("Age", "Young (< 30)", "Adultos (≥ 30)", "Jóvenes / Adultos (30+)"),
        ("Education", "Secondary / Other", "Higher Ed (Grad/Univ)", "Media / Superior"),
        ("Marriage", "Single (Soltero/a)", "Married (Casado/a)", "Soltero / Casado"),
    ]

    raw_comps = []
    welch_pvals = []
    fisher_pvals = []

    for dim_name, group_a, group_b, ratio_label in pairs:
        b_a = group_burdens[group_a]
        b_b = group_burdens[group_b]

        mean_a = group_results[group_a]["mean_burden"]
        mean_b = group_results[group_b]["mean_burden"]
        ratio = round(mean_a / mean_b, 3) if mean_b > 0 else 1.0

        # Welch's t-test (conditional burden)
        t_stat, t_pval = stats.ttest_ind(b_a, b_b, equal_var=False) if (b_a and b_b) else (0.0, 1.0)
        # Mann-Whitney U test (non-parametric conditional burden)
        u_stat, u_pval = stats.mannwhitneyu(b_a, b_b, alternative="two-sided") if (b_a and b_b) else (0.0, 1.0)

        # Fisher's Exact Test on Recourse Feasibility (2x2 contingency table)
        succ_a = group_results[group_a]["successful_recourse_n"]
        fail_a = group_results[group_a]["infeasible_recourse_n"]
        succ_b = group_results[group_b]["successful_recourse_n"]
        fail_b = group_results[group_b]["infeasible_recourse_n"]
        contingency_table = [[succ_a, fail_a], [succ_b, fail_b]]
        odds_ratio, f_pval = stats.fisher_exact(contingency_table)

        welch_pvals.append(float(t_pval))
        fisher_pvals.append(float(f_pval))

        raw_comps.append({
            "dimension": dim_name,
            "group_a": group_a,
            "group_b": group_b,
            "disparity_ratio": ratio,
            "ratio_label": ratio_label,
            "mean_difference": round(mean_a - mean_b, 2),
            "welch_t_stat": round(float(t_stat), 3),
            "welch_t_pvalue": round(float(t_pval), 4),
            "mann_whitney_u": round(float(u_stat), 1),
            "mann_whitney_pvalue": round(float(u_pval), 4),
            "feasibility_gap_pct": round(group_results[group_a]["feasibility_rate_pct"] - group_results[group_b]["feasibility_rate_pct"], 1),
            "fisher_odds_ratio": round(float(odds_ratio), 3),
            "fisher_feasibility_pvalue": round(float(f_pval), 4),
        })

    # Domain-specific Holm correction (4 tests each for burden and feasibility)
    welch_pvals_adj_domain = holm_bonferroni_correction(welch_pvals)
    fisher_pvals_adj_domain = holm_bonferroni_correction(fisher_pvals)

    # Global experiment-wide Holm correction (all 8 tests jointly)
    all_8_pvals = welch_pvals + fisher_pvals
    all_8_adj = holm_bonferroni_correction(all_8_pvals)
    welch_pvals_adj_global = all_8_adj[:len(welch_pvals)]
    fisher_pvals_adj_global = all_8_adj[len(welch_pvals):]

    dimension_comparisons = {}
    for idx, c in enumerate(raw_comps):
        dim_key = c["dimension"].lower()
        t_pval = c["welch_t_pvalue"]
        t_adj_domain = welch_pvals_adj_domain[idx]
        t_adj_global = welch_pvals_adj_global[idx]

        f_pval = c["fisher_feasibility_pvalue"]
        f_adj_domain = fisher_pvals_adj_domain[idx]
        f_adj_global = fisher_pvals_adj_global[idx]

        # DECISION DRIVEN STRICTLY BY HOLM-ADJUSTED P-VALUE
        is_burden_sig = bool(t_adj_domain < 0.05)
        is_feasibility_sig = bool(f_adj_domain < 0.05)

        burden_conclusion = (
            f"Diferencia estadísticamente significativa en carga (p = {t_pval:.4f}, p_adj = {t_adj_domain:.4f} < 0.05)."
            if is_burden_sig else
            f"Sin evidencia de disparidad en carga algorítmica (p = {t_pval:.4f}, p_adj = {t_adj_domain:.4f} >= 0.05); no se rechaza H0."
        )

        feasibility_conclusion = (
            f"Brecha de factibilidad estadísticamente significativa (p = {f_pval:.4f}, p_adj = {f_adj_domain:.4f} < 0.05)."
            if is_feasibility_sig else
            f"Sin brecha estadísticamente significativa en factibilidad (p = {f_pval:.4f}, p_adj = {f_adj_domain:.4f} >= 0.05)."
        )

        c["welch_t_pvalue_holm_adjusted"] = t_adj_domain
        c["welch_t_pvalue_holm_global_8tests"] = t_adj_global
        c["fisher_pvalue_holm_adjusted"] = f_adj_domain
        c["fisher_pvalue_holm_global_8tests"] = f_adj_global
        c["is_burden_statistically_significant"] = is_burden_sig
        c["is_feasibility_statistically_significant"] = is_feasibility_sig
        c["is_statistically_significant"] = bool(is_burden_sig or is_feasibility_sig)
        c["burden_conclusion"] = burden_conclusion
        c["feasibility_conclusion"] = feasibility_conclusion
        c["conclusion"] = f"{burden_conclusion} {feasibility_conclusion}"

        dimension_comparisons[dim_key] = c

    # 6. Overall Feasibility & Audit Summary
    all_evaluated = sum(g["evaluated_n"] for g in group_results.values())
    all_successful = sum(g["successful_recourse_n"] for g in group_results.values())
    overall_feasibility = round((all_successful / all_evaluated) * 100.0, 1) if all_evaluated > 0 else 0.0

    all_burdens = [b for b_list in group_burdens.values() for b in b_list]
    overall_mean, ov_ci_l, ov_ci_u = compute_ci(all_burdens) if all_burdens else (0.0, 0.0, 0.0)

    audit_summary = {
        "metadata": {
            "test_sample_size": int(len(X_test)),
            "model_rejected_sample_size": int(len(rejected_test_df)),
            "threshold": threshold,
            "sample_per_group": sample_per_group,
            "actionable_features": ACTIONABLE_FEATURES,
            "immutable_features": IMMUTABLE_FEATURES,
            "historical_features": HISTORICAL_FEATURES,
            "institutional_features": INSTITUTIONAL_FEATURES,
            "methodology_notes": [
                "Algorithmic burden is evaluated conditionally on applicants who achieve feasible recourse within strict constraints (PAY_1 <= min(0, current), BILL_AMT1 <= current, PAY_AMT1 <= plan bill).",
                "SURVIVORSHIP BIAS WARNING: Conditional burden ratios below 1.00x should NOT be interpreted as lighter burden if accompanied by lower feasibility. Groups with the most financially distressed applicants see harder cases dropped as infeasible, which can artificially depress conditional burden.",
                "HYPOTHESIS TESTING ARCHITECTURE: We report both (1) domain-specific 4-test Holm-Bonferroni corrections (treating conditional burden and feasibility as two distinct 4-hypothesis families) and (2) global experiment-wide 8-test Holm-Bonferroni corrections covering all simultaneous tests. Significance decisions are strictly governed by adjusted p-values."
            ]
        },
        "overall": {
            "evaluated_total": all_evaluated,
            "successful_total": all_successful,
            "feasibility_rate_pct": overall_feasibility,
            "mean_burden": overall_mean,
            "ci_95_lower": ov_ci_l,
            "ci_95_upper": ov_ci_u
        },
        "subgroups": group_results,
        "comparisons": dimension_comparisons
    }

    # 7. Save Structured JSON Output
    json_path = os.path.join(output_dir, "audit_disparity_results.json")
    with open(json_path, "w") as f:
        json.dump(audit_summary, f, indent=4)
    print(f"\n[audit] Audit results successfully saved to: {json_path}")

    # 8. Generate Publication-Quality Figure with 95% Confidence Intervals
    plot_data = []
    for g_name, g_info in group_results.items():
        plot_data.append({
            "Subgroup": g_name,
            "Mean Burden": g_info["mean_burden"],
            "CI Lower": g_info["ci_95_lower"],
            "CI Upper": g_info["ci_95_upper"],
            "Error Lower": g_info["mean_burden"] - g_info["ci_95_lower"],
            "Error Upper": g_info["ci_95_upper"] - g_info["mean_burden"],
            "Category": (
                "Gender" if "Hombre" in g_name or "Mujer" in g_name else
                "Age" if "30" in g_name else
                "Education" if "Ed" in g_name or "Secondary" in g_name else
                "Marriage"
            )
        })
    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(11, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")

    palette = {
        "Gender": "#2563eb",
        "Age": "#d97706",
        "Education": "#7c3aed",
        "Marriage": "#059669"
    }

    colors = [palette[c] for c in df_plot["Category"]]

    ax = sns.barplot(
        data=df_plot,
        x="Subgroup",
        y="Mean Burden",
        palette=colors,
        hue="Subgroup",
        legend=False
    )

    # Add error bars for 95% CI
    yerr = [df_plot["Error Lower"].values, df_plot["Error Upper"].values]
    ax.errorbar(
        x=range(len(df_plot)),
        y=df_plot["Mean Burden"],
        yerr=yerr,
        fmt="none",
        c="black",
        capsize=4,
        elinewidth=1.2
    )

    for idx, row in df_plot.iterrows():
        mean_val = row["Mean Burden"]
        err_val = row["Error Upper"]
        ax.annotate(
            f"{mean_val:.2f}\n[±{err_val:.2f}]",
            (idx, mean_val + err_val),
            ha="center", va="bottom", xytext=(0, 4),
            textcoords="offset points", fontweight="bold", fontsize=8.5
        )

    legend_elements = [Patch(facecolor=color, label=label) for label, color in palette.items()]
    ax.legend(handles=legend_elements, title="Demographic Dimension", fontsize=9, title_fontsize=9.5, loc="upper right")

    plt.title(
        "Algorithmic Recourse Burden across Demographic Dimensions\n"
        "(Conditional Normalized L1 Distance on Hold-out Test Set with 95% Confidence Intervals)",
        fontsize=12, fontweight="bold", pad=14
    )
    plt.xlabel("Demographic Subgroup", fontsize=10, fontweight="bold")
    plt.ylabel("Mean Conditional L1 Burden", fontsize=10, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.ylim(0, max(df_plot["CI Upper"]) * 1.25)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "audit_disparity_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[audit] Audit comparison plot with CIs saved to: {plot_path}")

    return audit_summary


if __name__ == "__main__":
    summary = run_demographic_recourse_audit(sample_per_group=80)
    print("\n--- AUDIT HYPOTHESIS TESTING SUMMARY (WITH HOLM-BONFERRONI CORRECTION) ---")
    for dim, comp in summary["comparisons"].items():
        print(f"\nDimension: {comp['dimension'].upper()}")
        print(f"  Ratio ({comp['ratio_label']}): {comp['disparity_ratio']}x (Mean diff: {comp['mean_difference']:+.2f})")
        print(f"  Conditional Burden Welch t-test: t = {comp['welch_t_stat']}, p_raw = {comp['welch_t_pvalue']}, p_holm = {comp['welch_t_pvalue_holm_adjusted']}")
        print(f"  Feasibility Fisher Exact Test:   gap = {comp['feasibility_gap_pct']:+.1f}%, p_raw = {comp['fisher_feasibility_pvalue']}, p_holm = {comp['fisher_pvalue_holm_adjusted']}")
        print(f"  Conclusion: {comp['conclusion']}")
