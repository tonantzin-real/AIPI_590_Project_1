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
      - Enforces strict actionability constraints (immutable, historical, and institutional traits locked).
      - Computes empirical feasibility rates (% receiving viable recourse).
      - Performs formal statistical hypothesis tests (Welch's t-test and Mann-Whitney U test).
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

# Relative import with robust base directory resolution
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from src.data_loader import get_train_test_data, DEFAULT_DATA_PATH
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

    # 1. Load out-of-sample test set (prevents training data leakage)
    _, X_test, _, y_test = get_train_test_data(random_state=random_state)
    model = joblib.load(model_path)

    # 2. Identify applicants ACTUALLY rejected by the model decision (P(default) >= 0.50)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    threshold = 0.50
    model_rejected_mask = (y_pred_proba >= threshold)
    rejected_test_df = X_test[model_rejected_mask].copy()

    print("\n" + "=" * 65)
    print("STARTING RESPONSIBLE AI DEMOGRAPHIC RECOURSE AUDIT")
    print(f"Total Test Set Size:               {len(X_test)} applicants")
    print(f"Model-Predicted Denials in Test:   {len(rejected_test_df)} ({len(rejected_test_df)/len(X_test):.1%})")
    print(f"Sample Evaluated per Subgroup:     {sample_per_group} applicants")
    print("=" * 65)

    engine = CreditRecourseEngine(model_path=model_path, random_state=random_state)

    # 3. Define Subgroups across 4 Demographic Dimensions
    demographic_groups = {
        "Male (Hombre)": rejected_test_df[rejected_test_df["SEX"] == 1],
        "Female (Mujer)": rejected_test_df[rejected_test_df["SEX"] == 2],
        "Young (< 30)": rejected_test_df[rejected_test_df["AGE"] < 30],
        "Older (>= 30)": rejected_test_df[rejected_test_df["AGE"] >= 30],
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

        for idx, row in sample.iterrows():
            instance_df = pd.DataFrame([row])
            _, deltas, mean_b = engine.generate_recourse_plans(instance_df, total_cfs=1, random_seed=random_state)
            if not deltas.empty and mean_b > 0:
                burdens.append(mean_b)
                cfs_found_count += 1

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
            "feasibility_rate_pct": round(feasibility_rate, 1),
            "mean_burden": mean_b,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "median_burden": round(median_b, 2),
            "std_burden": round(std_b, 2),
        }
        print(f"  -> Feasibility Rate: {feasibility_rate:.1f}% ({cfs_found_count}/{len(sample)})")
        print(f"  -> Mean Burden:     {mean_b:.2f} (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])")

    # 4. Perform Formal Statistical Hypothesis Testing between Pairs
    pairs = [
        ("Gender", "Female (Mujer)", "Male (Hombre)", "Female / Male"),
        ("Age", "Young (< 30)", "Older (>= 30)", "Young / Older"),
        ("Education", "Secondary / Other", "Higher Ed (Grad/Univ)", "Secondary / Higher"),
        ("Marriage", "Single (Soltero/a)", "Married (Casado/a)", "Single / Married"),
    ]

    dimension_comparisons = {}
    for dim_name, group_a, group_b, ratio_label in pairs:
        b_a = group_burdens[group_a]
        b_b = group_burdens[group_b]

        mean_a = group_results[group_a]["mean_burden"]
        mean_b = group_results[group_b]["mean_burden"]
        ratio = round(mean_a / mean_b, 3) if mean_b > 0 else 1.0

        # Welch's t-test (unequal variances assumed)
        t_stat, t_pval = stats.ttest_ind(b_a, b_b, equal_var=False)
        # Mann-Whitney U test (non-parametric test of distributions)
        u_stat, u_pval = stats.mannwhitneyu(b_a, b_b, alternative="two-sided")

        is_significant = bool(t_pval < 0.05)
        conclusion = (
            f"Statistically significant disparity detected (p = {t_pval:.4f} < 0.05)."
            if is_significant else
            f"No statistically significant disparity (p = {t_pval:.4f} >= 0.05); observed difference is within sampling noise."
        )

        dimension_comparisons[dim_name.lower()] = {
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
            "is_statistically_significant": is_significant,
            "conclusion": conclusion
        }

    # 5. Overall Feasibility & Audit Summary
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
            "actionable_features": ["PAY_1", "BILL_AMT1", "PAY_AMT1"],
            "locked_features": ["SEX", "AGE", "EDUCATION", "MARRIAGE", "LIMIT_BAL", "PAY_2..6", "BILL_AMT2..6", "PAY_AMT2..6"]
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

    # 6. Save Structured JSON Output
    json_path = os.path.join(output_dir, "audit_disparity_results.json")
    with open(json_path, "w") as f:
        json.dump(audit_summary, f, indent=4)
    print(f"\n[audit] Audit results successfully saved to: {json_path}")

    # 7. Generate Publication-Quality Figure with 95% Confidence Intervals
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

    from matplotlib.patches import Patch
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
        "Algorithmic Recourse Burden Disparity across Demographic Dimensions\n"
        "(Normalized L1 Distance on Hold-out Test Set with 95% Confidence Intervals)",
        fontsize=12, fontweight="bold", pad=14
    )
    plt.xlabel("Demographic Subgroup", fontsize=10, fontweight="bold")
    plt.ylabel("Mean Normalized L1 Burden", fontsize=10, fontweight="bold")
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
    print("\n--- AUDIT HYPOTHESIS TESTING SUMMARY ---")
    for dim, comp in summary["comparisons"].items():
        print(f"\nDimension: {comp['dimension'].upper()}")
        print(f"  Ratio ({comp['ratio_label']}): {comp['disparity_ratio']}x (Mean diff: {comp['mean_difference']:+.2f})")
        print(f"  Welch t-test: t = {comp['welch_t_stat']}, p = {comp['welch_t_pvalue']}")
        print(f"  Conclusion:   {comp['conclusion']}")
