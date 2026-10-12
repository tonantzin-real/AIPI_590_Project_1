"""
=============================================================================
MODULE: audit.py
PROJECT: AIPI 590 - Project 1: ¿Por qué me rechazaron?
AUTHOR: Tonantzin Real Rojas
DESCRIPTION:
    Audits Algorithmic Recourse Disparity and Burden Inequality across
    demographic groups (Gender and Age cohorts). Evaluates whether the
    recourse engine inadvertently imposes an unequal financial burden on
    vulnerable consumer segments.
=============================================================================
"""

import os
import json
import joblib
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Relative import
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.xai_recourse import CreditRecourseEngine
from src.data_loader import IMMUTABLE_FEATURES


def run_demographic_recourse_audit(
    sample_per_group: int = 50,
    output_dir: str = "models",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs an empirical audit of Algorithmic Burden across:
      1. Gender (Hombre / Male vs. Mujer / Female)
      2. Age Cohorts (Jóvenes / Under 30 vs. Mayores / 30 and older)

    Parameters:
        sample_per_group (int): Number of rejected instances to evaluate per cohort.
        output_dir (str): Directory to save plots and JSON audit results.
        random_state (int): Reproducibility seed.

    Returns:
        dict: Summary statistics including mean burden, std, and disparity ratios.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv("data/credit_card_clean.csv")
    engine = CreditRecourseEngine()

    # Isolate rejected applicants (DEFAULT = 1)
    rejected_df = df[df["DEFAULT"] == 1].copy()

    # Define Demographic Groups
    groups = {
        "Male (Hombre)": rejected_df[rejected_df["SEX"] == 1],
        "Female (Mujer)": rejected_df[rejected_df["SEX"] == 2],
        "Young (< 30)": rejected_df[rejected_df["AGE"] < 30],
        "Older (>= 30)": rejected_df[rejected_df["AGE"] >= 30]
    }

    audit_records = []
    print("\n" + "="*60)
    print("STARTING DEMOGRAPHIC ALGORITHMIC RECOURSE AUDIT")
    print("="*60)

    for group_name, group_data in groups.items():
        print(f"\n[audit] Auditing group: {group_name} (Total pool: {len(group_data)} applicants)...")
        # Sample representative batch
        sample = group_data.sample(min(sample_per_group, len(group_data)), random_state=random_state)
        sample_features = sample.drop(columns=["DEFAULT"])

        burdens = []
        cfs_found_count = 0

        for idx, row in sample_features.iterrows():
            instance_df = pd.DataFrame([row])
            try:
                _, deltas, mean_burden = engine.generate_recourse_plans(instance_df, total_cfs=1)
                if not deltas.empty and mean_burden > 0:
                    burdens.append(mean_burden)
                    cfs_found_count += 1
            except Exception as e:
                continue

        mean_b = float(np.mean(burdens)) if burdens else 0.0
        median_b = float(np.median(burdens)) if burdens else 0.0
        std_b = float(np.std(burdens)) if burdens else 0.0
        feasibility_rate = (cfs_found_count / len(sample)) * 100

        print(f"  -> Mean Algorithmic Burden: {mean_b:.2f}")
        print(f"  -> Median Burden:          {median_b:.2f}")
        print(f"  -> Recourse Success Rate:  {feasibility_rate:.1f}%")

        audit_records.append({
            "Group": group_name,
            "Mean_Burden": round(mean_b, 2),
            "Median_Burden": round(median_b, 2),
            "Std_Burden": round(std_b, 2),
            "Feasibility_Rate_Pct": round(feasibility_rate, 1),
            "Evaluated_N": len(burdens)
        })

    audit_df = pd.DataFrame(audit_records)

    # Compute Disparity Ratios
    # Disparity Ratio = Female Mean Burden / Male Mean Burden
    male_mean = audit_df.loc[audit_df["Group"] == "Male (Hombre)", "Mean_Burden"].values[0]
    female_mean = audit_df.loc[audit_df["Group"] == "Female (Mujer)", "Mean_Burden"].values[0]
    gender_disparity = round(female_mean / male_mean, 3) if male_mean > 0 else 1.0

    young_mean = audit_df.loc[audit_df["Group"] == "Young (< 30)", "Mean_Burden"].values[0]
    older_mean = audit_df.loc[audit_df["Group"] == "Older (>= 30)", "Mean_Burden"].values[0]
    age_disparity = round(young_mean / older_mean, 3) if older_mean > 0 else 1.0

    audit_summary = {
        "groups": audit_records,
        "gender_disparity_ratio": gender_disparity,
        "age_disparity_ratio": age_disparity,
        "interpretation": {
            "gender": f"Female applicants require {gender_disparity:.2f}x the algorithmic effort of male applicants to achieve approval.",
            "age": f"Younger applicants require {age_disparity:.2f}x the algorithmic effort of older applicants to achieve approval."
        }
    }

    # Save JSON summary
    json_path = os.path.join(output_dir, "audit_disparity_results.json")
    with open(json_path, "w") as f:
        json.dump(audit_summary, f, indent=4)
    print(f"\n[audit] Results saved to: {json_path}")

    # Generate Publication-Ready Comparison Plot for Poster
    plt.figure(figsize=(9, 5), dpi=300)
    sns.set_theme(style="whitegrid")
    palette = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]
    ax = sns.barplot(
        data=audit_df,
        x="Group",
        y="Mean_Burden",
        palette=palette,
        hue="Group",
        legend=False
    )
    plt.title("Algorithmic Burden Disparity by Demographic Subgroup\n(Normalized L1 Recourse Effort to Flip Rejection)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Demographic Group", fontsize=11, fontweight="bold")
    plt.ylabel("Mean Normalized L1 Burden", fontsize=11, fontweight="bold")

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 7),
                    textcoords="offset points", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "audit_disparity_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[audit] Audit comparison plot saved for poster to: {plot_path}")

    return audit_summary


if __name__ == "__main__":
    # Run audit on representative sample
    summary = run_demographic_recourse_audit(sample_per_group=30)
    print("\n--- Final Audit Summary ---")
    print(json.dumps(summary, indent=2))
