"""
=============================================================================
MODULE: xai_shap.py
PROJECT: AIPI 590 - Project 1: Why Was I Rejected?
AUTHOR: Tonantzin Real Rojas
DESCRIPTION:
    Computes global and local feature attributions using TreeSHAP for the
    credit scoring classifier. Explains *why* an applicant was rejected from
    a retrospective perspective, serving as the benchmark that we compare
    against prescriptive algorithmic recourse (DiCE).

    Also audits SHAP "actionability": for every model-rejected test applicant,
    how many of SHAP's top reasons for the rejection are features the applicant
    can actually change today (research question 1).
=============================================================================
"""

import os
import json
import joblib
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Relative import
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from src.data_loader import (
    get_train_test_data,
    IMMUTABLE_FEATURES,
    HISTORICAL_FEATURES,
    INSTITUTIONAL_FEATURES,
    ACTIONABLE_FEATURES,
)

DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_model.joblib")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "models")


def feature_category(feature: str) -> str:
    """Classifies a feature by whether the applicant can change it today."""
    if feature in ACTIONABLE_FEATURES:
        return "actionable"
    if feature in IMMUTABLE_FEATURES:
        return "immutable"
    if feature in HISTORICAL_FEATURES:
        return "historical"
    if feature in INSTITUTIONAL_FEATURES:
        return "institutional"
    return "other"


class CreditSHAPExplainer:
    """
    Wrapper around TreeSHAP for tree-based credit default models.
    Provides global feature importance summaries and local applicant-level
    explanations.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        """
        Loads the serialized credit risk model and initializes TreeExplainer.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Run train_model.py first.")

        self.model = joblib.load(model_path)
        # TreeExplainer is exact, fast, and optimized for LightGBM/XGBoost
        self.explainer = shap.TreeExplainer(self.model)
        print("[xai_shap] TreeExplainer successfully initialized.")

    def explain_instance(self, instance_df: pd.DataFrame) -> Tuple[np.ndarray, float, pd.DataFrame]:
        """
        Generates local SHAP attribution values for a single credit applicant.

        Parameters:
            instance_df (pd.DataFrame): 1-row DataFrame containing applicant features.

        Returns:
            shap_values (np.ndarray): SHAP values for class 1 (Default/Risk).
            base_value (float): Model base (expected) value log-odds / margin.
            summary_df (pd.DataFrame): Sorted table of top positive & negative driving factors.
        """
        # shap_values returns array of shape (1, n_features) or (1, n_features, 2)
        raw_shap = self.explainer(instance_df)
        
        # Handle multi-output vs single-output trees
        if len(raw_shap.values.shape) == 3:
            # Take class 1 (Default / High Risk)
            values = raw_shap.values[0, :, 1]
            base_value = raw_shap.base_values[0, 1]
        else:
            values = raw_shap.values[0]
            base_value = raw_shap.base_values[0]

        summary_df = pd.DataFrame({
            "Feature": instance_df.columns,
            "Value": instance_df.iloc[0].values,
            "SHAP_Attribution": values,
            "Absolute_Impact": np.abs(values)
        }).sort_values(by="Absolute_Impact", ascending=False)

        # Categorize impact direction
        summary_df["Direction"] = summary_df["SHAP_Attribution"].apply(
            lambda x: "Increases Default Risk (Negative)" if x > 0 else "Lowers Default Risk (Positive)"
        )

        return values, float(base_value), summary_df

    def compute_global_importance(self, X_sample: pd.DataFrame) -> pd.DataFrame:
        """
        Computes mean absolute SHAP values across an evaluation sample.

        Parameters:
            X_sample (pd.DataFrame): Representative batch of test instances.

        Returns:
            pd.DataFrame: Global feature rankings sorted by importance.
        """
        shap_values = self.explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            # Class 1 (default)
            vals = shap_values[1]
        else:
            vals = shap_values

        mean_abs_shap = np.abs(vals).mean(axis=0)
        global_df = pd.DataFrame({
            "Feature": X_sample.columns,
            "Mean_Abs_SHAP": mean_abs_shap
        }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

        return global_df


def audit_shap_actionability(
    model_path: str = DEFAULT_MODEL_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    top_k: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Measures how actionable SHAP's explanations are for rejected applicants.

    For every model-rejected applicant in the hold-out test set, takes the top_k
    features that INCREASE default risk the most (SHAP > 0) and classifies each as
    actionable (PAY_1, BILL_AMT1, PAY_AMT1), immutable (demographics), historical
    (past months), or institutional (bank-set credit limit).

    Parameters:
        model_path (str): Path to trained credit classification model.
        output_dir (str): Directory where the JSON results are saved.
        top_k (int): Number of top risk-increasing SHAP features per applicant.
        random_state (int): Seed for the train/test split.

    Returns:
        dict: Share of top-k reasons per category and applicant-level summaries.
    """
    os.makedirs(output_dir, exist_ok=True)

    threshold = 0.50
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            threshold = json.load(f).get("threshold", 0.50)

    _, X_test, _, _ = get_train_test_data(random_state=random_state)
    explainer = CreditSHAPExplainer(model_path=model_path)
    rejected = X_test[explainer.model.predict_proba(X_test)[:, 1] >= threshold]

    raw_shap = explainer.explainer(rejected)
    values = raw_shap.values[:, :, 1] if len(raw_shap.values.shape) == 3 else raw_shap.values
    features = np.array(rejected.columns)

    category_counts = {"actionable": 0, "immutable": 0, "historical": 0, "institutional": 0, "other": 0}
    top1_counts: Dict[str, int] = {}
    total_slots = 0
    no_actionable_in_top_k = 0
    top1_not_actionable = 0
    demographic_in_top_k = 0

    for row in values:
        order = np.argsort(-row)
        top_features = [features[i] for i in order[:top_k] if row[i] > 0]
        if not top_features:
            continue
        categories = [feature_category(f) for f in top_features]
        for c in categories:
            category_counts[c] += 1
        total_slots += len(categories)
        no_actionable_in_top_k += int("actionable" not in categories)
        top1_not_actionable += int(categories[0] != "actionable")
        demographic_in_top_k += int("immutable" in categories)
        top1_counts[top_features[0]] = top1_counts.get(top_features[0], 0) + 1

    n = len(values)
    results = {
        "metadata": {
            "rejected_applicants": int(n),
            "threshold": threshold,
            "top_k": top_k,
            "actionable_features": ACTIONABLE_FEATURES,
        },
        "share_of_top_k_reasons_pct": {
            c: round(100.0 * k / total_slots, 1) for c, k in category_counts.items() if total_slots
        },
        "applicants_with_no_actionable_reason_in_top_k_pct": round(100.0 * no_actionable_in_top_k / n, 1),
        "applicants_whose_top_reason_is_not_actionable_pct": round(100.0 * top1_not_actionable / n, 1),
        "applicants_with_demographic_reason_in_top_k_pct": round(100.0 * demographic_in_top_k / n, 1),
        "most_common_top_reasons": dict(sorted(top1_counts.items(), key=lambda x: -x[1])[:5]),
    }

    json_path = os.path.join(output_dir, "shap_actionability_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[xai_shap] SHAP actionability results saved to: {json_path}")
    return results


if __name__ == "__main__":
    results = audit_shap_actionability()
    print(f"\n--- SHAP Actionability Audit ({results['metadata']['rejected_applicants']} rejected applicants) ---")
    print(f"Share of top-{results['metadata']['top_k']} SHAP reasons by category: {results['share_of_top_k_reasons_pct']}")
    print(f"Top reason is NOT actionable:                {results['applicants_whose_top_reason_is_not_actionable_pct']}%")
    print(f"No actionable reason in top-{results['metadata']['top_k']}:             {results['applicants_with_no_actionable_reason_in_top_k_pct']}%")
    print(f"Demographic trait among top-{results['metadata']['top_k']} reasons:     {results['applicants_with_demographic_reason_in_top_k_pct']}%")
    print(f"Most common top reasons: {results['most_common_top_reasons']}")
