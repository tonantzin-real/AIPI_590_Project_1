"""
=============================================================================
MODULE: xai_shap.py
PROJECT: AIPI 590 - Project 1: ¿Por qué me rechazaron?
AUTHOR: Tonantzin Real Rojas
DESCRIPTION:
    Computes global and local feature attributions using TreeSHAP for the
    credit scoring classifier. Explains *why* an applicant was rejected from
    a retrospective perspective, serving as the benchmark that we compare
    against prescriptive algorithmic recourse (DiCE).
=============================================================================
"""

import os
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
from src.data_loader import get_train_test_data, RAW_COLUMN_MAPPING

DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_model.joblib")


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


if __name__ == "__main__":
    # Test script standalone execution
    _, X_test, _, y_test = get_train_test_data()
    explainer = CreditSHAPExplainer()

    # Find a sample rejected applicant (predicted default = 1)
    sample_applicant = X_test.iloc[[0]]
    shap_vals, base_val, summary_df = explainer.explain_instance(sample_applicant)

    print("\n--- Local SHAP Explanation (Sample Applicant) ---")
    print(f"Base Value: {base_val:.4f}")
    print("\nTop 5 Influential Features:")
    print(summary_df.head(5)[["Feature", "Value", "SHAP_Attribution", "Direction"]])
