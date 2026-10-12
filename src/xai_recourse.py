"""
=============================================================================
MODULE: xai_recourse.py
PROJECT: AIPI 590 - Project 1: ¿Por qué me rechazaron?
AUTHOR: Tonantzin Real Rojas
DESCRIPTION:
    Algorithmic Recourse engine using Microsoft's DiCE (Diverse Counterfactual
    Explanations). Implements domain-grounded actionability constraints:
      - Freezes immutable demographic features (SEX, AGE, MARRIAGE, EDUCATION).
      - Constrains mutable financial levers (BILL_AMT, PAY_AMT, PAY_1).
      - Computes normalized Algorithmic Burden (L1 effort distance).
=============================================================================
"""

import os
import joblib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import dice_ml

# Relative import
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import (
    download_and_clean_data,
    IMMUTABLE_FEATURES,
    ACTIONABLE_FEATURES,
    SEMI_MUTABLE_FEATURES
)


class CreditRecourseEngine:
    """
    Generates actionable counterfactual recourse plans for rejected loan applicants
    subject to realistic financial and demographic constraints.
    """

    def __init__(
        self,
        model_path: str = "models/credit_model.joblib",
        data_path: str = "data/credit_card_clean.csv",
        sample_size: int = 2000
    ):
        """
        Initializes the DiCE explainer with background reference data and model.

        Parameters:
            model_path (str): Path to trained credit classification model.
            data_path (str): Path to clean training dataset.
            sample_size (int): Background sample size for fast counterfactual search.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}.")

        self.model = joblib.load(model_path)
        self.df = pd.read_csv(data_path)

        # Compute feature standard deviations for normalized L1 recourse distance
        self.feature_stds = self.df.drop(columns=["DEFAULT"]).std()

        # Identify continuous vs categorical features for DiCE
        continuous_cols = [
            c for c in self.df.columns
            if c not in ["DEFAULT", "SEX", "EDUCATION", "MARRIAGE"]
        ]

        # Use a representative background sample for KD-Tree / Random search speed
        background_sample = self.df.sample(min(sample_size, len(self.df)), random_state=42)

        self.d = dice_ml.Data(
            dataframe=background_sample,
            continuous_features=continuous_cols,
            outcome_name="DEFAULT"
        )
        self.m = dice_ml.Model(model=self.model, backend="sklearn")
        
        # 'random' sampling method is fast, robust, and supports strict feature constraints
        self.exp = dice_ml.Dice(self.d, self.m, method="random")
        print("[xai_recourse] DiCE Recourse Engine successfully initialized.")

    def generate_recourse_plans(
        self,
        query_instance: pd.DataFrame,
        total_cfs: int = 2,
        custom_features_to_vary: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Generates actionable counterfactual paths to flip a rejection (DEFAULT=1)
        into an approval (DEFAULT=0), strictly preserving immutable attributes.

        Parameters:
            query_instance (pd.DataFrame): 1-row DataFrame of the rejected applicant.
            total_cfs (int): Number of diverse counterfactual recommendations to generate.
            custom_features_to_vary (list, optional): Overrides mutable feature list if specified.

        Returns:
            cf_df (pd.DataFrame): The full counterfactual applicant profiles.
            deltas_df (pd.DataFrame): Clean table highlighting ONLY the necessary changes.
            mean_burden (float): Average normalized L1 recourse effort required.
        """
        # Exclude immutable features: SEX, AGE, MARRIAGE, EDUCATION must NEVER be varied
        if custom_features_to_vary is None:
            features_to_vary = [
                c for c in query_instance.columns
                if c not in IMMUTABLE_FEATURES
            ]
        else:
            features_to_vary = [c for c in custom_features_to_vary if c not in IMMUTABLE_FEATURES]

        # Generate diverse counterfactuals targeting DEFAULT=0 (Approved)
        dice_exp = self.exp.generate_counterfactuals(
            query_instance,
            total_CFs=total_cfs,
            desired_class=0,
            features_to_vary=features_to_vary
        )

        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
        if cf_df is None or len(cf_df) == 0:
            print("[xai_recourse] Warning: No valid counterfactual found within current constraints.")
            return pd.DataFrame(), pd.DataFrame(), 0.0

        # Calculate deltas (what specifically needs to change)
        query_row = query_instance.iloc[0]
        delta_records = []

        for idx, cf_row in cf_df.iterrows():
            changes = {}
            total_l1_burden = 0.0

            for col in features_to_vary:
                orig_val = query_row[col]
                cf_val = cf_row[col]

                # If the value changes significantly (more than 1% or 1 integer unit)
                if abs(cf_val - orig_val) > 0.01:
                    changes[col] = f"{orig_val:,.0f} ➔ {cf_val:,.0f} (Δ {cf_val - orig_val:+,.0f})"
                    # Normalized L1 burden: effort scaled by feature standard deviation
                    col_std = self.feature_stds.get(col, 1.0)
                    if col_std > 0:
                        total_l1_burden += abs(cf_val - orig_val) / col_std

            changes["Plan_ID"] = f"Plan {idx + 1}"
            changes["Normalized_Burden"] = round(total_l1_burden, 2)
            delta_records.append(changes)

        deltas_df = pd.DataFrame(delta_records)
        mean_burden = float(deltas_df["Normalized_Burden"].mean()) if "Normalized_Burden" in deltas_df else 0.0

        return cf_df, deltas_df, mean_burden


if __name__ == "__main__":
    # Test script standalone execution
    df = pd.read_csv("data/credit_card_clean.csv")
    engine = CreditRecourseEngine()

    # Find a sample rejected applicant (DEFAULT = 1)
    rejected_sample = df[df["DEFAULT"] == 1].drop(columns=["DEFAULT"]).iloc[[0]]
    print("\nRejected Applicant Profile (Original):")
    print(rejected_sample[["LIMIT_BAL", "SEX", "AGE", "PAY_1", "BILL_AMT1", "PAY_AMT1"]])

    cfs, deltas, burden = engine.generate_recourse_plans(rejected_sample, total_cfs=2)
    print("\n--- Actionable Recourse Recommendations (Deltas) ---")
    print(deltas)
    print(f"\nMean Algorithmic Burden (Normalized L1 Distance): {burden:.2f}")
