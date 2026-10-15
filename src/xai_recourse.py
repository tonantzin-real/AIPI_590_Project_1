"""
=============================================================================
MODULE: xai_recourse.py
PROJECT: AIPI 590 - Project 1: ¿Por qué me rechazaron?
AUTHOR: Tonantzin Real Rojas
DESCRIPTION:
    Algorithmic Recourse engine using Microsoft's DiCE (Diverse Counterfactual
    Explanations). Implements domain-grounded actionability constraints:
      - Freezes immutable demographic features (SEX, AGE, MARRIAGE, EDUCATION).
      - Freezes historical payment records (PAY_2..6, BILL_AMT2..6, PAY_AMT2..6).
      - Freezes institutional parameters (LIMIT_BAL set by card issuer).
      - Constrains mutable financial levers (PAY_1, BILL_AMT1, PAY_AMT1).
      - Enforces directional validity (debt reduction, on-time payment).
      - Computes normalized Algorithmic Burden (L1 effort distance).
=============================================================================
"""

import os
import joblib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import dice_ml

# Relative import with robust base directory resolution
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from src.data_loader import (
    DEFAULT_DATA_PATH,
    IMMUTABLE_FEATURES,
    HISTORICAL_FEATURES,
    INSTITUTIONAL_FEATURES,
    ACTIONABLE_FEATURES,
)

DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_model.joblib")


class CreditRecourseEngine:
    """
    Generates actionable counterfactual recourse plans for rejected loan applicants
    subject to realistic financial and demographic constraints.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        data_path: str = DEFAULT_DATA_PATH,
        sample_size: int = 2000,
        random_state: int = 42
    ):
        """
        Initializes the DiCE explainer with background reference data and model.

        Parameters:
            model_path (str): Path to trained credit classification model.
            data_path (str): Path to clean training dataset.
            sample_size (int): Background sample size for fast counterfactual search.
            random_state (int): Reproducibility seed for sampling reference data.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}.")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found at {data_path}.")

        self.model = joblib.load(model_path)
        self.df = pd.read_csv(data_path)
        self.random_state = random_state

        # Compute feature standard deviations for normalized L1 recourse distance
        self.feature_stds = self.df.drop(columns=["DEFAULT"]).std()

        # Identify continuous vs categorical features for DiCE
        continuous_cols = [
            c for c in self.df.columns
            if c not in ["DEFAULT", "SEX", "EDUCATION", "MARRIAGE"]
        ]

        # Use a representative background sample for KD-Tree / Random search speed
        background_sample = self.df.sample(min(sample_size, len(self.df)), random_state=random_state)

        self.d = dice_ml.Data(
            dataframe=background_sample,
            continuous_features=continuous_cols,
            outcome_name="DEFAULT"
        )
        self.m = dice_ml.Model(model=self.model, backend="sklearn")

        # 'random' sampling method is fast, robust, and supports strict feature constraints
        self.exp = dice_ml.Dice(self.d, self.m, method="random")
        print("[xai_recourse] DiCE Recourse Engine successfully initialized with strict actionability constraints.")

    def generate_recourse_plans(
        self,
        query_instance: pd.DataFrame,
        total_cfs: int = 2,
        custom_features_to_vary: Optional[List[str]] = None,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Generates actionable counterfactual paths to flip a rejection (DEFAULT=1)
        into an approval (DEFAULT=0), strictly preserving immutable demographic
        traits and past credit history.

        Parameters:
            query_instance (pd.DataFrame): 1-row DataFrame of the rejected applicant.
            total_cfs (int): Number of diverse counterfactual recommendations to generate.
            custom_features_to_vary (list, optional): Overrides mutable feature list if specified.
            random_seed (int): Seed for reproducible counterfactual generation.

        Returns:
            cf_df (pd.DataFrame): The full counterfactual applicant profiles.
            deltas_df (pd.DataFrame): Clean table highlighting ONLY the necessary changes.
            mean_burden (float): Average normalized L1 recourse effort required.
        """
        # Strictly enforce actionability: default to ACTIONABLE_FEATURES (PAY_1, BILL_AMT1, PAY_AMT1)
        # Lock immutable traits (SEX, AGE, MARRIAGE, EDUCATION), past history (PAY_2..6), and bank limits (LIMIT_BAL)
        if custom_features_to_vary is None:
            features_to_vary = [c for c in ACTIONABLE_FEATURES if c in query_instance.columns]
        else:
            prohibited = set(IMMUTABLE_FEATURES + HISTORICAL_FEATURES + INSTITUTIONAL_FEATURES)
            features_to_vary = [c for c in custom_features_to_vary if c not in prohibited]

        query_row = query_instance.iloc[0]
        current_bill = float(query_row.get("BILL_AMT1", 0.0))
        current_pay = float(query_row.get("PAY_AMT1", 0.0))

        # Directional and realistic boundary constraints:
        # 1. PAY_1: Must be on-time, fully paid, or no consumption (<= 0), cannot worsen delay.
        # 2. BILL_AMT1: Must pay down or maintain debt (0 to current bill), never increase debt.
        # 3. PAY_AMT1: Must increase or maintain payment amount.
        permitted_range = {}
        if "PAY_1" in features_to_vary:
            permitted_range["PAY_1"] = [-2, 0]
        if "BILL_AMT1" in features_to_vary:
            permitted_range["BILL_AMT1"] = [0.0, max(current_bill, 0.0)]
        if "PAY_AMT1" in features_to_vary:
            permitted_range["PAY_AMT1"] = [max(current_pay, 0.0), max(current_pay * 3, 100000.0)]

        # Generate diverse counterfactuals targeting DEFAULT=0 (Approved)
        try:
            dice_exp = self.exp.generate_counterfactuals(
                query_instance,
                total_CFs=total_cfs,
                desired_class=0,
                features_to_vary=features_to_vary,
                permitted_range=permitted_range if permitted_range else None,
                random_seed=random_seed,
                verbose=False
            )
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
        except Exception as e:
            # When no counterfactual exists within the strict constraints, report gracefully
            return pd.DataFrame(), pd.DataFrame(), 0.0

        if cf_df is None or len(cf_df) == 0:
            return pd.DataFrame(), pd.DataFrame(), 0.0

        # Calculate deltas (what specifically needs to change)
        delta_records = []
        for idx, cf_row in cf_df.iterrows():
            changes = {}
            total_l1_burden = 0.0

            for col in features_to_vary:
                orig_val = query_row[col]
                cf_val = cf_row[col]

                # Significant change check
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
    engine = CreditRecourseEngine()
    df = pd.read_csv(DEFAULT_DATA_PATH)

    # Find a sample applicant
    sample = df[df["DEFAULT"] == 1].drop(columns=["DEFAULT"]).iloc[[0]]
    print("\nRejected Applicant Profile (Original):")
    print(sample[["LIMIT_BAL", "SEX", "AGE", "PAY_1", "BILL_AMT1", "PAY_AMT1"]])

    cfs, deltas, burden = engine.generate_recourse_plans(sample, total_cfs=2)
    print("\n--- Actionable Recourse Recommendations (Deltas) ---")
    print(deltas)
    print(f"\nMean Algorithmic Burden (Normalized L1 Distance): {burden:.2f}")
