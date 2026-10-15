"""
=============================================================================
MODULE: data_loader.py
PROJECT: AIPI 590 - Project 1: ¿Por qué me rechazaron?
AUTHOR: Tonantzin Real Rojas
DESCRIPTION:
    Data ingestion, preprocessing, and feature metadata definitions for the
    UCI Default of Credit Card Clients dataset. Handles caching, column
    renaming, categorical normalization, and train/test splitting.
=============================================================================
"""

import os
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# FEATURE DEFINITIONS & DOMAIN METADATA
# ---------------------------------------------------------------------------
# Mapping OpenML generic column names (x1..x23, y) to standard UCI credit names
RAW_COLUMN_MAPPING = {
    "x1": "LIMIT_BAL",
    "x2": "SEX",
    "x3": "EDUCATION",
    "x4": "MARRIAGE",
    "x5": "AGE",
    "x6": "PAY_1",
    "x7": "PAY_2",
    "x8": "PAY_3",
    "x9": "PAY_4",
    "x10": "PAY_5",
    "x11": "PAY_6",
    "x12": "BILL_AMT1",
    "x13": "BILL_AMT2",
    "x14": "BILL_AMT3",
    "x15": "BILL_AMT4",
    "x16": "BILL_AMT5",
    "x17": "BILL_AMT6",
    "x18": "PAY_AMT1",
    "x19": "PAY_AMT2",
    "x20": "PAY_AMT3",
    "x21": "PAY_AMT4",
    "x22": "PAY_AMT5",
    "x23": "PAY_AMT6",
    "y": "DEFAULT"
}

# Base project directory for robust cross-environment file resolution
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "credit_card_clean.csv")

# Immutable demographic features: The model and recourse engine must NEVER demand changes to these
IMMUTABLE_FEATURES = ["SEX", "AGE", "EDUCATION", "MARRIAGE"]

# Historical payment records: Past events that cannot be retroactively altered in the present
HISTORICAL_FEATURES = [
    "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

# Institutional features: Dictated by the lender/card issuer, not unilaterally set by applicant
INSTITUTIONAL_FEATURES = ["LIMIT_BAL"]

# Directly actionable financial levers: Current behaviors the applicant can execute today
# 1. PAY_1: Settle current delinquency to achieve on-time repayment status (<= 0)
# 2. BILL_AMT1: Pay down outstanding card balance (debt reduction)
# 3. PAY_AMT1: Increase current monthly payment towards the bill
ACTIONABLE_FEATURES = ["PAY_1", "BILL_AMT1", "PAY_AMT1"]

# Human-readable labels for categorical codes (Spanish / English for bilingual UI)
CATEGORICAL_LABELS = {
    "SEX": {1: "Hombre / Male", 2: "Mujer / Female"},
    "EDUCATION": {
        1: "Posgrado / Graduate School",
        2: "Universidad / University",
        3: "Preparatoria / High School",
        4: "Otro / Other"
    },
    "MARRIAGE": {
        1: "Casado(a) / Married",
        2: "Soltero(a) / Single",
        3: "Otro / Other"
    }
}


def download_and_clean_data(
    cache_path: str = DEFAULT_DATA_PATH,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Downloads the credit default dataset from OpenML or loads from local cache.
    Performs data cleaning, normalization of categorical codes, and verifies
    data integrity.

    Parameters:
        cache_path (str): Local path to save/load the preprocessed CSV.
        force_download (bool): If True, re-downloads even if cached file exists.

    Returns:
        pd.DataFrame: Cleaned credit default dataset with 30,000 rows.
    """
    if os.path.exists(cache_path) and not force_download:
        print(f"[data_loader] Loading cached dataset from: {cache_path}")
        df = pd.read_csv(cache_path)
        return df

    print("[data_loader] Fetching dataset from OpenML (ID: 42477)...")
    raw_data = fetch_openml(data_id=42477, as_frame=True)
    df = raw_data.frame.copy()

    # Step 1: Rename columns to standard, interpretable names
    if "x1" in df.columns:
        df = df.rename(columns=RAW_COLUMN_MAPPING)
    elif "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "DEFAULT", "PAY_0": "PAY_1"})

    # Step 2: Ensure target is integer (0 = Non-default/Approved, 1 = Default/High-risk)
    df["DEFAULT"] = df["DEFAULT"].astype(int)

    # Step 3: Clean up categorical anomalies (per UCI documentation)
    # EDUCATION: 1=grad, 2=university, 3=high school, 4=others. Values 0, 5, 6 represent undocumented -> map to 4 (Others)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE: 1=married, 2=single, 3=others. Value 0 represents undocumented -> map to 3 (Others)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # Step 4: Ensure all numeric columns have appropriate integer/float types
    for col in df.columns:
        if col != "DEFAULT":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Step 5: Save cleaned dataset to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"[data_loader] Clean dataset cached successfully ({df.shape[0]} rows, {df.shape[1]} cols) -> {cache_path}")

    return df


def get_train_test_data(
    test_size: float = 0.2,
    random_state: int = 42,
    cache_path: str = DEFAULT_DATA_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the cleaned dataset into stratified train and test sets.

    Parameters:
        test_size (float): Proportion of dataset to include in test split (default 0.2).
        random_state (int): Reproducibility seed.
        cache_path (str): Path to clean dataset cache.

    Returns:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Testing feature matrix.
        y_train (pd.Series): Training target labels.
        y_test (pd.Series): Testing target labels.
    """
    df = download_and_clean_data(cache_path=cache_path)

    features = [c for c in df.columns if c != "DEFAULT"]
    X = df[features]
    y = df["DEFAULT"]

    # Stratified split ensures equal default rate (approx 22.1%) in both train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[data_loader] Train split: {X_train.shape[0]} rows | Test split: {X_test.shape[0]} rows")
    print(f"[data_loader] Baseline Default Rate: {y.mean():.2%}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test script standalone execution
    df = download_and_clean_data()
    X_train, X_test, y_train, y_test = get_train_test_data()
    print("[data_loader] Sanity check passed successfully!")
