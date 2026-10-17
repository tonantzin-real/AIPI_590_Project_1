"""
=============================================================================
MODULE: train_model.py
PROJECT: AIPI 590 - Project 1: Why Was I Rejected?
AUTHOR: Tonantzin Real Rojas
DESCRIPTION:
    Trains, evaluates, and serializes the champion credit scoring classifier
    using LightGBM. We prioritize ROC-AUC (ranking quality) and use balanced
    class weights so the ~22% minority default class is not ignored; tree
    ensembles also enable exact TreeSHAP computation. Note: predicted
    probabilities are not calibrated (class weighting shifts them upward).
=============================================================================
"""

import os
import json
import joblib
from typing import Dict, Any
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Relative import within src or absolute import
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from src.data_loader import (
    get_train_test_data,
    IMMUTABLE_FEATURES,
    ACTIONABLE_FEATURES,
)

DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models")


def train_credit_models(
    model_dir: str = DEFAULT_MODEL_DIR,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains the champion credit scoring model (LightGBM), evaluates
    performance on an out-of-sample test set, and saves the champion
    model and its training artifacts.

    Parameters:
        model_dir (str): Directory where trained models and metrics are saved.
        random_state (int): Seed for reproducible model training.

    Returns:
        dict: Evaluation metrics and metadata for the trained models.
    """
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load train/test data
    X_train, X_test, y_train, y_test = get_train_test_data(random_state=random_state)
    feature_names = list(X_train.columns)

    print("\n" + "=" * 60)
    print("TRAINING CREDIT RISK UNDERWRITING MODELS")
    print("=" * 60)

    # 2. Configure LightGBM (Champion model: fast, non-linear, TreeSHAP compatible)
    # class_weight='balanced' addresses the default imbalance (~22% default rate)
    lgb_model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=random_state,
        verbose=-1,
    )

    print("[train_model] Fitting LightGBM Classifier...")
    lgb_model.fit(X_train, y_train)

    # 3. Model Evaluation on Hold-out Test Set
    # Predict default probabilities: P(DEFAULT = 1)
    y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]

    # In fintech lending, a standard risk cutoff of 0.50 (or tailored threshold) is used
    threshold = 0.50
    y_pred = (y_pred_proba >= threshold).astype(int)

    auc_score = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Model Evaluation (Hold-out Test Set) ---")
    print(f"ROC-AUC Score:      {auc_score:.4f} (Strong discriminative power)")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Precision (Risk):   {prec:.4f}")
    print(f"Recall (Defaults):  {rec:.4f} (High recall catches risky applicants)")
    print(f"F1-Score:           {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # 4. Serialize Champion Model & Metadata
    model_path = os.path.join(model_dir, "credit_model.joblib")
    joblib.dump(lgb_model, model_path)
    print(f"\n[train_model] Model successfully serialized to: {model_path}")

    # Save metadata JSON for the Streamlit dashboard and audit modules
    metadata = {
        "model_name": "LightGBM Credit Underwriter",
        "threshold": threshold,
        "feature_names": feature_names,
        "immutable_features": IMMUTABLE_FEATURES,
        "actionable_features": ACTIONABLE_FEATURES,
        "metrics": {
            "roc_auc": round(float(auc_score), 4),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "test_sample_size": int(len(y_test)),
        },
    }

    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[train_model] Model metadata saved to: {metadata_path}")

    return metadata


if __name__ == "__main__":
    metrics = train_credit_models()
    print("\n[train_model] Training pipeline completed successfully!")
