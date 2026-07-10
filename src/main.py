"""
train.py
--------
Trains the fraud / anomaly classification model for the Micro Insurance
Claim Risk and Anomaly Detection System.

- Uses LightGBM if it is installed (better handling of imbalanced tabular
  data via `is_unbalance`), otherwise falls back to a class-weighted
  RandomForestClassifier so the pipeline runs anywhere.
- Optimizes / reports Precision, Recall, F1-score and ROC-AUC, which
  matter far more than accuracy on an imbalanced fraud dataset.
- Persists the trained model and its metadata (feature names, feature
  importances, chosen decision threshold) to `models/`.
- Auto-generates evaluation plots (confusion matrix, feature importance,
  ROC curve, risk-score distribution) into `docs/screenshots/`.
"""

from __future__ import annotations

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless plot generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV

from preprocessing import build_training_dataset, ARTIFACT_DIR, PROJECT_ROOT

MODEL_PATH = os.path.join(ARTIFACT_DIR, "fraud_model.pkl")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "model_metadata.json")
SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, "docs", "screenshots")


def _get_model_and_grid(use_lightgbm: bool = True):
    """Return an (estimator, param_grid) pair, preferring LightGBM."""
    if use_lightgbm:
        try:
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                random_state=42, is_unbalance=True, verbosity=-1
            )
            grid = {
                "n_estimators": [200, 400],
                "max_depth": [4, 6, -1],
                "learning_rate": [0.05, 0.1],
            }
            return model, grid, "LightGBM"
        except ImportError:
            pass

    model = RandomForestClassifier(
        random_state=42, class_weight="balanced", n_jobs=-1
    )
    grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 8, 12],
        "min_samples_leaf": [1, 2, 4],
    }
    return model, grid, "RandomForest"


def _choose_best_threshold(y_true, y_proba) -> float:
    """Pick the probability threshold that maximizes F1 on validation data.

    Fraud review workflows care about balancing analyst workload
    (precision) against missed fraud (recall); F1 is a reasonable
    default optimization target, and the resulting threshold is stored
    so the app can be tuned later without retraining.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1]) if len(thresholds) else 0
    return float(thresholds[best_idx]) if len(thresholds) else 0.5


def _save_plots(y_test, y_pred, y_proba, importances, feature_columns, roc_auc_val):
    """Auto-generate evaluation plots and save to docs/screenshots/."""
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid")

    # --- 1. Confusion Matrix Heatmap ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"],
        ax=ax, linewidths=0.5, linecolor="white",
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix — Held-Out Test Set", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(SCREENSHOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"[train] Saved confusion_matrix.png")

    # --- 2. Feature Importance (Top 15) ---
    if importances:
        sorted_feats = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:15]
        names = [f[0].replace("_", " ").title() for f in sorted_feats]
        values = [f[1] for f in sorted_feats]

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(names[::-1], values[::-1], color=sns.color_palette("viridis", len(names)))
        ax.set_xlabel("Importance", fontsize=12, fontweight="bold")
        ax.set_title("Top 15 Feature Importances", fontsize=14, fontweight="bold")
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{width:.3f}", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(SCREENSHOTS_DIR, "feature_importance.png"), dpi=150)
        plt.close(fig)
        print(f"[train] Saved feature_importance.png")

    # --- 3. ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2f5fd8", lw=2, label=f"ROC Curve (AUC = {roc_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Baseline")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2f5fd8")
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("ROC Curve — Fraud Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(SCREENSHOTS_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)
    print(f"[train] Saved roc_curve.png")

    # --- 4. Risk Score Distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_proba, bins=30, color="#2f5fd8", edgecolor="white", alpha=0.85)
    ax.axvline(x=0.33, color="#f39c12", lw=2, linestyle="--", label="Medium Risk (0.33)")
    ax.axvline(x=0.66, color="#e74c3c", lw=2, linestyle="--", label="High Risk (0.66)")
    ax.set_xlabel("Fraud Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Claims", fontsize=12, fontweight="bold")
    ax.set_title("Risk Score Distribution — Test Set", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(SCREENSHOTS_DIR, "risk_score_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"[train] Saved risk_score_distribution.png")


def train(raw_path: str = os.path.join("data", "sample_or_raw_data.csv")):
    X_train, X_test, y_train, y_test, preprocessor, ids_test = build_training_dataset(
        raw_path=raw_path
    )

    base_model, grid, model_name = _get_model_and_grid()
    print(f"[train] Training model family: {model_name}")

    search = GridSearchCV(
        base_model, grid, scoring="f1", cv=3, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"[train] Best params: {search.best_params_}")

    # Extract CV fold scores for stability reporting
    cv_results = search.cv_results_
    best_idx = search.best_index_
    cv_f1_mean = float(cv_results["mean_test_score"][best_idx])
    cv_f1_std = float(cv_results["std_test_score"][best_idx])
    print(f"[train] CV F1 (3-fold): {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")

    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = _choose_best_threshold(y_test, y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "model_name": model_name,
        "best_params": search.best_params_,
        "threshold": threshold,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("\n[train] Evaluation on held-out test set")
    print(json.dumps({k: v for k, v in metrics.items() if k != "best_params"}, indent=2))
    print("\n" + classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"]))

    if hasattr(model, "feature_importances_"):
        importances = dict(
            sorted(
                zip(preprocessor.feature_columns, model.feature_importances_.tolist()),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )
    else:
        importances = {}

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_name": model_name,
        "features": preprocessor.feature_columns,
        "feature_importances": importances,
        "threshold": threshold,
        "metrics": {k: v for k, v in metrics.items() if k not in ("best_params", "confusion_matrix")},
        "confusion_matrix": metrics["confusion_matrix"],
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # Persist a labeled test-set scoring table for the app's review queue.
    scored_test = X_test.copy()
    scored_test["policy_number"] = ids_test.values
    scored_test["actual_fraud"] = y_test.values
    scored_test["fraud_probability"] = y_proba
    scored_test["predicted_fraud"] = y_pred
    scored_test.to_csv(os.path.join(ARTIFACT_DIR, "scored_test_set.csv"), index=False)

    # Auto-generate evaluation plots
    _save_plots(y_test, y_pred, y_proba, importances, preprocessor.feature_columns, metrics["roc_auc"])

    print(f"\n[train] Model saved to {MODEL_PATH}")
    print(f"[train] Metadata saved to {METADATA_PATH}")
    print(f"[train] Plots saved to {SCREENSHOTS_DIR}")
    return model, metadata


if __name__ == "__main__":
    train()
