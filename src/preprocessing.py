"""
preprocessing.py
-----------------
Data preprocessing pipeline for the Micro Insurance Claim Risk and
Anomaly Detection System.

Responsibilities
1. Load the raw claims data (CSV or the original Excel export).
2. Replace '?' placeholders with proper missing values and impute them
   using statistically/business-appropriate strategies.
3. Parse date fields and engineer time-based risk features.
4. Encode categorical fields (persisted LabelEncoders for reuse at
   inference time).
5. Scale numeric features (persisted StandardScaler).
6. Balance the target classes for model training (SMOTE if the
   `imbalanced-learn` package is available, otherwise a safe fallback
   that still returns balanced training data).

The module exposes a single high level entry point,
`build_training_dataset()`, used by `train.py`, as well as
`transform_single_record()` used by the Streamlit app / explainability
layer to preprocess one claim the same way at inference time.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "sample_or_raw_data.csv")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "models")
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")

TARGET_COL = "fraud_reported"
ID_COL = "policy_number"

# Columns dropped outright: free-text / near-unique identifiers that do not
# generalize (kept in a separate lookup table for the UI instead).
DROP_COLS = ["incident_location"]

# Date columns used to engineer features, then discarded in their raw form.
DATE_COLS = ["policy_bind_date", "incident_date"]

# Columns known to contain the '?' missing-value placeholder in this dataset.
PLACEHOLDER_COLS = ["collision_type", "property_damage", "police_report_available"]

CATEGORICAL_COLS = [
    "policy_state",
    "policy_csl",
    "insured_sex",
    "insured_education_level",
    "insured_occupation",
    "insured_hobbies",
    "insured_relationship",
    "incident_type",
    "collision_type",
    "incident_severity",
    "authorities_contacted",
    "incident_state",
    "incident_city",
    "property_damage",
    "bodily_injuries_band",  # engineered
    "police_report_available",
    "auto_make",
    "auto_model",
]

NUMERIC_COLS = [
    "months_as_customer",
    "age",
    "policy_deductable",
    "policy_annual_premium",
    "umbrella_limit",
    "capital-gains",
    "capital-loss",
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "total_claim_amount",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
    "auto_year",
    # engineered numeric features
    "tenure_at_incident_days",
    "vehicle_age_at_incident",
    "claim_to_premium_ratio",
    "injury_claim_ratio",
    "property_claim_ratio",
    "vehicle_claim_ratio",
]


@dataclass
class Preprocessor:
    """Holds every fitted artifact needed to reproduce preprocessing."""

    label_encoders: dict = field(default_factory=dict)
    scaler: Optional[StandardScaler] = None
    feature_columns: list = field(default_factory=list)
    numeric_impute_values: dict = field(default_factory=dict)
    categorical_impute_values: dict = field(default_factory=dict)

    def save(self, path: str = PREPROCESSOR_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = PREPROCESSOR_PATH) -> "Preprocessor":
        return joblib.load(path)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw claims dataset from CSV or Excel."""
    if path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# Cleaning & feature engineering
# --------------------------------------------------------------------------- #

def _replace_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '?' placeholders with NaN so they can be imputed properly."""
    df = df.copy()
    df.replace("?", np.nan, inplace=True)
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create business-relevant derived features."""
    df = df.copy()

    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Customer tenure (days) between policy start and the incident. A very
    # short tenure before a claim is a classic fraud red flag.
    df["tenure_at_incident_days"] = (
        df["incident_date"] - df["policy_bind_date"]
    ).dt.days
    df["tenure_at_incident_days"] = df["tenure_at_incident_days"].clip(lower=0)

    # Vehicle age at the time of the incident.
    df["vehicle_age_at_incident"] = (
        df["incident_date"].dt.year - df["auto_year"]
    ).clip(lower=0)

    # Ratios that highlight abnormal claim structure.
    df["claim_to_premium_ratio"] = df["total_claim_amount"] / df[
        "policy_annual_premium"
    ].replace(0, np.nan)
    df["injury_claim_ratio"] = df["injury_claim"] / df["total_claim_amount"].replace(0, np.nan)
    df["property_claim_ratio"] = df["property_claim"] / df["total_claim_amount"].replace(0, np.nan)
    df["vehicle_claim_ratio"] = df["vehicle_claim"] / df["total_claim_amount"].replace(0, np.nan)

    for col in ["claim_to_premium_ratio", "injury_claim_ratio", "property_claim_ratio", "vehicle_claim_ratio"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Bucketed bodily injuries -> lets the model/business rules treat
    # "0 injuries" vs "1+ injuries" as a categorical severity signal.
    df["bodily_injuries_band"] = pd.cut(
        df["bodily_injuries"], bins=[-1, 0, 1, 100], labels=["none", "one", "multiple"]
    ).astype(str)

    df.drop(columns=DATE_COLS + DROP_COLS, inplace=True, errors="ignore")
    return df


def _impute_missing(df: pd.DataFrame, preprocessor: Preprocessor, fit: bool) -> pd.DataFrame:
    """Impute missing values.

    Categorical placeholders (collision_type, property_damage,
    police_report_available) are imputed with the column mode ("most
    likely" category) computed on the training data. Numeric columns are
    imputed with the median, which is robust to the outliers typical of
    claim-amount data.
    """
    df = df.copy()

    for col in PLACEHOLDER_COLS:
        if col not in df.columns:
            continue
        if fit:
            mode_val = df[col].mode(dropna=True)
            mode_val = mode_val.iloc[0] if not mode_val.empty else "UNKNOWN"
            preprocessor.categorical_impute_values[col] = mode_val
        fill_val = preprocessor.categorical_impute_values.get(col, "UNKNOWN")
        df[col] = df[col].fillna(fill_val)

    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    for col in numeric_present:
        if fit:
            preprocessor.numeric_impute_values[col] = df[col].median()
        fill_val = preprocessor.numeric_impute_values.get(col, 0)
        df[col] = df[col].fillna(fill_val)

    return df


def _encode_categoricals(df: pd.DataFrame, preprocessor: Preprocessor, fit: bool) -> pd.DataFrame:
    """Label-encode categorical columns, persisting encoders for reuse."""
    df = df.copy()
    cat_present = [c for c in CATEGORICAL_COLS if c in df.columns]

    for col in cat_present:
        df[col] = df[col].astype(str)
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            preprocessor.label_encoders[col] = le
        else:
            le = preprocessor.label_encoders[col]
            # Gracefully handle unseen categories at inference time.
            df[col] = df[col].apply(lambda v: v if v in le.classes_ else "__unseen__")
            if "__unseen__" not in le.classes_:
                le.classes_ = np.append(le.classes_, "__unseen__")
            df[col] = le.transform(df[col])
    return df


def _scale_numeric(df: pd.DataFrame, preprocessor: Preprocessor, fit: bool) -> pd.DataFrame:
    """Scale numeric columns with a persisted StandardScaler."""
    df = df.copy()
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]

    if fit:
        preprocessor.scaler = StandardScaler()
        df[numeric_present] = preprocessor.scaler.fit_transform(df[numeric_present])
    else:
        df[numeric_present] = preprocessor.scaler.transform(df[numeric_present])
    return df


def _balance_classes(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """Balance the training classes.

    Uses SMOTE (imbalanced-learn) when available. Falls back to random
    oversampling of the minority class so the pipeline still runs in
    environments where `imbalanced-learn` isn't installed.
    """
    try:
        from imblearn.over_sampling import SMOTE

        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X, y)
        print(f"[preprocessing] Balanced classes with SMOTE: {y.value_counts().to_dict()} -> "
              f"{pd.Series(y_res).value_counts().to_dict()}")
        return X_res, y_res
    except ImportError:
        warnings.warn(
            "imbalanced-learn not installed; falling back to random oversampling "
            "of the minority class. Install `imbalanced-learn` for SMOTE."
        )
        df = X.copy()
        df["__target__"] = y.values
        majority = df[df["__target__"] == df["__target__"].value_counts().idxmax()]
        minority = df[df["__target__"] != df["__target__"].value_counts().idxmax()]
        minority_upsampled = minority.sample(
            n=len(majority), replace=True, random_state=random_state
        )
        balanced = pd.concat([majority, minority_upsampled]).sample(
            frac=1, random_state=random_state
        )
        y_res = balanced.pop("__target__")
        print(f"[preprocessing] Balanced classes with random oversampling: "
              f"{y.value_counts().to_dict()} -> {y_res.value_counts().to_dict()}")
        return balanced, y_res


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def build_training_dataset(
    raw_path: str = RAW_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    balance_train: bool = True,
):
    """Full preprocessing pipeline used at training time.

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor, ids_test
    """
    df = load_raw_data(raw_path)
    df = _replace_placeholders(df)

    ids = df[ID_COL].copy()
    df = _engineer_features(df)

    y = df[TARGET_COL].map({"Y": 1, "N": 0}).astype(int)
    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = Preprocessor()

    X_train = _impute_missing(X_train, preprocessor, fit=True)
    X_train = _encode_categoricals(X_train, preprocessor, fit=True)
    X_train = _scale_numeric(X_train, preprocessor, fit=True)
    preprocessor.feature_columns = X_train.columns.tolist()

    X_test = _impute_missing(X_test, preprocessor, fit=False)
    X_test = _encode_categoricals(X_test, preprocessor, fit=False)
    X_test = _scale_numeric(X_test, preprocessor, fit=False)
    X_test = X_test[preprocessor.feature_columns]

    if balance_train:
        X_train, y_train = _balance_classes(X_train, y_train, random_state=random_state)

    preprocessor.save()

    return X_train, X_test, y_train, y_test, preprocessor, ids_test


def transform_single_record(raw_record: dict, preprocessor: Preprocessor) -> pd.DataFrame:
    """Apply the exact same preprocessing to a single new claim record.

    `raw_record` should contain the original raw column names (as in the
    source dataset), e.g. one row pulled from the claims lookup table.
    Used by the Streamlit app and the explainability layer at inference
    time.
    """
    df = pd.DataFrame([raw_record])
    df = _replace_placeholders(df)
    df = _engineer_features(df)
    df = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")

    df = _impute_missing(df, preprocessor, fit=False)
    df = _encode_categoricals(df, preprocessor, fit=False)
    df = _scale_numeric(df, preprocessor, fit=False)
    df = df[preprocessor.feature_columns]
    return df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor, ids_test = build_training_dataset()
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Saved preprocessor to", PREPROCESSOR_PATH)
