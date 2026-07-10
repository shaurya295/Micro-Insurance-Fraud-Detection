"""
explainability.py
------------------
Explainable AI layer for the Micro Insurance Claim Risk and Anomaly
Detection System.

Produces human-readable "Reason Codes" for a flagged claim by combining:
  1. Model-driven signals - which features pushed this specific claim's
     prediction up, using SHAP if installed, otherwise a fast
     mean/feature-importance based approximation.
  2. Business-rule signals - known insurance-fraud red flags (e.g. a
     "Major Damage" incident where authorities were never contacted, a
     claim filed very soon after the policy started, a claim amount far
     out of proportion to the premium paid, missing documentation on a
     severe incident, etc.)

The combined, de-duplicated, ranked list of reasons is what a claims
adjuster sees in the app next to the risk score, so every flag must be
understandable to a non-technical reviewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class ReasonCode:
    code: str
    message: str
    severity: str  # "high", "medium", "low"
    source: str  # "model" or "rule"


# --------------------------------------------------------------------------- #
# Business rules
# --------------------------------------------------------------------------- #
# Each rule receives the *raw* claim record (original column names/values,
# before encoding/scaling) and returns a ReasonCode or None.

def _rule_major_damage_no_authorities(raw: dict) -> ReasonCode | None:
    if raw.get("incident_severity") == "Major Damage" and raw.get(
        "authorities_contacted"
    ) in (None, "None", "nan", np.nan):
        return ReasonCode(
            code="MAJOR_DAMAGE_NO_AUTHORITIES",
            message="Major damage was reported but no authorities were contacted — unusual for a serious incident.",
            severity="high",
            source="rule",
        )
    return None


def _rule_early_claim(raw: dict, threshold_days: int = 30) -> ReasonCode | None:
    months = raw.get("months_as_customer")
    if months is not None and months * 30 <= threshold_days:
        return ReasonCode(
            code="EARLY_CLAIM",
            message=f"Claim was filed only ~{int(months)} month(s) after the policy began — early claims carry higher fraud risk.",
            severity="high",
            source="rule",
        )
    return None


def _rule_high_claim_to_premium(raw: dict, ratio_threshold: float = 20.0) -> ReasonCode | None:
    premium = raw.get("policy_annual_premium") or 0
    claim = raw.get("total_claim_amount") or 0
    if premium > 0 and (claim / premium) >= ratio_threshold:
        return ReasonCode(
            code="HIGH_CLAIM_TO_PREMIUM_RATIO",
            message=f"Claim amount (₹{claim:,.0f}) is {claim / premium:.1f}x the annual premium — disproportionately large.",
            severity="medium",
            source="rule",
        )
    return None


def _rule_missing_documentation(raw: dict) -> ReasonCode | None:
    severity = raw.get("incident_severity")
    police_report = raw.get("police_report_available")
    if severity in ("Major Damage", "Total Loss") and police_report in (
        None, "?", "NO", "nan", np.nan
    ):
        return ReasonCode(
            code="MISSING_POLICE_REPORT",
            message="No police report is on file despite a severe incident — documentation gap worth verifying.",
            severity="medium",
            source="rule",
        )
    return None


def _rule_no_witnesses_multi_vehicle(raw: dict) -> ReasonCode | None:
    vehicles = raw.get("number_of_vehicles_involved") or 0
    witnesses = raw.get("witnesses")
    if vehicles >= 2 and witnesses == 0:
        return ReasonCode(
            code="NO_WITNESSES_MULTI_VEHICLE",
            message=f"{int(vehicles)} vehicles were involved but zero witnesses were recorded.",
            severity="low",
            source="rule",
        )
    return None


def _rule_odd_hour_incident(raw: dict) -> ReasonCode | None:
    hour = raw.get("incident_hour_of_the_day")
    witnesses = raw.get("witnesses")
    if hour is not None and (0 <= hour <= 5) and (witnesses in (0, None)):
        return ReasonCode(
            code="ODD_HOUR_NO_WITNESS",
            message=f"Incident occurred at {int(hour)}:00 with no witnesses — a common pattern in staged-accident cases.",
            severity="low",
            source="rule",
        )
    return None


def _rule_high_umbrella_limit(raw: dict) -> ReasonCode | None:
    umbrella = raw.get("umbrella_limit") or 0
    claim = raw.get("total_claim_amount") or 0
    if umbrella >= 5_000_000 and claim >= 50_000:
        return ReasonCode(
            code="HIGH_UMBRELLA_LARGE_CLAIM",
            message="A large claim combined with a high umbrella policy limit warrants a closer look.",
            severity="low",
            source="rule",
        )
    return None


BUSINESS_RULES = [
    _rule_major_damage_no_authorities,
    _rule_early_claim,
    _rule_high_claim_to_premium,
    _rule_missing_documentation,
    _rule_no_witnesses_multi_vehicle,
    _rule_odd_hour_incident,
    _rule_high_umbrella_limit,
]


def apply_business_rules(raw_record: dict) -> List[ReasonCode]:
    reasons = []
    for rule in BUSINESS_RULES:
        try:
            result = rule(raw_record)
        except Exception:
            result = None
        if result is not None:
            reasons.append(result)
    return reasons


# --------------------------------------------------------------------------- #
# Model-driven signals
# --------------------------------------------------------------------------- #

_SEVERITY_BY_RANK = ["high", "high", "medium", "medium", "low"]

# Human-friendly phrasing for the engineered / raw features most likely to
# appear in the top drivers, so adjusters don't see raw column names.
_FEATURE_LABELS = {
    "incident_severity": "the reported incident severity",
    "insured_hobbies": "the policyholder's declared hobbies",
    "insured_zip": "the policyholder's zip code risk profile",
    "vehicle_claim": "the vehicle-damage claim amount",
    "total_claim_amount": "the total claim amount",
    "claim_to_premium_ratio": "the claim-to-premium ratio",
    "property_claim": "the property-damage claim amount",
    "policy_annual_premium": "the policy's annual premium",
    "insured_occupation": "the policyholder's occupation",
    "tenure_at_incident_days": "how long the policy had been active before the incident",
    "injury_claim": "the injury claim amount",
    "months_as_customer": "the customer's tenure",
    "auto_year": "the vehicle's model year",
    "incident_hour_of_the_day": "the time of day of the incident",
    "capital-gains": "the policyholder's reported capital gains",
    "vehicle_age_at_incident": "the vehicle's age at the time of the incident",
    "number_of_vehicles_involved": "the number of vehicles involved",
    "witnesses": "the number of witnesses reported",
    "bodily_injuries": "the number of bodily injuries reported",
}


def model_driven_reasons(
    model,
    X_row: pd.DataFrame,
    feature_columns: List[str],
    background_means: pd.Series,
    top_n: int = 3,
) -> List[ReasonCode]:
    """Explain the single-row prediction using SHAP when available.

    Falls back to a lightweight approximation: for each feature, the
    (scaled) deviation from the population mean multiplied by the
    model's global feature importance, which approximates how much that
    feature pushes this specific claim's score away from "typical".
    """
    reasons: List[ReasonCode] = []

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row)
        # For binary classifiers, shap may return a list [class0, class1]
        values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        contributions = pd.Series(values, index=feature_columns)
    except Exception:
        if not hasattr(model, "feature_importances_"):
            return reasons
        importances = pd.Series(model.feature_importances_, index=feature_columns)
        deviation = (X_row.iloc[0] - background_means).fillna(0)
        contributions = importances * deviation

    top_features = contributions.sort_values(ascending=False).head(top_n)
    for rank, (feat, contribution) in enumerate(top_features.items()):
        if contribution <= 0:
            continue
        label = _FEATURE_LABELS.get(feat, feat.replace("_", " "))
        severity = _SEVERITY_BY_RANK[min(rank, len(_SEVERITY_BY_RANK) - 1)]
        reasons.append(
            ReasonCode(
                code=f"MODEL_DRIVER_{feat.upper()}",
                message=f"The model flagged {label} as an unusual value compared to typical claims.",
                severity=severity,
                source="model",
            )
        )
    return reasons


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def generate_reason_codes(
    raw_record: dict,
    model,
    X_row: pd.DataFrame,
    feature_columns: List[str],
    background_means: pd.Series,
    max_reasons: int = 6,
) -> List[ReasonCode]:
    """Combine business rules with model-driven signals into one ranked list."""
    rule_reasons = apply_business_rules(raw_record)
    model_reasons = model_driven_reasons(model, X_row, feature_columns, background_means)

    combined = rule_reasons + model_reasons
    combined.sort(key=lambda r: _SEVERITY_ORDER.get(r.severity, 3))

    seen_codes = set()
    deduped = []
    for r in combined:
        if r.code in seen_codes:
            continue
        seen_codes.add(r.code)
        deduped.append(r)

    return deduped[:max_reasons]


def risk_tier(probability: float) -> str:
    """Map a fraud probability to a human risk tier used across the app."""
    if probability >= 0.66:
        return "High Risk"
    if probability >= 0.33:
        return "Medium Risk"
    return "Low Risk"
