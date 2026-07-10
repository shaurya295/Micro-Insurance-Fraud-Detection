"""
utils.py
--------
Small shared helpers used across the pipeline (preprocessing, training,
explainability, app).
"""

import json
import os


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_currency(value: float) -> str:
    return f"₹{value:,.2f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning `default` on zero/None denominators instead of raising."""
    if not denominator:
        return default
    return numerator / denominator
