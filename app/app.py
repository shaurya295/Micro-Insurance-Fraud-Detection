"""
app.py
------
Streamlit dashboard for the Micro Insurance Claim Risk and Anomaly
Detection System.

A human claims adjuster can:
  1. Look up any policy number.
  2. See the claim's AI-generated fraud risk score and risk tier.
  3. Read the plain-English reason codes explaining WHY it was flagged.
  4. Record a decision — Approved / Denied / Escalated for Manual Review
     — which is appended to a review log (data/review_log.csv) to close
     the human-in-the-loop workflow.
  5. Batch-score a CSV of claims.
  6. View portfolio-level analytics with embedded evaluation charts.

Run with:  streamlit run app/app.py   (from the project root)
"""

import io
import json
import os
import sys
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import Preprocessor, load_raw_data, transform_single_record  # noqa: E402
from explainability import generate_reason_codes, risk_tier  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths / cached resources
# --------------------------------------------------------------------------- #

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(ROOT, "data", "sample_or_raw_data.csv")
MODEL_PATH = os.path.join(ROOT, "models", "fraud_model.pkl")
PREPROCESSOR_PATH = os.path.join(ROOT, "models", "preprocessor.pkl")
SCORED_TEST_PATH = os.path.join(ROOT, "models", "scored_test_set.csv")
REVIEW_LOG_PATH = os.path.join(ROOT, "data", "review_log.csv")
METADATA_PATH = os.path.join(ROOT, "models", "model_metadata.json")
SCREENSHOTS_DIR = os.path.join(ROOT, "docs", "screenshots")

st.set_page_config(
    page_title="Claim Risk & Anomaly Detection", page_icon="🛡️", layout="wide"
)

# --------------------------------------------------------------------------- #
# Custom CSS for premium look
# --------------------------------------------------------------------------- #

st.markdown("""
<style>
    /* Overall font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Risk card styling */
    .risk-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .risk-card h3 {
        margin: 0 0 0.3rem 0;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .risk-card .value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .card-low {
        background: linear-gradient(135deg, #0d3b2e, #1a5c46);
        border: 1px solid #2a7a5e;
        color: #a8f0d4;
    }
    .card-low .value { color: #4ade80; }
    .card-medium {
        background: linear-gradient(135deg, #3d2e0a, #5c4a1a);
        border: 1px solid #7a6a2a;
        color: #f0d8a8;
    }
    .card-medium .value { color: #fbbf24; }
    .card-high {
        background: linear-gradient(135deg, #3b0d0d, #5c1a1a);
        border: 1px solid #7a2a2a;
        color: #f0a8a8;
    }
    .card-high .value { color: #f87171; }
    .card-neutral {
        background: linear-gradient(135deg, #1a1d23, #2a2d33);
        border: 1px solid #3a3d43;
        color: #b0b5c0;
    }
    .card-neutral .value { color: #e2e8f0; }

    /* Reason code cards */
    .reason-card {
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        border-left: 4px solid;
    }
    .reason-high {
        background-color: rgba(248, 113, 113, 0.08);
        border-left-color: #f87171;
    }
    .reason-medium {
        background-color: rgba(251, 191, 36, 0.08);
        border-left-color: #fbbf24;
    }
    .reason-low {
        background-color: rgba(250, 204, 21, 0.06);
        border-left-color: #facc15;
    }

    /* Progress bar override */
    .stProgress > div > div > div > div {
        border-radius: 8px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #2f5fd8;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = Preprocessor.load(PREPROCESSOR_PATH)
    return model, preprocessor


@st.cache_data
def load_claims():
    return load_raw_data(DATA_PATH)


@st.cache_data
def load_background_means():
    scored = pd.read_csv(SCORED_TEST_PATH)
    return scored


@st.cache_data
def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            return json.load(f)
    return {}


def score_claim(raw_record: dict, model, preprocessor):
    X_row = transform_single_record(raw_record, preprocessor)
    proba = float(model.predict_proba(X_row)[0, 1])
    return proba, X_row


def log_decision(policy_number, proba, tier, decision, reviewer_note):
    """Append a new decision. Guarded by the caller (existing active decisions
    must be cleared via `clear_decision` first) so a claim is never
    re-decided silently."""
    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "policy_number": policy_number,
                "fraud_probability": round(proba, 4),
                "risk_tier": tier,
                "decision": decision,
                "reviewer_note": reviewer_note,
                "status": "active",
            }
        ]
    )
    header = not os.path.exists(REVIEW_LOG_PATH)
    row.to_csv(REVIEW_LOG_PATH, mode="a", header=header, index=False)


def get_existing_decision(policy_number):
    """Return the current active decision for a policy, or None.

    A claim should have at most one active decision at a time — this is
    what stops the adjuster from silently re-approving/denying the same
    claim over and over.
    """
    if not os.path.exists(REVIEW_LOG_PATH):
        return None
    log = pd.read_csv(REVIEW_LOG_PATH)
    if "status" not in log.columns:
        log["status"] = "active"  # backward-compat with older log files
    active = log[(log["policy_number"] == policy_number) & (log["status"] == "active")]
    if active.empty:
        return None
    return active.sort_values("timestamp").iloc[-1].to_dict()


def clear_decision(policy_number):
    """Mark this policy's active decision as superseded rather than
    deleting it, preserving a full audit trail while allowing a
    deliberate override (e.g. by a supervisor)."""
    if not os.path.exists(REVIEW_LOG_PATH):
        return
    log = pd.read_csv(REVIEW_LOG_PATH)
    if "status" not in log.columns:
        log["status"] = "active"
    mask = (log["policy_number"] == policy_number) & (log["status"] == "active")
    log.loc[mask, "status"] = "superseded"
    log.to_csv(REVIEW_LOG_PATH, index=False)


# --------------------------------------------------------------------------- #
# Helper: render risk card HTML
# --------------------------------------------------------------------------- #

def risk_card(label, value_text, card_class="card-neutral"):
    st.markdown(
        f'<div class="risk-card {card_class}">'
        f'<h3>{label}</h3>'
        f'<p class="value">{value_text}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# App layout
# --------------------------------------------------------------------------- #

def main():
    st.title("🛡️ Micro Insurance Claim Risk & Anomaly Detection")
    st.caption(
        "AI-assisted triage for claims adjusters — every score comes with plain-English "
        "reason codes. This tool supports human review; it does not make final decisions."
    )

    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)):
        st.error(
            "No trained model found. Run `python src/train.py` from the project root first."
        )
        st.stop()

    model, preprocessor = load_artifacts()
    claims = load_claims()
    scored_test = load_background_means()
    background_means = scored_test[preprocessor.feature_columns].mean()
    metadata = load_metadata()

    tab_lookup, tab_dashboard, tab_batch = st.tabs(
        ["🔍 Claim Lookup", "📊 Portfolio Dashboard", "📁 Batch Scoring"]
    )

    # ------------------------------------------------------------------- #
    # Tab 1: single claim lookup + review workflow
    # ------------------------------------------------------------------- #
    with tab_lookup:
        col_search, col_spacer = st.columns([1, 2])
        with col_search:
            policy_number = st.selectbox(
                "Enter or select a policy number",
                options=sorted(claims["policy_number"].unique().tolist()),
            )

        record = claims[claims["policy_number"] == policy_number].iloc[0]
        raw_record = record.to_dict()

        proba, X_row = score_claim(raw_record, model, preprocessor)
        tier = risk_tier(proba)

        # --- Color-coded risk cards ---
        st.divider()
        tier_card_class = {
            "High Risk": "card-high",
            "Medium Risk": "card-medium",
            "Low Risk": "card-low",
        }

        col1, col2, col3 = st.columns(3)
        with col1:
            risk_card("Fraud Risk Score", f"{proba * 100:.1f}%", tier_card_class.get(tier, "card-neutral"))
            # Progress bar gauge
            st.progress(min(proba, 1.0))
        with col2:
            tier_icon = {"High Risk": "🔴", "Medium Risk": "🟠", "Low Risk": "🟢"}
            risk_card("Risk Tier", f"{tier_icon.get(tier, '')} {tier}", tier_card_class.get(tier, "card-neutral"))
        with col3:
            risk_card("Total Claim Amount", f"₹{record['total_claim_amount']:,.0f}", "card-neutral")

        left, right = st.columns([1, 1])

        with left:
            st.markdown('<p class="section-header">📋 Claim & Policy Details</p>', unsafe_allow_html=True)
            details = {
                "Policy state": str(record["policy_state"]),
                "Customer tenure (months)": str(record["months_as_customer"]),
                "Age": str(record["age"]),
                "Incident type": str(record["incident_type"]),
                "Incident severity": str(record["incident_severity"]),
                "Collision type": str(record["collision_type"]),
                "Authorities contacted": str(record["authorities_contacted"]),
                "Police report available": str(record["police_report_available"]),
                "Witnesses": str(record["witnesses"]),
                "Vehicles involved": str(record["number_of_vehicles_involved"]),
                "Annual premium": f"₹{record['policy_annual_premium']:,.2f}",
                "Injury claim": f"₹{record['injury_claim']:,.0f}",
                "Property claim": f"₹{record['property_claim']:,.0f}",
                "Vehicle claim": f"₹{record['vehicle_claim']:,.0f}",
            }
            st.table(pd.DataFrame(details.items(), columns=["Field", "Value"]))

        with right:
            st.markdown('<p class="section-header">⚠️ Why was this claim flagged?</p>', unsafe_allow_html=True)
            reasons = generate_reason_codes(
                raw_record, model, X_row, preprocessor.feature_columns, background_means
            )
            if not reasons:
                st.info("No significant risk indicators detected for this claim.")
            else:
                severity_icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}
                for r in reasons:
                    severity_class = f"reason-{r.severity}"
                    st.markdown(
                        f'<div class="reason-card {severity_class}">'
                        f'{severity_icon.get(r.severity, "⚪")} &nbsp; <strong>{r.message}</strong>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.caption(
                "Reason codes combine model feature-importance signals with insurance "
                "fraud business rules. They are decision support, not proof of fraud."
            )

        st.divider()
        st.markdown('<p class="section-header">📝 Adjuster Decision</p>', unsafe_allow_html=True)

        existing_decision = get_existing_decision(policy_number)

        if existing_decision is not None:
            st.info(
                f"**Already reviewed — {existing_decision['decision']}** "
                f"on {existing_decision['timestamp']}"
                + (f'  \n_Note: "{existing_decision["reviewer_note"]}"_' if existing_decision.get("reviewer_note") else "")
            )
            st.caption(
                "This claim already has a decision on file. Reopening should be a "
                "deliberate action (e.g. a supervisor override), not a re-click."
            )
            if st.button("Change decision"):
                clear_decision(policy_number)
                st.rerun()
        else:
            decision_col, note_col = st.columns([1, 2])
            with decision_col:
                decision = st.radio(
                    "Record a decision for this claim",
                    ["Approved", "Denied", "Escalated for Manual Review"],
                    horizontal=False,
                )
            with note_col:
                note = st.text_area("Reviewer note (optional)", height=100)

            if st.button("Submit Decision", type="primary"):
                log_decision(policy_number, proba, tier, decision, note)
                st.rerun()

        if os.path.exists(REVIEW_LOG_PATH):
            with st.expander("View review log"):
                log_df = pd.read_csv(REVIEW_LOG_PATH)
                st.dataframe(log_df, width="stretch")
                # Download button
                csv_data = log_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Review Log (CSV)",
                    data=csv_data,
                    file_name="review_log.csv",
                    mime="text/csv",
                )

    # ------------------------------------------------------------------- #
    # Tab 2: portfolio-level view with embedded charts
    # ------------------------------------------------------------------- #
    with tab_dashboard:
        st.markdown('<p class="section-header">📈 Held-Out Test Set — Risk Distribution</p>', unsafe_allow_html=True)
        st.caption(
            "Scores from the model's evaluation set, shown here to illustrate overall "
            "portfolio risk skew. In production this would run over the live claim book."
        )

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            risk_card("Claims Scored", str(len(scored_test)), "card-neutral")
        with col_b:
            risk_card("Flagged High Risk", str(int((scored_test["fraud_probability"] >= 0.66).sum())), "card-high")
        with col_c:
            risk_card("Actual Fraud Rate", f"{scored_test['actual_fraud'].mean() * 100:.1f}%", "card-medium")
        with col_d:
            # CV stability metric
            cv_mean = metadata.get("cv_f1_mean")
            cv_std = metadata.get("cv_f1_std")
            if cv_mean is not None:
                risk_card("CV F1 (3-fold)", f"{cv_mean:.3f} ± {cv_std:.3f}", "card-neutral")
            else:
                risk_card("CV F1", "N/A", "card-neutral")

        st.bar_chart(
            scored_test["fraud_probability"]
            .apply(risk_tier)
            .value_counts()
        )

        # --- Embedded evaluation charts ---
        st.divider()
        st.markdown('<p class="section-header">🔬 Model Evaluation Charts</p>', unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Confusion matrix from metadata
            cm_data = metadata.get("confusion_matrix")
            if cm_data:
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    np.array(cm_data), annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not Fraud", "Fraud"],
                    yticklabels=["Not Fraud", "Fraud"],
                    ax=ax, linewidths=0.5, linecolor="white",
                    annot_kws={"size": 16, "weight": "bold"},
                )
                ax.set_xlabel("Predicted", fontsize=11, fontweight="bold")
                ax.set_ylabel("Actual", fontsize=11, fontweight="bold")
                ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Confusion matrix data not available. Re-run training.")

        with chart_col2:
            # ROC curve from screenshot if available, otherwise from data
            roc_path = os.path.join(SCREENSHOTS_DIR, "roc_curve.png")
            if os.path.exists(roc_path):
                st.image(roc_path, caption="ROC Curve — Fraud Detection")
            else:
                st.info("ROC curve plot not found. Re-run `python src/train.py` to generate.")

        st.divider()
        st.markdown('<p class="section-header">🏆 Top Global Risk Drivers</p>', unsafe_allow_html=True)
        if metadata.get("feature_importances"):
            importances = pd.Series(metadata["feature_importances"]).sort_values(
                ascending=False
            ).head(10)
            # Render with matplotlib for a nicer look
            fig, ax = plt.subplots(figsize=(8, 4))
            names = [n.replace("_", " ").title() for n in importances.index[::-1]]
            values = importances.values[::-1]
            bars = ax.barh(names, values, color=sns.color_palette("viridis", len(names)))
            ax.set_xlabel("Importance", fontsize=11, fontweight="bold")
            ax.set_title("Top 10 Feature Importances", fontsize=13, fontweight="bold")
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                        f"{width:.3f}", va="center", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Model info summary
        metrics = metadata.get("metrics", {})
        if metrics:
            st.divider()
            st.markdown('<p class="section-header">📊 Model Performance Summary</p>', unsafe_allow_html=True)
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                risk_card("Precision", f"{metrics.get('precision', 0):.2f}", "card-neutral")
            with mcol2:
                risk_card("Recall", f"{metrics.get('recall', 0):.2f}", "card-neutral")
            with mcol3:
                risk_card("F1-Score", f"{metrics.get('f1_score', 0):.2f}", "card-neutral")
            with mcol4:
                risk_card("ROC-AUC", f"{metrics.get('roc_auc', 0):.2f}", "card-neutral")

        st.caption(
            "⚠️ Limitations: model trained on a small historical sample; scores should "
            "inform, not replace, professional judgment. Always follow your organization's "
            "responsible-use and fair-claims-handling policies."
        )

    # ------------------------------------------------------------------- #
    # Tab 3: Batch scoring
    # ------------------------------------------------------------------- #
    with tab_batch:
        st.markdown('<p class="section-header">📁 Batch Claim Scoring</p>', unsafe_allow_html=True)
        st.caption(
            "Upload a CSV file with claim records (same columns as the training data) "
            "to score all claims at once. Download the results with risk scores and tiers."
        )

        uploaded = st.file_uploader("Upload claims CSV", type=["csv"])

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                st.success(f"Loaded {len(batch_df)} claims from the uploaded file.")

                with st.spinner("Scoring all claims..."):
                    results = []
                    for _, row in batch_df.iterrows():
                        raw = row.to_dict()
                        try:
                            proba_val, _ = score_claim(raw, model, preprocessor)
                            tier_val = risk_tier(proba_val)
                        except Exception:
                            proba_val = None
                            tier_val = "Error"
                        results.append({
                            "policy_number": raw.get("policy_number", "N/A"),
                            "fraud_probability": round(proba_val, 4) if proba_val is not None else None,
                            "risk_tier": tier_val,
                            "total_claim_amount": raw.get("total_claim_amount", ""),
                        })

                    results_df = pd.DataFrame(results)

                # Summary cards
                scored_count = results_df["fraud_probability"].notna().sum()
                high_count = (results_df["risk_tier"] == "High Risk").sum()
                medium_count = (results_df["risk_tier"] == "Medium Risk").sum()

                rcol1, rcol2, rcol3, rcol4 = st.columns(4)
                with rcol1:
                    risk_card("Total Claims", str(len(results_df)), "card-neutral")
                with rcol2:
                    risk_card("High Risk", str(high_count), "card-high")
                with rcol3:
                    risk_card("Medium Risk", str(medium_count), "card-medium")
                with rcol4:
                    risk_card("Low Risk", str(scored_count - high_count - medium_count), "card-low")

                st.dataframe(results_df, width="stretch")

                # Download scored results
                csv_out = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Scored Results (CSV)",
                    data=csv_out,
                    file_name="batch_scored_claims.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.info("Upload a CSV to get started. The file should have the same column structure as the training dataset.")


if __name__ == "__main__":
    main()
