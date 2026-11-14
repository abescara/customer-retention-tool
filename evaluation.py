"""Capacity-based comparison of customer risk rankings."""

import numpy as np
import pandas as pd


def select_priority(customers: pd.DataFrame, score: str, share: int) -> pd.DataFrame:
    """Select the highest scores with a stable customer-ID tie break."""
    count = int(np.ceil(len(customers) * share / 100))
    return customers.sort_values(
        [score, "Customer ID"], ascending=[False, True], kind="stable"
    ).head(count).copy()


def capacity_comparison(customers: pd.DataFrame, shares=(10, 20, 30)) -> pd.DataFrame:
    """Compare observed nonreturn among selected customers on the test cohort."""
    total_nonreturn = int(customers["did_not_return_90d"].sum())
    rows = []
    for share in shares:
        for method, score in [("Recency only", "days_since_last_order"), ("Trained model", "model_risk")]:
            selected = select_priority(customers, score, share)
            nonreturn = int(selected["did_not_return_90d"].sum())
            rows.append({
                "Review share": f"{share}%",
                "Method": method,
                "Customers reviewed": len(selected),
                "Nonreturners found": nonreturn,
                "Selected nonreturn share": nonreturn / len(selected),
                "Nonreturner capture": nonreturn / total_nonreturn,
            })
    return pd.DataFrame(rows)


def ranking_overlap(customers: pd.DataFrame, share: int) -> dict[str, int]:
    """Count review-list agreement and nonreturners unique to each method."""
    recency = select_priority(customers, "days_since_last_order", share).set_index("Customer ID")
    model = select_priority(customers, "model_risk", share).set_index("Customer ID")
    common = recency.index.intersection(model.index)
    recency_only = recency.loc[recency.index.difference(common)]
    model_only = model.loc[model.index.difference(common)]
    return {
        "shared_customers": len(common),
        "recency_only_customers": len(recency_only),
        "model_only_customers": len(model_only),
        "recency_only_nonreturners": int(recency_only["did_not_return_90d"].sum()),
        "model_only_nonreturners": int(model_only["did_not_return_90d"].sum()),
    }


def calibration_summary(customers: pd.DataFrame) -> pd.DataFrame:
    """Compare predicted nonreturn risk with observed rates in 20-point bands."""
    risk = customers["model_risk"]
    if risk.isna().any() or not risk.between(0, 1).all():
        raise ValueError("Model risk must contain probabilities between 0 and 1.")
    bands = np.minimum((risk * 5).astype(int), 4)
    labelled = customers.assign(risk_band=bands.map(
        {i: f"{i * 20}–{(i + 1) * 20}%" for i in range(5)}
    ))
    return labelled.groupby("risk_band", sort=True, observed=True).agg(
        customers=("Customer ID", "size"),
        average_predicted_risk=("model_risk", "mean"),
        observed_nonreturn_share=("did_not_return_90d", "mean"),
    ).reset_index()
