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
