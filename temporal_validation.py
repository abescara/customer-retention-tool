"""Compare rankings on successive later customer snapshots."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation import select_priority
from modeling import fit_risk_model, nonreturn_risk
from retention import CUTOFFS, build_cohort, load_orders


def evaluate_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Train only where the entire 90-day label window predates the test cutoff."""
    rows = []
    for cutoff in sorted(snapshots["cutoff"].unique()):
        train = snapshots.loc[snapshots["cutoff"] + pd.Timedelta(days=90) <= cutoff]
        test = snapshots.loc[snapshots["cutoff"] == cutoff].copy()
        if train.empty or test.empty or train["did_not_return_90d"].nunique() < 2 or test["did_not_return_90d"].nunique() < 2:
            continue
        model = fit_risk_model(train)
        test["model_risk"] = nonreturn_risk(model, test)
        actual = test["did_not_return_90d"]
        for method, score in [("Recency only", "days_since_last_order"), ("Trained model", "model_risk")]:
            selected = select_priority(test, score, 20)
            rows.append({
                "test_cutoff": pd.Timestamp(cutoff).date().isoformat(),
                "method": method,
                "training_snapshots": train["cutoff"].nunique(),
                "test_customers": len(test),
                "nonreturn_share": actual.mean(),
                "roc_auc": roc_auc_score(actual, test[score]),
                "average_precision": average_precision_score(actual, test[score]),
                "top_20_nonreturn_share": selected["did_not_return_90d"].mean(),
            })
    return pd.DataFrame(rows)


def main() -> None:
    source = Path(__file__).parent / "data/raw/online_retail_II.xlsx"
    if not source.exists():
        raise SystemExit("Download online_retail_II.xlsx from UCI into data/raw/ first.")
    orders = load_orders(source)
    if orders["order_date"].max() < CUTOFFS[-1] + pd.Timedelta(days=90):
        raise SystemExit("The source does not cover the final 90-day outcome window.")
    snapshots = pd.concat([build_cohort(orders, date) for date in CUTOFFS], ignore_index=True)
    results = evaluate_snapshots(snapshots)
    if results.empty:
        raise SystemExit("No evaluable later cohorts were found.")
    print(results.to_string(index=False, formatters={
        "nonreturn_share": "{:.3f}".format,
        "roc_auc": "{:.3f}".format,
        "average_precision": "{:.3f}".format,
        "top_20_nonreturn_share": "{:.3f}".format,
    }))


if __name__ == "__main__":
    main()
