"""Audit duplicate lines and unusually large orders in the source workbook."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from retention import CUTOFFS, FEATURE_COLUMNS, build_cohort, clean_purchases, load_transactions, make_orders


def later_auc(orders: pd.DataFrame) -> float:
    snapshots = pd.concat([build_cohort(orders, date) for date in CUTOFFS])
    train = snapshots[snapshots["cutoff"] < CUTOFFS[-1]]
    test = snapshots[snapshots["cutoff"] == CUTOFFS[-1]]
    model = make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(train[FEATURE_COLUMNS], train["did_not_return_90d"])
    return roc_auc_score(test["did_not_return_90d"], model.predict_proba(test[FEATURE_COLUMNS])[:, 1])


def main() -> None:
    path = Path(__file__).parent / "data/raw/online_retail_II.xlsx"
    transactions = load_transactions(path)
    purchases = clean_purchases(transactions)
    deduplicated = clean_purchases(transactions.drop_duplicates())
    orders = make_orders(purchases)
    deduplicated_orders = make_orders(deduplicated)

    print(f"Source rows: {len(transactions):,}")
    print(f"Exact duplicate source rows: {transactions.duplicated().sum():,}")
    print(f"Valid purchase rows: {len(purchases):,}")
    print(f"Exact duplicate valid purchase rows: {purchases.duplicated(subset=transactions.columns).sum():,}")
    print(f"Orders: {len(orders):,}")
    print(f"Purchase-line revenue, keeping duplicates: {purchases['line_revenue'].sum():,.2f}")
    print(f"Purchase-line revenue, removing exact duplicates: {deduplicated['line_revenue'].sum():,.2f}")
    print(f"Largest order: {orders['order_value'].max():,.2f}")
    print(f"99th percentile order value: {orders['order_value'].quantile(.99):,.2f}")
    print(f"Top 1% order revenue share: {orders.nlargest(max(1, round(len(orders) * .01)), 'order_value')['order_value'].sum() / orders['order_value'].sum():.2%}")

    for name, current in [("Original", orders), ("Exact rows removed", deduplicated_orders)]:
        cohort = build_cohort(current, "2011-09-01")
        print(f"{name} September cohort: {len(cohort):,} customers, {cohort['did_not_return_90d'].mean():.2%} did not return, model ROC AUC {later_auc(current):.3f}")

    # Trimming these orders is a sensitivity check, not a proposed cleaning rule.
    threshold = orders["order_value"].quantile(.99)
    trimmed = orders.loc[orders["order_value"] <= threshold]
    cohort = build_cohort(trimmed, "2011-09-01")
    print(f"Top 1% orders removed: {len(cohort):,} customers, {cohort['did_not_return_90d'].mean():.2%} did not return, model ROC AUC {later_auc(trimmed):.3f}")


if __name__ == "__main__":
    main()
