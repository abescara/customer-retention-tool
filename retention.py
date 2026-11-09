"""Reproduce the notebook's purchase cleaning and historical customer cohorts."""

from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "days_since_last_order",
    "order_count_180d",
    "spend_180d",
    "average_order_value",
]
CUTOFFS = pd.to_datetime([
    "2010-06-01", "2010-09-01", "2010-12-01",
    "2011-03-01", "2011-06-01", "2011-09-01",
])


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Read both workbook sheets into one transaction table."""
    workbook = pd.ExcelFile(path)
    transactions = pd.concat(
        [pd.read_excel(workbook, sheet_name=name) for name in workbook.sheet_names],
        ignore_index=True,
    )
    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"], errors="coerce"
    )
    return transactions


def clean_purchases(transactions: pd.DataFrame) -> pd.DataFrame:
    """Keep identifiable, positive, uncancelled purchase lines."""
    purchases = transactions.loc[
        transactions["Customer ID"].notna()
        & transactions["InvoiceDate"].notna()
        & ~transactions["Invoice"].astype(str).str.upper().str.startswith("C")
        & (transactions["Quantity"] > 0)
        & (transactions["Price"] > 0)
    ].copy()
    purchases["line_revenue"] = purchases["Quantity"] * purchases["Price"]
    return purchases


def make_orders(purchases: pd.DataFrame) -> pd.DataFrame:
    """Sum product lines into one row per customer invoice."""
    return purchases.groupby(["Customer ID", "Invoice"], as_index=False).agg(
        order_date=("InvoiceDate", "min"),
        order_value=("line_revenue", "sum"),
    )


def load_orders(path: str | Path) -> pd.DataFrame:
    """Load and clean the workbook, then create invoice-level orders."""
    return make_orders(clean_purchases(load_transactions(path)))


def build_cohort(orders: pd.DataFrame, cutoff: str | pd.Timestamp) -> pd.DataFrame:
    """Use the previous 180 days for features and next 90 for evaluation."""
    cutoff = pd.Timestamp(cutoff)
    history = orders.loc[
        orders["order_date"].ge(cutoff - pd.Timedelta(days=180))
        & orders["order_date"].lt(cutoff)
    ]
    future = orders.loc[
        orders["order_date"].ge(cutoff)
        & orders["order_date"].lt(cutoff + pd.Timedelta(days=90))
    ]
    cohort = history.groupby("Customer ID").agg(
        last_order_date=("order_date", "max"),
        order_count_180d=("Invoice", "nunique"),
        spend_180d=("order_value", "sum"),
        average_order_value=("order_value", "mean"),
    )
    cohort["days_since_last_order"] = (cutoff - cohort["last_order_date"]).dt.days
    cohort["returned_90d"] = cohort.index.isin(future["Customer ID"]).astype(int)
    cohort["did_not_return_90d"] = 1 - cohort["returned_90d"]
    return cohort.drop(columns="last_order_date").reset_index().assign(cutoff=cutoff)
