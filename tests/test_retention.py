"""Check that purchase and outcome rules respect the decision date."""

import unittest

import pandas as pd

from retention import build_cohort, clean_purchases, make_orders


class RetentionTests(unittest.TestCase):
    def test_cleaning_excludes_unknown_customers_cancellations_and_nonpositive_lines(self):
        rows = pd.DataFrame({
            "Customer ID": [1, None, 1, 1, 1, 1],
            "Invoice": ["A", "B", "C123", "D", "E", "F"],
            "InvoiceDate": pd.to_datetime(["2011-01-01"] * 5 + [None]),
            "Quantity": [2, 1, 1, 0, 1, 1],
            "Price": [3, 3, 3, 3, 0, 3],
        })
        cleaned = clean_purchases(rows)
        self.assertEqual(cleaned["Invoice"].tolist(), ["A"])
        self.assertEqual(cleaned["line_revenue"].tolist(), [6])

    def test_cohort_uses_strict_history_and_future_boundaries(self):
        cutoff = pd.Timestamp("2011-06-01")
        orders = pd.DataFrame({
            "Customer ID": [1, 1, 2, 2, 3, 4],
            "Invoice": ["old", "last", "boundary", "future", "end", "future-only"],
            "order_date": [
                cutoff - pd.Timedelta(days=181),
                cutoff - pd.Timedelta(days=1),
                cutoff - pd.Timedelta(days=180),
                cutoff,
                cutoff + pd.Timedelta(days=90),
                cutoff + pd.Timedelta(days=1),
            ],
            "order_value": [999, 10, 20, 30, 40, 50],
        })
        cohort = build_cohort(orders, cutoff).set_index("Customer ID")
        self.assertEqual(set(cohort.index), {1, 2})
        self.assertEqual(cohort.loc[1, "spend_180d"], 10)
        self.assertEqual(cohort.loc[2, "spend_180d"], 20)
        self.assertEqual(cohort.loc[1, "did_not_return_90d"], 1)
        self.assertEqual(cohort.loc[2, "returned_90d"], 1)

    def test_multiple_lines_make_one_order(self):
        purchases = pd.DataFrame({
            "Customer ID": [1, 1], "Invoice": ["A", "A"],
            "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-01"]),
            "line_revenue": [6, 4],
        })
        orders = make_orders(purchases)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders.iloc[0]["order_value"], 10)


if __name__ == "__main__":
    unittest.main()
