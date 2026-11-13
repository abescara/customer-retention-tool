"""Prevent a later cohort from learning labels that are not yet known."""

import unittest

import pandas as pd

from temporal_validation import evaluate_snapshots


class TemporalValidationTests(unittest.TestCase):
    def test_only_complete_prior_outcome_windows_enter_training(self):
        dates = pd.to_datetime(["2010-01-01", "2010-03-01", "2010-04-01", "2010-07-01"])
        rows = []
        for date in dates:
            for i in range(10):
                rows.append({
                    "cutoff": date, "Customer ID": i,
                    "days_since_last_order": i + 1,
                    "order_count_180d": 10 - i,
                    "spend_180d": 100 - i * 5,
                    "average_order_value": 20 - i,
                    "did_not_return_90d": int(i >= 5),
                })
        results = evaluate_snapshots(pd.DataFrame(rows))
        april = results.loc[results["test_cutoff"] == "2010-04-01"]
        july = results.loc[results["test_cutoff"] == "2010-07-01"]
        self.assertEqual(set(april["training_snapshots"]), {1})
        self.assertEqual(set(july["training_snapshots"]), {3})
        self.assertEqual(len(results), 4)


if __name__ == "__main__":
    unittest.main()
