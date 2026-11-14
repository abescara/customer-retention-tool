"""Check capacity selection and measurement on a small known cohort."""

import unittest

import pandas as pd

from evaluation import capacity_comparison, ranking_overlap, select_priority


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.customers = pd.DataFrame({
            "Customer ID": [4, 3, 2, 1, 5],
            "days_since_last_order": [10, 40, 30, 20, 40],
            "model_risk": [0.9, 0.1, 0.8, 0.2, 0.3],
            "did_not_return_90d": [1, 0, 1, 0, 1],
        })

    def test_selection_uses_capacity_and_stable_tie_break(self):
        selected = select_priority(self.customers, "days_since_last_order", 40)
        self.assertEqual(selected["Customer ID"].tolist(), [3, 5])

    def test_comparison_counts_selected_outcomes(self):
        summary = capacity_comparison(self.customers, shares=(40,))
        model = summary.loc[summary["Method"] == "Trained model"].iloc[0]
        self.assertEqual(model["Customers reviewed"], 2)
        self.assertEqual(model["Nonreturners found"], 2)
        self.assertAlmostEqual(model["Nonreturner capture"], 2 / 3)

    def test_overlap_explains_distinct_review_lists(self):
        overlap = ranking_overlap(self.customers, 40)
        self.assertEqual(overlap, {
            "shared_customers": 0,
            "recency_only_customers": 2,
            "model_only_customers": 2,
            "recency_only_nonreturners": 1,
            "model_only_nonreturners": 2,
        })


if __name__ == "__main__":
    unittest.main()
