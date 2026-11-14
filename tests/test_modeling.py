"""Check the shared model scores only from the supplied feature columns."""

import unittest

import numpy as np
import pandas as pd

from modeling import eligible_training_snapshots, fit_risk_model, nonreturn_risk


class ModelingTests(unittest.TestCase):
    def test_training_excludes_unfinished_outcome_windows(self):
        snapshots = pd.DataFrame({"cutoff": pd.to_datetime([
            "2011-03-01", "2011-06-01", "2011-08-01", "2011-09-01"
        ])})
        eligible = eligible_training_snapshots(snapshots, "2011-09-01")
        self.assertEqual(eligible["cutoff"].dt.strftime("%Y-%m-%d").tolist(), [
            "2011-03-01", "2011-06-01"
        ])

    def test_risk_is_probability_and_does_not_require_outcomes_at_scoring(self):
        train = pd.DataFrame({
            "days_since_last_order": [1, 3, 10, 20, 30, 50],
            "order_count_180d": [4, 3, 3, 2, 1, 1],
            "spend_180d": [400, 300, 200, 100, 50, 10],
            "average_order_value": [100, 100, 67, 50, 50, 10],
            "did_not_return_90d": [0, 0, 0, 1, 1, 1],
        })
        model = fit_risk_model(train)
        scored = nonreturn_risk(model, train.drop(columns="did_not_return_90d"))
        self.assertEqual(len(scored), len(train))
        self.assertTrue(np.all((scored >= 0) & (scored <= 1)))
        self.assertGreater(scored[-1], scored[0])


if __name__ == "__main__":
    unittest.main()
