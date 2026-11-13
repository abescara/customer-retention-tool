# How to interpret the retention evaluation

## Reproduce the analysis

Download the original `online_retail_II.xlsx` workbook from the [UCI dataset page](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and put it at `data/raw/online_retail_II.xlsx`. The workbook is not stored in Git. Install the packages in `requirements.txt` and run `python temporal_validation.py` from the repository root. That command loads both years, applies the same purchase rules as the dashboard, and prints a recency-only and model row for each evaluable later cutoff.

## What each row means

- **Test cutoff:** The date on which a historical review list would have been made. Only customers with an order in the preceding 180 days are eligible.
- **Training snapshots:** Earlier cutoffs whose complete 90-day outcome windows ended by the test cutoff. This avoids training on future outcomes.
- **Nonreturn share:** Fraction of eligible customers with no valid purchase in the next 90 days. It is the base rate for that cohort.
- **ROC AUC:** How often a randomly selected nonreturner is ranked above a returner. A value of 0.5 is the random-ranking reference.
- **Average precision:** A ranking summary that depends on the cohort's nonreturn share; compare methods within the same test cutoff.
- **Top-20% nonreturn share:** Among the highest-risk fifth of eligible customers, the observed fraction who did not return. This describes a fixed review capacity.

The **recency-only** score is days since the last order. The **trained model** uses recency, order count, spend, and average order value from the prior 180 days. Both are evaluated on identical customers for a given cutoff. Model training uses only historical snapshots with known outcomes, while each test cutoff is scored from its own preceding purchase history.

## Limits on conclusions

This is a historical time test, not a test on new customers. A customer can occur in multiple snapshots, so rows across cutoffs are correlated. The script does not calculate a confidence interval or claim that one method always wins. The comparison measures ranking of *nonreturn*, not the causal benefit of contacting anyone. The data includes neither campaign assignment nor response, so no revenue lift or prevented churn can be inferred. A real outreach trial would need a contactable customer population and randomized treatment and control groups.
