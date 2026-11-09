# Data quality and sensitivity audit

Run `python audit_data.py` after placing the UCI Online Retail II workbook in `data/raw/`. All monetary values below are in the dataset's recorded currency; the source is a UK retailer.

| Check | Result |
| --- | ---: |
| Source lines | 1,067,371 |
| Exact duplicate source lines | 34,335 |
| Valid purchase lines under the documented rules | 805,549 |
| Exact duplicates among valid purchase lines | 26,124 |
| Revenue keeping valid duplicate lines | 17,743,429.18 |
| Revenue after removing exact duplicates | 17,374,804.27 |
| Largest order | 168,469.60 |
| 99th percentile order value | 3,697.92 |
| Revenue share of the highest-value 1% of orders | 17.06% |

An exact duplicate row is not necessarily an accidental duplicate purchase: the workbook does not provide a unique line-item ID. Removing these rows lowers recorded revenue by 368,624.91 (2.08%). We keep them in the main analysis to preserve the notebook's original cleaning rules and make the choice explicit.

The largest orders may represent bulk buying, but the workbook alone cannot confirm why they are large. The sensitivity check removes orders above the dataset-wide 99th percentile solely to show how results respond; it is not a recommended default rule.

| Scenario | September 2011 eligible customers | Did not return within 90 days | Later-period model ROC AUC |
| --- | ---: | ---: | ---: |
| Main analysis | 2,772 | 39.29% | 0.727 |
| Exact duplicate lines removed | 2,772 | 39.29% | 0.727 |
| Highest-value 1% of orders removed | 2,767 | 39.32% | 0.727 |

ROC AUC is rounded to three decimals. Matching rounded values do not imply identical predictions or identical customer review lists. The top-order check also changes the eligible population, so it is not a controlled comparison of predictions on precisely the same customers.
