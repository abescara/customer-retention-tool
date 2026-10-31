# Customer Retention Tool

An analysis of repeat purchasing using the [UCI Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii). The project asks: among customers who bought in the previous 180 days, which ones buy again in the next 90 days?

## Current work

The [data inspection notebook](notebook/01_data_inspection.ipynb) combines both years of transactions, documents purchase-cleaning rules, creates one row per invoice, and compares six historical customer cohorts. A prediction model and decision-support dashboard are planned next.

The cleaned data has 805,549 purchase line items, 36,969 orders, and 5,878 identified customers. At the June 1, 2011 cutoff, 1,331 of 2,659 eligible customers bought again within 90 days. These are descriptive results, not model performance or evidence that an outreach campaign would work.

## Run the notebook

1. Download `online_retail_II.xlsx` from the [UCI dataset page](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
2. Place it at `data/raw/online_retail_II.xlsx`. This raw file is ignored by Git.
3. Create a Python environment and install the packages in `requirements.txt`.
4. Open `notebook/01_data_inspection.ipynb` in VS Code and run cells from top to bottom. Reading both Excel sheets can take several minutes.

## Data and limits

The dataset covers a UK retailer from December 2009 to December 2011. Rows without a customer ID cannot support customer-level retention analysis; cancellation invoices and nonpositive quantities or prices are excluded from valid purchases. Exact duplicate lines are kept pending further investigation. The data includes no campaign assignments or outreach responses, so this project cannot measure the causal effect of contacting a customer.

Source: Chen, D. (2012). *Online Retail II*. UCI Machine Learning Repository. [DOI: 10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D). Dataset licensed under CC BY 4.0.
