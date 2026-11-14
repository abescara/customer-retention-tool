"""Historical decision-support view; run with `streamlit run app.py`."""

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation import capacity_comparison, select_priority
from modeling import eligible_training_snapshots, fit_risk_model, nonreturn_risk
from retention import CUTOFFS, build_cohort, load_orders


st.set_page_config(page_title="Customer Retention Tool", layout="wide")
st.title("Customer Retention Tool")
st.caption("Historical UK retail data · 180-day purchase history · 90-day return outcome")

source = Path(__file__).parent / "data/raw/online_retail_II.xlsx"
if not source.exists():
    st.error("Download Online Retail II from UCI and put online_retail_II.xlsx in data/raw/.")
    st.stop()


@st.cache_data(show_spinner="Reading and preparing the workbook…")
def get_snapshots(path: str, modified: float) -> pd.DataFrame:
    orders = load_orders(path)
    if CUTOFFS[-1] + pd.Timedelta(days=90) > orders["order_date"].max():
        raise ValueError("The workbook does not cover the full test outcome window.")
    return pd.concat([build_cohort(orders, date) for date in CUTOFFS], ignore_index=True)


snapshots = get_snapshots(str(source), source.stat().st_mtime)
train = eligible_training_snapshots(snapshots, CUTOFFS[-1])
test = snapshots[snapshots["cutoff"] == CUTOFFS[-1]].copy()

model = fit_risk_model(train)
test["model_risk"] = nonreturn_risk(model, test)

st.info(
    "This is a replay of the September 2011 test period. The 90-day outcomes are "
    "shown only for evaluation; the ranking uses purchase history before September 1."
)
share = st.slider("Share of customers to review", min_value=5, max_value=50, value=20, step=5)
method = st.radio("Ranking method", ["Trained model", "Recency only"], horizontal=True)
score = "model_risk" if method == "Trained model" else "days_since_last_order"
priority = select_priority(test, score, share)

left, middle, right = st.columns(3)
left.metric("Eligible customers", f"{len(test):,}")
middle.metric("Customers to review", f"{len(priority):,}")
right.metric("Historical nonreturn share in selected group", f"{priority['did_not_return_90d'].mean():.1%}")

st.subheader("Model comparison on the later period")
actual = test["did_not_return_90d"]
comparison = pd.DataFrame({
    "Method": ["Recency only", "Trained model"],
    "ROC AUC": [roc_auc_score(actual, test["days_since_last_order"]), roc_auc_score(actual, test["model_risk"])],
    "Average precision": [average_precision_score(actual, test["days_since_last_order"]), average_precision_score(actual, test["model_risk"])],
})
st.dataframe(comparison.style.format({"ROC AUC": "{:.3f}", "Average precision": "{:.3f}"}), hide_index=True)

st.subheader("What different review capacities find")
st.write("Nonreturner capture is the share of all customers who did not return that appear on the review list. Results are historical test outcomes, not estimated campaign gains.")
capacity = capacity_comparison(test)
st.dataframe(capacity.style.format({"Selected nonreturn share": "{:.1%}", "Nonreturner capture": "{:.1%}"}), hide_index=True)

st.subheader("Customers for review")
st.write("Review recent order history and spend before deciding whether a customer warrants outreach. The trained-model score estimates historical nonreturn risk; the recency score is simply days since the last order. Neither proves an offer will change behavior.")
visible = priority[["Customer ID", "days_since_last_order", "order_count_180d", "spend_180d", "average_order_value"]].copy()
visible["Customer ID"] = visible["Customer ID"].astype(int)
visible["ranking_score"] = priority[score]
visible.insert(1, "review_reason", visible.apply(
    lambda row: f"Days since last order: {int(row['days_since_last_order'])}; orders in 180 days: {int(row['order_count_180d'])}",
    axis=1,
))
score_format = "{:.3f}" if method == "Trained model" else "{:.0f}"
st.dataframe(visible.style.format({"spend_180d": "{:,.2f}", "average_order_value": "{:,.2f}", "ranking_score": score_format}), hide_index=True)
st.download_button("Download review list", visible.to_csv(index=False).encode("utf-8"), "customer_review_list.csv", "text/csv")

st.caption("Limits: this historical data has no campaign assignments or contact outcomes. The model does not measure outreach benefit; customers can recur across training snapshots and the later test cohort. Customer ID is a pseudonymous dataset identifier; do not treat this export as a contact list.")
