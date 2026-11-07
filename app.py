"""Historical decision-support view; run with `streamlit run app.py`."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from retention import CUTOFFS, FEATURE_COLUMNS, build_cohort, load_orders


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
train = snapshots[snapshots["cutoff"] < CUTOFFS[-1]]
test = snapshots[snapshots["cutoff"] == CUTOFFS[-1]].copy()

model = make_pipeline(
    FunctionTransformer(np.log1p),
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42),
)
model.fit(train[FEATURE_COLUMNS], train["did_not_return_90d"])
test["model_risk"] = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
test["recency_rank"] = test["days_since_last_order"].rank(pct=True)

st.info(
    "This is a replay of the September 2011 test period. The 90-day outcomes are "
    "shown only for evaluation; the ranking uses purchase history before September 1."
)
share = st.slider("Share of customers to review", min_value=5, max_value=50, value=20, step=5)
method = st.radio("Ranking method", ["Trained model", "Recency only"], horizontal=True)
score = "model_risk" if method == "Trained model" else "recency_rank"
count = int(np.ceil(len(test) * share / 100))
priority = test.nlargest(count, score).copy()

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

st.subheader("Customers for review")
st.write("Review recent order history and spend before deciding whether a customer warrants outreach. A high score is a risk estimate, not proof an offer will change their behavior.")
visible = priority[["Customer ID", "days_since_last_order", "order_count_180d", "spend_180d", "average_order_value", score]].copy()
visible = visible.rename(columns={score: "ranking_score"})
st.dataframe(visible.style.format({"spend_180d": "{:,.2f}", "average_order_value": "{:,.2f}", "ranking_score": "{:.3f}"}), hide_index=True)
st.download_button("Download review list", visible.to_csv(index=False).encode("utf-8"), "customer_review_list.csv", "text/csv")

st.caption("Limits: this historical data has no campaign assignments or contact outcomes. The model does not measure outreach benefit; customers can recur across training snapshots and the later test cohort.")
