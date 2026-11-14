"""One shared definition of the historical nonreturn model."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from retention import FEATURE_COLUMNS


def eligible_training_snapshots(snapshots: pd.DataFrame, test_cutoff: str | pd.Timestamp) -> pd.DataFrame:
    """Use labels only after their entire 90-day outcome window has ended."""
    cutoff = pd.Timestamp(test_cutoff)
    dates = pd.to_datetime(snapshots["cutoff"])
    return snapshots.loc[dates + pd.Timedelta(days=90) <= cutoff].copy()


def fit_risk_model(train: pd.DataFrame):
    """Fit only on supplied, already-labelled historical customer snapshots."""
    model = make_pipeline(
        FunctionTransformer(np.log1p),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(train[FEATURE_COLUMNS], train["did_not_return_90d"])
    return model


def nonreturn_risk(model, customers: pd.DataFrame) -> np.ndarray:
    """Return the probability of no order in the following 90 days."""
    return model.predict_proba(customers[FEATURE_COLUMNS])[:, 1]
