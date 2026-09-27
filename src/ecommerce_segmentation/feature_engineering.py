"""
RFM (Recency, Frequency, Monetary) feature engineering.

Mirrors notebook cells 29-45 ("EXPLORATORY DATA ANALYSIS" / "RFM ANALYSIS"),
extracted into functions that take an explicit `reference_date` instead of
recomputing `df['InvoiceDate'].max()` in several different places (the
notebook does this independently for the "full data" RFM, the churn-model
features, and the "historical" segmentation RFM - three separate reference
dates that are easy to lose track of).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_rfm(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build one row per CustomerID with Recency / Frequency / Monetary,
    all measured as of `reference_date`.
    """
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    )
    return rfm


def log_transform(rfm: pd.DataFrame) -> pd.DataFrame:
    """log1p transform to reduce the heavy right-skew of RFM columns."""
    return np.log1p(rfm[["Recency", "Frequency", "Monetary"]])


def fit_scaler(rfm_log: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(rfm_log)
    return scaler


def scale(rfm_log: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    scaled = scaler.transform(rfm_log)
    return pd.DataFrame(scaled, columns=rfm_log.columns, index=rfm_log.index)


def build_churn_features(
    historical_df: pd.DataFrame, prediction_split_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Per-customer feature table used by the churn model: Recency, Frequency,
    Monetary and AverageOrderValue, all computed only from transactions
    *before* `prediction_split_date` (mirrors notebook cell 72).
    """
    features = historical_df.groupby("CustomerID").agg(
        LastPurchaseDate=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    )
    features["Recency"] = (prediction_split_date - features["LastPurchaseDate"]).dt.days
    features["AverageOrderValue"] = features["Monetary"] / features["Frequency"]
    return features[["Recency", "Frequency", "Monetary", "AverageOrderValue"]]


def build_churn_target(
    historical_df: pd.DataFrame, future_df: pd.DataFrame
) -> pd.Series:
    """
    Target = 1 if the customer did NOT appear again in `future_df`
    (i.e. "no future purchase" / churned), 0 if they did buy again.
    Mirrors notebook cell 71.
    """
    future_customers = set(future_df["CustomerID"].unique())
    historical_customers = historical_df["CustomerID"].unique()
    target = pd.Series(
        [0 if c in future_customers else 1 for c in historical_customers],
        index=historical_customers,
        name="Target",
    )
    return target
