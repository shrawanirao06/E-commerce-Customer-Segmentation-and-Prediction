"""
Minimal smoke tests. These do not touch the real 540k-row dataset (too slow
for CI) - they build a tiny synthetic transaction log so the feature
engineering, segmentation and churn-model code paths are exercised.

Run with: pytest
"""

from __future__ import annotations

import pandas as pd
import pytest

from ecommerce_segmentation import feature_engineering as fe
from ecommerce_segmentation.churn_model import ChurnModel
from ecommerce_segmentation.segmentation import SegmentModel


@pytest.fixture
def toy_transactions() -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2023-01-01")
    for customer_id, n_orders, price in [(1, 10, 50), (2, 1, 10), (3, 5, 100), (4, 2, 20)]:
        for i in range(n_orders):
            rows.append(
                {
                    "InvoiceNo": f"{customer_id}-{i}",
                    "CustomerID": customer_id,
                    "InvoiceDate": base + pd.Timedelta(days=i * 10),
                    "Quantity": 2,
                    "UnitPrice": price,
                    "Revenue": 2 * price,
                }
            )
    return pd.DataFrame(rows)


def test_compute_rfm_shapes(toy_transactions):
    reference_date = toy_transactions["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = fe.compute_rfm(toy_transactions, reference_date)
    assert set(rfm.columns) == {"Recency", "Frequency", "Monetary"}
    assert len(rfm) == 4


def test_segment_model_roundtrip(tmp_path, toy_transactions):
    reference_date = toy_transactions["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = fe.compute_rfm(toy_transactions, reference_date)

    model = SegmentModel.fit(rfm, n_clusters=2, random_state=42)
    labeled = model.predict(rfm)
    assert "Segment" in labeled.columns

    kmeans_path = tmp_path / "kmeans.pkl"
    scaler_path = tmp_path / "scaler.pkl"
    labels_path = tmp_path / "labels.json"
    model.save(kmeans_path, scaler_path, labels_path)

    reloaded = SegmentModel.load(kmeans_path, scaler_path, labels_path)
    assert reloaded.segment_names == model.segment_names


def test_churn_model_predicts_label():
    X_train = pd.DataFrame(
        {
            "Recency": [5, 200, 10, 300],
            "Frequency": [10, 1, 8, 1],
            "Monetary": [1000, 50, 800, 20],
            "AverageOrderValue": [100, 50, 100, 20],
        }
    )
    y_train = pd.Series([0, 1, 0, 1])

    model = ChurnModel.fit(X_train, y_train, random_state=42)
    label = model.predict_label(recency=5, frequency=10, monetary=1000, average_order_value=100)
    assert label in {"Future Purchase", "No Future Purchase"}
