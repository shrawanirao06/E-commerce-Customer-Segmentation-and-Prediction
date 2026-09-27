"""
Thin convenience layer used by app.py so the Streamlit UI code has no
modelling logic in it at all - it just calls `load_artifacts()` once and
`score_customer(...)` per button click.
"""

from __future__ import annotations

from dataclasses import dataclass

from .churn_model import ChurnModel
from .config import Config
from .segmentation import SegmentModel


@dataclass
class Artifacts:
    segment_model: SegmentModel
    churn_model: ChurnModel


def load_artifacts(config: Config) -> Artifacts:
    segment_model = SegmentModel.load(
        config.kmeans_model_path,
        config.rfm_scaler_path,
        config.segment_labels_path,
    )
    churn_model = ChurnModel.load(config.churn_model_path)
    return Artifacts(segment_model=segment_model, churn_model=churn_model)


def score_customer(
    artifacts: Artifacts,
    recency: float,
    frequency: float,
    monetary: float,
    average_order_value: float,
) -> dict:
    segment = artifacts.segment_model.predict_one(recency, frequency, monetary)
    purchase_prediction = artifacts.churn_model.predict_label(
        recency, frequency, monetary, average_order_value
    )
    return {"segment": segment, "prediction": purchase_prediction}
