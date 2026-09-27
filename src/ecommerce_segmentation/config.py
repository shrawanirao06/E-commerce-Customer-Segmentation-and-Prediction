"""
Central place for every path and "magic number" the pipeline uses.

Nothing in the original notebook was configurable: the data path was a
hard-coded Windows path (``C:/Users/SHRAWANI/Downloads/...``), the K-Means
cluster count was retyped in three different cells, and the prediction
cut-off date was a bare literal. Collecting them here means every module
and the Streamlit app agree on the same values, and the whole project can
be pointed at a different machine by changing one file (or a handful of
environment variables).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root = two levels up from this file (src/ecommerce_segmentation/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var_name: str, default: Path) -> Path:
    value = os.environ.get(var_name)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Config:
    # ---- directories -----------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: _env_path(
        "ECS_DATA_DIR", PROJECT_ROOT / "data" / "raw"
    ))
    models_dir: Path = field(default_factory=lambda: _env_path(
        "ECS_MODELS_DIR", PROJECT_ROOT / "models"
    ))
    outputs_dir: Path = field(default_factory=lambda: _env_path(
        "ECS_OUTPUTS_DIR", PROJECT_ROOT / "outputs"
    ))

    # ---- input file --------------------------------------------------
    raw_data_filename: str = "data.csv"
    raw_data_encoding: str = "ISO-8859-1"

    # ---- artifact filenames (kept identical to the original project so
    #      the Streamlit app / notebook / PPT references keep working) --
    kmeans_model_filename: str = "customer_segmentation_kmeans_model.pkl"
    rfm_scaler_filename: str = "rfm_scaler.pkl"
    segment_labels_filename: str = "segment_labels.json"
    churn_model_filename: str = "gradient_boosting_customer_prediction_model.pkl"
    churn_scaler_filename: str = "prediction_scaler.pkl"
    final_predictions_filename: str = "final_customer_predictions.csv"

    # ---- modelling constants ------------------------------------------
    random_state: int = 42
    n_clusters: int = 4  # see FEEDBACK.md: the notebook settled on 4 clusters
    #                      for its business narrative but only ever *saved*
    #                      a 2-cluster model. This package saves whatever
    #                      n_clusters is set here.
    test_size: float = 0.20
    # Cut-off used to build the "did this customer buy again?" label.
    # Historical period: [start, prediction_split_date)
    # Future / label window: [prediction_split_date, end of data]
    prediction_split_date: str = "2011-10-10"

    # ---- convenience properties -----------------------------------------
    @property
    def raw_data_path(self) -> Path:
        return self.data_dir / self.raw_data_filename

    @property
    def kmeans_model_path(self) -> Path:
        return self.models_dir / self.kmeans_model_filename

    @property
    def rfm_scaler_path(self) -> Path:
        return self.models_dir / self.rfm_scaler_filename

    @property
    def segment_labels_path(self) -> Path:
        return self.models_dir / self.segment_labels_filename

    @property
    def churn_model_path(self) -> Path:
        return self.models_dir / self.churn_model_filename

    @property
    def churn_scaler_path(self) -> Path:
        return self.models_dir / self.churn_scaler_filename

    @property
    def final_predictions_path(self) -> Path:
        return self.outputs_dir / self.final_predictions_filename

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.models_dir, self.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)
