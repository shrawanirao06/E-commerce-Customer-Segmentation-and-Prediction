"""
Customer segmentation (K-Means on log-scaled RFM features).

Notebook parity + one fix
--------------------------
The original notebook hard-coded a ``{0: "Recent Low-Value Customers", ...}``
dict to turn cluster IDs into business-friendly names. That mapping is only
valid for the *one* K-Means fit it was written next to: K-Means cluster IDs
are arbitrary (cluster "2" in one fit has no relation to cluster "2" in a
different fit on different data/seed/library version), and the notebook
reused the identical dict for a second, independently-fit K-Means model
later on (the "historical" segmentation, cell 104) without checking that
the ID-to-profile mapping still held.

``SegmentModel`` below fixes this by deriving the label *from the fitted
cluster's own RFM medians* every time (rank clusters by Recency ascending
and Monetary descending), so the name always matches the cluster's actual
behaviour, however many clusters there are or however they were ordered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import feature_engineering as fe

# Ordered worst -> best; only the first `n_clusters` are used.
_SEGMENT_NAME_LADDER = [
    "At-Risk / Inactive Customers",
    "Recent Low-Value Customers",
    "Moderate Customers",
    "High-Value Active Customers",
]


def _rank_clusters(rfm_with_cluster: pd.DataFrame) -> Dict[int, str]:
    """
    Rank clusters from "worst" to "best" customer using a simple score
    (higher Monetary and Frequency, lower Recency, is "better"), then map
    them onto human-readable names. This replaces the notebook's hard-coded
    cluster-id -> name dict.
    """
    profile = rfm_with_cluster.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].median()

    # z-score each column across clusters so Recency/Frequency/Monetary are
    # comparable, then combine into one "value" score per cluster.
    z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, 1)
    score = z["Monetary"] + z["Frequency"] - z["Recency"]
    ordered_cluster_ids = score.sort_values().index.tolist()  # worst -> best

    n = len(ordered_cluster_ids)
    ladder = _SEGMENT_NAME_LADDER
    if n != len(ladder):
        # Fall back to generic ordinal names for any other cluster count.
        ladder = [f"Segment {i + 1} of {n} (worst to best)" for i in range(n)]

    return {cluster_id: ladder[i] for i, cluster_id in enumerate(ordered_cluster_ids)}


@dataclass
class SegmentModel:
    """Bundles the fitted scaler, K-Means model, and derived segment names."""

    scaler: StandardScaler
    kmeans: KMeans
    segment_names: Dict[int, str]

    @classmethod
    def fit(cls, rfm: pd.DataFrame, n_clusters: int, random_state: int) -> "SegmentModel":
        rfm_log = fe.log_transform(rfm)
        scaler = fe.fit_scaler(rfm_log)
        rfm_scaled = fe.scale(rfm_log, scaler)

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        clusters = kmeans.fit_predict(rfm_scaled)

        labeled = rfm.copy()
        labeled["Cluster"] = clusters
        segment_names = _rank_clusters(labeled)

        return cls(scaler=scaler, kmeans=kmeans, segment_names=segment_names)

    def predict(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Return `rfm` with added Cluster and Segment columns."""
        rfm_log = fe.log_transform(rfm)
        rfm_scaled = fe.scale(rfm_log, self.scaler)
        clusters = self.kmeans.predict(rfm_scaled)

        out = rfm.copy()
        out["Cluster"] = clusters
        out["Segment"] = out["Cluster"].map(self.segment_names)
        return out

    def predict_one(self, recency: float, frequency: float, monetary: float) -> str:
        """Convenience for a single customer (used by the Streamlit app)."""
        row = pd.DataFrame({"Recency": [recency], "Frequency": [frequency], "Monetary": [monetary]})
        return self.predict(row)["Segment"].iloc[0]

    # -- persistence ---------------------------------------------------
    def save(self, kmeans_path: Path, scaler_path: Path, labels_path: Path) -> None:
        joblib.dump(self.kmeans, kmeans_path)
        joblib.dump(self.scaler, scaler_path)
        labels_path.write_text(json.dumps({str(k): v for k, v in self.segment_names.items()}, indent=2))

    @classmethod
    def load(cls, kmeans_path: Path, scaler_path: Path, labels_path: Path | None = None) -> "SegmentModel":
        kmeans = joblib.load(kmeans_path)
        scaler = joblib.load(scaler_path)

        if labels_path is not None and Path(labels_path).exists():
            raw = json.loads(Path(labels_path).read_text())
            segment_names = {int(k): v for k, v in raw.items()}
        else:
            # Backwards compatibility with artifacts saved before this
            # package existed (no segment_labels.json alongside them):
            # fall back to a generic Low/High split by cluster id.
            segment_names = {
                i: ("High-Value / Highly-Engaged Customer" if i == 1 else "Low-Value / Less-Engaged Customer")
                for i in range(getattr(kmeans, "n_clusters", 2))
            }

        return cls(scaler=scaler, kmeans=kmeans, segment_names=segment_names)
