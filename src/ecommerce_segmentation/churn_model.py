"""
"Will this customer buy again?" classifier.

Mirrors notebook cells 69-95 ("PREDICTION TARGET CREATION" through
"FINAL CUSTOMER PREDICTIONS"). The notebook trained and compared five
classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient
Boosting, SVM) and kept Gradient Boosting. That comparison step is
reproduced in `train_pipeline.py`; this module only wraps the *chosen*
model so the app and any batch-scoring script have one clear place to
load it from.

Note on scaling: the notebook fit a StandardScaler for this stage
(`prediction_scaler.pkl`) but only ever used it for Logistic Regression /
SVM - the Gradient Boosting model that was actually saved and shipped was
trained on the *raw*, unscaled features. `ChurnModel.predict` below does
the same (no scaling), consistent with how the shipped model was trained.
The scaler is still produced/saved by the training pipeline for parity
with the original artifacts and in case a future model needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

FEATURE_COLUMNS = ["Recency", "Frequency", "Monetary", "AverageOrderValue"]

LABELS = {0: "Future Purchase", 1: "No Future Purchase"}


@dataclass
class ChurnModel:
    model: GradientBoostingClassifier

    @classmethod
    def fit(cls, X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> "ChurnModel":
        model = GradientBoostingClassifier(random_state=random_state)
        model.fit(X_train[FEATURE_COLUMNS], y_train)
        return cls(model=model)

    def predict_label(self, recency: float, frequency: float, monetary: float, average_order_value: float) -> str:
        row = pd.DataFrame(
            {
                "Recency": [recency],
                "Frequency": [frequency],
                "Monetary": [monetary],
                "AverageOrderValue": [average_order_value],
            }
        )[FEATURE_COLUMNS]
        pred = int(self.model.predict(row)[0])
        return LABELS[pred]

    def predict_batch(self, X: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(X[FEATURE_COLUMNS])
        return pd.Series(preds, index=X.index).map(LABELS)

    def feature_importance(self) -> Dict[str, float]:
        return dict(zip(FEATURE_COLUMNS, self.model.feature_importances_))

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path) -> "ChurnModel":
        return cls(model=joblib.load(path))
