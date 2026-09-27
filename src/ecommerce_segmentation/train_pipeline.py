"""
End-to-end training pipeline (CLI).

Run with:

    python -m ecommerce_segmentation.train_pipeline

This replaces the original 122-cell notebook run for anyone who just wants
regenerated artifacts. The notebook itself is kept under notebooks/ for the
exploratory analysis, plots and narrative write-up - this script only
reproduces the parts that actually produce the files under models/ and
outputs/.

What it does, in order:
  1. Load + clean the raw transaction log.
  2. Fit the RFM segmentation model (K-Means) on the full data and save it,
     together with human-readable segment names (see segmentation.py).
  3. Build the historical/future split for the churn model, train and
     compare five classifiers, and save the Gradient Boosting model (the
     one the notebook selected) plus a small comparison table.
  4. Score every historical customer with both models and write
     outputs/final_customer_predictions.csv.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import data_loading as dl
from . import feature_engineering as fe
from .churn_model import FEATURE_COLUMNS, ChurnModel
from .config import Config
from .segmentation import SegmentModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def compare_classifiers(X_train, X_test, y_train, y_test, random_state: int) -> pd.DataFrame:
    """Reproduces notebook cells 79-85: train + score 5 candidate models."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        "Logistic Regression": (LogisticRegression(random_state=random_state), True),
        "Decision Tree": (DecisionTreeClassifier(random_state=random_state), False),
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=random_state), False),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=random_state), False),
        "SVM": (SVC(random_state=random_state), True),
    }

    rows = []
    for name, (model, needs_scaling) in candidates.items():
        Xtr = X_train_scaled if needs_scaling else X_train
        Xte = X_test_scaled if needs_scaling else X_test
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds),
                "Recall": recall_score(y_test, preds),
                "F1-Score": f1_score(y_test, preds),
            }
        )

    return pd.DataFrame(rows), scaler


def run(config: Config) -> None:
    config.ensure_dirs()

    log.info("Loading and cleaning raw data from %s", config.raw_data_path)
    df = dl.load_and_prepare(config.raw_data_path, encoding=config.raw_data_encoding)

    # ---- 1. Segmentation model (full-period RFM) -------------------------
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = fe.compute_rfm(df, reference_date)

    log.info("Fitting K-Means segmentation model (k=%s)", config.n_clusters)
    segment_model = SegmentModel.fit(rfm, n_clusters=config.n_clusters, random_state=config.random_state)
    segment_model.save(config.kmeans_model_path, config.rfm_scaler_path, config.segment_labels_path)
    log.info("Segment names: %s", segment_model.segment_names)

    rfm_labeled = segment_model.predict(rfm)

    # ---- 2. Churn ("no future purchase") model ---------------------------
    split_date = pd.Timestamp(config.prediction_split_date)
    historical_df = df[df["InvoiceDate"] < split_date].copy()
    future_df = df[df["InvoiceDate"] >= split_date].copy()

    churn_features = fe.build_churn_features(historical_df, split_date)
    churn_target = fe.build_churn_target(historical_df, future_df)
    churn_data = churn_features.join(churn_target)

    X = churn_data[FEATURE_COLUMNS]
    y = churn_data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )

    log.info("Comparing candidate classifiers...")
    comparison, churn_scaler = compare_classifiers(X_train, X_test, y_train, y_test, config.random_state)
    log.info("\n%s", comparison.to_string(index=False))

    log.info("Training final Gradient Boosting churn model")
    churn_model = ChurnModel.fit(X_train, y_train, random_state=config.random_state)
    churn_model.save(config.churn_model_path)

    import joblib

    joblib.dump(churn_scaler, config.churn_scaler_path)  # saved for parity; not used at inference

    # ---- 3. Final scored customer table -----------------------------------
    predictions = churn_model.predict_batch(X)
    final = churn_data.drop(columns=["Target"]).copy()
    final["Predicted_Status"] = predictions
    final = final.join(rfm_labeled[["Segment"]], how="left")
    final.to_csv(config.final_predictions_path, index=True)
    log.info("Wrote %s (%d customers)", config.final_predictions_path, len(final))

    comparison.to_csv(config.outputs_dir / "model_comparison.csv", index=False)
    log.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the segmentation + churn models.")
    parser.add_argument("--data-dir", type=str, default=None, help="Folder containing data.csv")
    parser.add_argument("--models-dir", type=str, default=None, help="Where to save model artifacts")
    parser.add_argument("--outputs-dir", type=str, default=None, help="Where to save scored CSV output")
    args = parser.parse_args()

    from pathlib import Path

    kwargs = {}
    if args.data_dir:
        kwargs["data_dir"] = Path(args.data_dir)
    if args.models_dir:
        kwargs["models_dir"] = Path(args.models_dir)
    if args.outputs_dir:
        kwargs["outputs_dir"] = Path(args.outputs_dir)

    config = Config(**kwargs)
    run(config)


if __name__ == "__main__":
    main()
