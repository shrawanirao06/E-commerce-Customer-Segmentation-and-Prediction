# Project Feedback

Review of the original `Ecommerce_Customer_Segmentation` deliverable (one
`app.py`, one 122-cell notebook, four `.pkl` files, a data CSV, a PPT, and
a `requirements.txt`) before it was restructured into the
`ecommerce_segmentation` package. Findings are ordered roughly by
severity: correctness issues first, then structure/maintainability, then
smaller polish items.

## 1. Correctness issues

### 1.1 The shipped segmentation model doesn't match the notebook's own analysis (bug)

This is the most important finding. In the notebook:

- Cell 49 fits a **2-cluster** K-Means model and saves it to
  `customer_segmentation_kmeans_model.pkl` — this is the file `app.py`
  actually loads.
- Cell 57 then re-fits a **4-cluster** K-Means model (`final_kmeans`) on
  the same data and uses *that* one for every segment profile table,
  chart, and the "FINAL CUSTOMER INSIGHTS" / "BUSINESS RECOMMENDATIONS"
  narrative later in the notebook — but this 4-cluster model is **never
  saved**.

Net effect: the deployed Streamlit app tells every user they're either a
generic *"High-Value / Highly-Engaged Customer"* or *"Low-Value /
Less-Engaged Customer"* (cluster 1 vs. everything else), while the
notebook's actual conclusions and business recommendations are written
for a **4-segment** model (`Recent Low-Value`, `Moderate`, `High-Value
Active`, `At-Risk / Inactive`) that was never persisted. Anyone reading
the PPT/notebook and then using the live app would see two different,
irreconcilable segmentation schemes.

**Fix applied:** `train_pipeline.py` fits and saves one 4-cluster model
(`Config.n_clusters`, default 4) and `app.py` now shows the same 4-segment
scheme the analysis describes. Re-running it reproduced the same model
comparison numbers as the notebook's hard-coded table (e.g. Gradient
Boosting accuracy 0.6644, matching cell 114 exactly), which is a good
sign the underlying pipeline itself is sound — it was specifically the
"which model got saved" step that had the mismatch.

### 1.2 Cluster IDs were treated as if they were stable, human-meaningful labels

Separately from 1.1: the notebook hard-codes
`segment_names = {0: "Recent Low-Value...", 1: "Moderate...", 2: "High-Value...", 3: "At-Risk..."}`
once (cell 59) and reuses the identical dict for a **second, independently
fit** K-Means model later on (cell 104, the "historical" segmentation used
to build the final customer table). K-Means cluster IDs are arbitrary —
cluster `2` in one `.fit()` call has no guaranteed relationship to cluster
`2` in a different `.fit()` call, even on similar data. There's no
assertion anywhere that both fits actually produced clusters in the same
order, so segment names in the final output table could silently be
wrong for one of the two clusterings.

**Fix applied:** `segmentation.py::SegmentModel` derives segment names
from each fitted cluster's own Recency/Frequency/Monetary medians every
time (ranks clusters worst→best by a simple combined z-score), so the
label always matches that specific model's actual cluster behaviour,
regardless of ID ordering or how many times the model is refit.

### 1.3 `prediction_scaler.pkl` is shipped but never used at inference time

The notebook fits `scaler_prediction` (cell 77) and saves it (cell 93),
but the model that actually got saved and deployed — Gradient Boosting —
was trained on the **unscaled** `X_train`, not `X_train_scaled` (only
Logistic Regression and SVM used the scaled version, and neither was
kept). `app.py` correctly never applies this scaler, but its presence
among the shipped artifacts is misleading: it looks load-bearing and
isn't. This is a "why is this file here" trap for whoever maintains the
project next.

**Fix applied:** kept for parity (`churn_model.py`'s docstring explains
why it's unused), but documented clearly rather than silently shipped.
If a future model does need scaled inputs, the artifact is still there.

## 2. Structural / maintainability issues (original layout)

- **Everything in one flat folder** — data, notebook, PPT, 4 model
  files, and the Streamlit script all sat side by side with no
  separation between "raw input", "trained artifact", "generated output",
  and "documentation".
- **No reusable modelling code** — all logic (cleaning, RFM, clustering,
  training, evaluation) existed only as sequential notebook cells with
  print-statement side effects. `app.py` re-implemented the RFM/scaling
  steps from scratch by hand rather than importing shared functions, so
  the notebook and the app were two independent copies of the same logic
  that could (and did, per §1.1) drift apart.
- **Hard-coded, machine-specific paths** — e.g.
  `r"C:/Users/SHRAWANI/Downloads/Ecommerce_Customer_Segmentation/data.csv"`
  and a matching hard-coded save path for the K-Means model. The notebook
  cannot be rerun on a different machine without manually editing several
  cells.
- **No tests** — none of the feature engineering or modelling logic had
  any automated check, which is how issues like §1.1 go unnoticed.
- **No pinned/complete dependency list for the notebook** — `requirements.txt`
  only covers what `app.py` needs (streamlit/pandas/numpy/scikit-learn/joblib);
  running the notebook also needs `matplotlib` and `seaborn`, which aren't
  listed anywhere.

**Fix applied:** the new layout separates `data/`, `models/`, `outputs/`,
`docs/`, `notebooks/`, and an installable `src/ecommerce_segmentation/`
package; `Config` centralizes every path/constant; `train_pipeline.py`
gives a single reproducible command instead of a notebook that must be
run top-to-bottom by hand; `tests/` has smoke tests for the RFM,
segmentation, and churn-model code paths; `requirements-dev.txt` adds the
notebook-only dependencies.

## 3. Smaller items worth knowing about (not changed, just flagged)

- **Model-selection methodology:** classifiers were compared on a single
  80/20 stratified split (cell 76) rather than cross-validated for
  selection (5-fold CV in cell 92 was run only *after* Gradient Boosting
  had already been picked, as a sanity check, not as the selection
  criterion). With Logistic Regression, SVM, and Gradient Boosting all
  landing within ~1 F1 point of each other (0.709 / 0.711 / 0.700), the
  "winner" is fairly sensitive to the random split. Worth a proper CV
  comparison before treating Gradient Boosting's edge as meaningful.
- **Short label window:** the churn label is "did this customer buy again
  after 2011-10-10?" — roughly a 2-month forward window against ~10 months
  of history. `Recency` at the cut-off date is, almost by construction,
  the strongest predictor of that label (see the notebook's own feature
  importance table), so a meaningful part of the model's lift may just be
  "customers who already look inactive tend to stay inactive for two more
  months" rather than a subtler pattern. Not a bug, but worth stating
  explicitly in any write-up of model performance, and worth testing with
  a longer forward window if more recent data becomes available.
- **Zero-price rows retained:** rows with `UnitPrice == 0` were inspected
  (cells 18/22/23) but intentionally not dropped or flagged in the final
  pipeline; they contribute `Revenue = 0` to Monetary, which is
  reasonable, but this decision wasn't stated anywhere in the notebook's
  written conclusions.
- **No error handling in the original `app.py`** for missing model files —
  a `joblib.load` failure at import time would crash the whole app with a
  raw traceback. The refactored `app.py` catches this and shows an actionable
  message instead.

## Summary

The core modelling approach (RFM → K-Means segmentation, time-split →
Gradient Boosting churn model) is sound and the notebook's business
narrative is coherent — the main problem was **operational**: the
notebook's real conclusions (4 segments) never made it into the deployed
artifact (2 segments), because of a re-fit-without-re-save mistake, and
there was no shared code path to catch that kind of drift. Restructuring
into a package with one training entry point and tests closes that gap
going forward.
