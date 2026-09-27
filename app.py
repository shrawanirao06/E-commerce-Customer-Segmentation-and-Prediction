"""
Streamlit app: E-commerce Customer Segmentation and Prediction.

Refactored from the original single-file app.py to delegate all modelling
logic to the `ecommerce_segmentation` package (see src/ecommerce_segmentation/).
The UI behaviour for a user is unchanged; what changed is *where* the model
loading and prediction code lives, so it can be unit-tested and reused by
train_pipeline.py without copy-pasting.
"""

from __future__ import annotations

import streamlit as st

from ecommerce_segmentation.config import Config
from ecommerce_segmentation.inference import load_artifacts, score_customer


@st.cache_resource
def get_artifacts():
    config = Config()
    return load_artifacts(config)


st.title("E-commerce Customer Segmentation and Prediction")
st.write(
    "Enter customer purchase information to identify the customer segment "
    "and predict whether the customer is likely to make a future purchase."
)

try:
    artifacts = get_artifacts()
except FileNotFoundError as exc:
    st.error(
        "Model artifacts not found. Run `python -m ecommerce_segmentation.train_pipeline` "
        f"to generate them, or check that models/ contains the expected files.\n\n{exc}"
    )
    st.stop()

recency = st.number_input("Recency (days)", min_value=0, value=30)
frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5)
monetary = st.number_input("Monetary Value", min_value=0.0, value=1000.0)
average_order_value = st.number_input("Average Order Value", min_value=0.0, value=200.0)

if st.button("Predict"):
    result = score_customer(
        artifacts,
        recency=recency,
        frequency=frequency,
        monetary=monetary,
        average_order_value=average_order_value,
    )

    st.subheader("Customer Segment")
    st.info(result["segment"])

    st.subheader("Future Purchase Prediction")
    if result["prediction"] == "No Future Purchase":
        st.warning(f"Prediction: {result['prediction']}")
    else:
        st.success(f"Prediction: {result['prediction']}")
