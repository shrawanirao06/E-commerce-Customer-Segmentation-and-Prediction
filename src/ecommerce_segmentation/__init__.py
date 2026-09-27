"""
ecommerce_segmentation
=======================

A small, importable package for:
  * cleaning the raw UK online-retail transaction log
  * building RFM (Recency, Frequency, Monetary) features
  * segmenting customers with K-Means
  * training a churn / "no future purchase" classifier
  * running both models together for a new customer (used by app.py)

See README.md for the full project layout and usage instructions,
and FEEDBACK.md for a review of the original notebook this package
was refactored from.
"""

from .config import Config

__all__ = ["Config"]
__version__ = "0.2.0"
