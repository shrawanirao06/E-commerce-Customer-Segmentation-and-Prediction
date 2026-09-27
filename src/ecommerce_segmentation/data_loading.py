"""
Loading and cleaning the raw transaction log.

Mirrors the "DATA UNDERSTANDING" / "DATA CLEANING AND PREPROCESSING"
sections of the original notebook (cells 1-23), as plain, reusable
functions instead of sequential notebook cells with print-statement
side effects.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_transactions(path: Path, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    """Read the raw UK online-retail transaction CSV."""
    return pd.read_csv(path, encoding=encoding)


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same cleaning steps as the notebook:

    1. Drop rows with a missing CustomerID (can't attribute a transaction
       to a customer, so they're useless for RFM / segmentation).
    2. Parse InvoiceDate to datetime.
    3. Drop exact duplicate rows.
    4. Drop non-positive Quantity rows (returns / cancellations - invoice
       numbers starting with "C", and any other Quantity <= 0 rows).

    Rows with UnitPrice == 0 are intentionally *not* dropped here, matching
    the notebook, which only inspected them (cells 18/22/23) but never
    filtered them out. Revenue for those rows is simply 0.
    """
    cleaned = df.dropna(subset=["CustomerID"]).copy()
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"])
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned[cleaned["Quantity"] > 0].copy()
    return cleaned


def add_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Add a Revenue = Quantity * UnitPrice column."""
    df = df.copy()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


def load_and_prepare(path: Path, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    """Convenience wrapper: load -> clean -> add Revenue."""
    raw = load_transactions(path, encoding=encoding)
    return add_revenue(clean_transactions(raw))
