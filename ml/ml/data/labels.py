"""Parsing of labeled_anomalies.csv (stringified-list columns need ast.literal_eval)
and construction of per-timestep ground-truth anomaly masks over the test split.
"""

import ast

import numpy as np
import pandas as pd

from ml.config import LABELED_ANOMALIES_CSV

_CACHE: pd.DataFrame | None = None


def _parse_class_list(s: str) -> list[str]:
    """The `class` column looks like "[point, contextual]" -- bare, unquoted
    words, so ast.literal_eval can't parse it directly. Split by hand.
    """
    return [token.strip() for token in s.strip("[]").split(",") if token.strip()]


def load_labels_table() -> pd.DataFrame:
    global _CACHE
    if _CACHE is None:
        df = pd.read_csv(LABELED_ANOMALIES_CSV)
        df["anomaly_sequences"] = df["anomaly_sequences"].apply(ast.literal_eval)
        df["class"] = df["class"].apply(_parse_class_list)
        _CACHE = df
    return _CACHE


def get_channel_row(chan_id: str) -> dict:
    df = load_labels_table()
    rows = df[df["chan_id"] == chan_id]
    if rows.empty:
        raise KeyError(f"no labeled_anomalies.csv row for channel {chan_id}")
    return rows.iloc[0].to_dict()


def build_ground_truth_mask(chan_id: str, num_values: int | None = None) -> np.ndarray:
    """Boolean array over the TEST split: True where a labeled anomaly is present.
    anomaly_sequences windows are treated as [start, end] inclusive.
    """
    row = get_channel_row(chan_id)
    length = num_values if num_values is not None else int(row["num_values"])
    mask = np.zeros(length, dtype=bool)
    for start, end in row["anomaly_sequences"]:
        end_clamped = min(end, length - 1)
        if start > end_clamped:
            continue
        mask[start : end_clamped + 1] = True
    return mask


def get_anomaly_windows(chan_id: str) -> list[dict]:
    """List of {start, end, class} ground-truth windows, for chart overlays."""
    row = get_channel_row(chan_id)
    sequences = row["anomaly_sequences"]
    classes = row["class"]
    return [
        {"start": int(s), "end": int(e), "anomaly_class": c}
        for (s, e), c in zip(sequences, classes)
    ]
