"""Sliding-window generation, train-split normalization, and a rolling buffer
for streaming (online replay) inference. Shared by training and the backend.
"""

from __future__ import annotations

import numpy as np


class ChannelScaler:
    """Z-score scaler fit on the train-split value column (column 0) only."""

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std if std > 1e-8 else 1.0

    def transform_value(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform_value(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": float(self.mean), "std": float(self.std)}

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelScaler":
        return cls(d["mean"], d["std"])

    @classmethod
    def fit(cls, train_arr: np.ndarray) -> "ChannelScaler":
        values = train_arr[:, 0]
        return cls(mean=float(values.mean()), std=float(values.std()))


def normalize_channel(arr: np.ndarray, scaler: ChannelScaler) -> np.ndarray:
    """Copy of arr with column 0 (value) normalized; context columns untouched."""
    out = arr.astype(np.float64, copy=True)
    out[:, 0] = scaler.transform_value(arr[:, 0])
    return out


def make_windows(arr: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """Dense sliding windows over a (T, F) array -> (N, window_size, F)."""
    t = arr.shape[0]
    if t < window_size:
        raise ValueError(f"series length {t} shorter than window_size {window_size}")
    n = (t - window_size) // stride + 1
    windows = np.empty((n, window_size, arr.shape[1]), dtype=arr.dtype)
    for i in range(n):
        start = i * stride
        windows[i] = arr[start : start + window_size]
    return windows


def make_forecast_windows(
    arr: np.ndarray, window_size: int, horizon: int, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding (input window, next-H value) pairs for forecaster training.

    Returns X (N, window_size, F) and y (N, horizon): y is the next `horizon`
    raw values (column 0) immediately following each window.
    """
    t = arr.shape[0]
    needed = window_size + horizon
    if t < needed:
        raise ValueError(f"series length {t} shorter than window_size+horizon {needed}")
    n = (t - needed) // stride + 1
    x = np.empty((n, window_size, arr.shape[1]), dtype=arr.dtype)
    y = np.empty((n, horizon), dtype=arr.dtype)
    for i in range(n):
        start = i * stride
        x[i] = arr[start : start + window_size]
        y[i] = arr[start + window_size : start + window_size + horizon, 0]
    return x, y


class RollingWindow:
    """Maintains the last `size` points of a (num_features,) stream, for streaming inference."""

    def __init__(self, size: int, num_features: int):
        self.size = size
        self.num_features = num_features
        self._buf = np.zeros((size, num_features), dtype=np.float64)
        self._count = 0

    def push(self, point: np.ndarray) -> None:
        self._buf = np.roll(self._buf, -1, axis=0)
        self._buf[-1] = point
        self._count = min(self._count + 1, self.size)

    @property
    def is_full(self) -> bool:
        return self._count >= self.size

    def get(self) -> np.ndarray:
        return self._buf.copy()
