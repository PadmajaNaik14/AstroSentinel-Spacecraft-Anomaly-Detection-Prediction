"""Isolation Forest: the secondary/ensemble anomaly detector.

Consumes 6 engineered rolling-statistics features built from the LSTM
Autoencoder's reconstruction-error stream (not raw telemetry) -- a small,
tabular feature set that is also what makes TreeExplainer-based SHAP
attribution tractable (see ml.explain.shap_explainer).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ENGINEERED_FEATURE_NAMES = [
    "error",
    "rolling_mean",
    "rolling_std",
    "rolling_max",
    "slope",
    "zscore_vs_train",
]


def build_engineered_features(
    errors: np.ndarray, train_mu: float, train_sigma: float, roll_window: int = 20
) -> np.ndarray:
    """errors: 1D per-timestep reconstruction-error array.

    Returns (T, 6) feature matrix aligned to `errors`. Rolling stats use
    min_periods=1 so every timestep (even before a full rolling window is
    available) gets a usable feature vector.
    """
    s = pd.Series(errors, dtype=np.float64)
    rolling = s.rolling(window=roll_window, min_periods=1)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std().fillna(0.0)
    rolling_max = rolling.max()
    slope = s.diff().fillna(0.0)
    sigma = train_sigma if train_sigma > 1e-8 else 1.0
    zscore = (s - train_mu) / sigma

    return np.stack(
        [
            s.to_numpy(),
            rolling_mean.to_numpy(),
            rolling_std.to_numpy(),
            rolling_max.to_numpy(),
            slope.to_numpy(),
            zscore.to_numpy(),
        ],
        axis=1,
    )


class IsolationForestDetector:
    def __init__(self, model: IsolationForest | None = None):
        self.model = model or IsolationForest(contamination="auto", random_state=42)

    def fit(self, features: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(features)
        return self

    def predict_is_anomaly(self, features: np.ndarray) -> np.ndarray:
        """True where IsolationForest flags an anomaly (raw label -1)."""
        return self.model.predict(features) == -1

    def decision_score(self, features: np.ndarray) -> np.ndarray:
        """Higher = more anomalous (sklearn's decision_function negated)."""
        return -self.model.decision_function(features)

    def save(self, path) -> None:
        import joblib

        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path) -> "IsolationForestDetector":
        import joblib

        return cls(model=joblib.load(path))
