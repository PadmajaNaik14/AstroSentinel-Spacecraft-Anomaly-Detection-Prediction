"""Trains the Isolation Forest secondary detector on engineered rolling-error
features derived from the LSTM Autoencoder's train-split reconstruction error.
"""

import numpy as np

from ml.models.isolation_forest import IsolationForestDetector, build_engineered_features


def train_isolation_forest(
    train_errors_smoothed: np.ndarray, train_mu: float, train_sigma: float
) -> tuple[IsolationForestDetector, np.ndarray]:
    features = build_engineered_features(train_errors_smoothed, train_mu, train_sigma)
    detector = IsolationForestDetector().fit(features)
    return detector, features
