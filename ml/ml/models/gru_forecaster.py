"""GRU multi-step forecaster: the failure-prediction module.

Given a window of telemetry, predicts the next `horizon` raw value-channel
steps directly (multi-output regression, not autoregressive -- avoids
compounding rollout error). Its forecast is later scored through the
already-trained LSTM Autoencoder (see ml.models.ensemble) to derive a
predicted-time-to-anomaly, so detection and prediction stay consistent.
"""

import torch
import torch.nn as nn

from ml.config import FORECAST_HORIZON, NUM_FEATURES


class GRUForecaster(nn.Module):
    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        horizon: int = FORECAST_HORIZON,
    ):
        super().__init__()
        self.horizon = horizon

        self.gru1 = nn.GRU(num_features, 64, batch_first=True)
        self.gru2 = nn.GRU(64, 32, batch_first=True)
        self.head = nn.Linear(32, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_size, num_features) -> (batch, horizon) forecast."""
        out, _ = self.gru1(x)
        _, h_n = self.gru2(out)
        latent = h_n[-1]  # (batch, 32)
        return self.head(latent)
