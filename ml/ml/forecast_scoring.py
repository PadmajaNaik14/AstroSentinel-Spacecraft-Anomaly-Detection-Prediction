"""Turns a GRU forecast into a predicted-time-to-anomaly by re-scoring the
forecasted trajectory through the already-trained LSTM Autoencoder + its
existing threshold -- so detection and prediction share one definition of
"anomalous," instead of inventing a second one for forecasts.

Single source of truth: imported by both ml/ (offline evaluation) and the
backend's forecast service (live scoring). Do not duplicate this logic.
"""

import numpy as np
import torch

from ml.data.windowing import ChannelScaler, normalize_channel
from ml.models.gru_forecaster import GRUForecaster
from ml.models.lstm_autoencoder import LSTMAutoencoder


def score_forecast(
    forecaster: GRUForecaster,
    ae_model: LSTMAutoencoder,
    scaler: ChannelScaler,
    recent_window_raw: np.ndarray,
    ae_threshold: float,
    window_size: int,
    horizon: int,
) -> dict:
    """recent_window_raw: (window_size, NUM_FEATURES) most recent raw telemetry.

    Forecasts the next `horizon` raw values, then re-scores each forecasted
    step through the trained AE (context columns held at their last-seen
    value, since future commands are unknown) to find the first step (if
    any) whose single-window reconstruction error would cross `ae_threshold`.

    Note: this compares a per-window raw error against a threshold derived
    from EWMA-smoothed errors (see ml.thresholding) -- a documented
    approximation, since a forward-looking forecast has no real smoothing
    history to lean on.
    """
    normalized = normalize_channel(recent_window_raw, scaler)

    forecaster.eval()
    ae_model.eval()
    with torch.no_grad():
        x = torch.tensor(normalized[np.newaxis, ...], dtype=torch.float32)
        forecast_normalized = forecaster(x).numpy()[0]  # (horizon,)

    forecast_raw = scaler.inverse_transform_value(forecast_normalized)

    last_context = normalized[-1, 1:]  # hold context columns constant into the future
    forecast_rows = np.column_stack(
        [forecast_normalized, np.tile(last_context, (horizon, 1))]
    )
    extended = np.vstack([normalized, forecast_rows])  # (window_size + horizon, F)

    predicted_steps_to_anomaly = None
    with torch.no_grad():
        for h in range(1, horizon + 1):
            window = extended[h : h + window_size]
            batch = torch.tensor(window[np.newaxis, ...], dtype=torch.float32)
            err = ae_model.reconstruction_error(batch).numpy()[0, -1]
            if err > ae_threshold:
                predicted_steps_to_anomaly = h
                break

    return {
        "forecast_values": [float(v) for v in forecast_raw],
        "predicted_steps_to_anomaly": predicted_steps_to_anomaly,
    }
