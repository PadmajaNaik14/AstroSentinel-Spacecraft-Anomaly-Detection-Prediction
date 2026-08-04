"""Per-channel evaluation: reconstruction-error series over the full test
split, precision/recall/F1 against labeled_anomalies.csv, and forecast MAE.
"""

import numpy as np
import torch

from ml.config import FORECAST_HORIZON, WINDOW_SIZE
from ml.data.windowing import ChannelScaler, make_forecast_windows, make_windows, normalize_channel
from ml.evaluation.metrics import forecast_mae
from ml.models.gru_forecaster import GRUForecaster
from ml.models.lstm_autoencoder import LSTMAutoencoder
from ml.thresholding import ewma_smooth, precision_recall_f1


def compute_error_series(
    model: LSTMAutoencoder,
    scaler: ChannelScaler,
    raw_arr: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    """Per-timestep AE reconstruction error over `raw_arr` (T, NUM_FEATURES).

    Timesteps before the first full window (t < window_size - 1) are
    backfilled with the series' own mean -- a documented simplification for
    the burn-in period, so every timestep still gets a usable value.
    """
    normalized = normalize_channel(raw_arr, scaler)
    t = normalized.shape[0]
    errors = np.full(t, np.nan, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        windows = make_windows(normalized, window_size, stride=1)
        batch = torch.tensor(windows, dtype=torch.float32)
        recon_err = model.reconstruction_error(batch).numpy()  # (N, window_size)
        last_step_err = recon_err[:, -1]
        errors[window_size - 1 :] = last_step_err

    valid_mean = float(np.nanmean(errors)) if np.any(~np.isnan(errors)) else 0.0
    return np.nan_to_num(errors, nan=valid_mean)


def evaluate_detector(
    model: LSTMAutoencoder,
    scaler: ChannelScaler,
    threshold: float,
    test_arr: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> dict:
    errors = compute_error_series(model, scaler, test_arr)
    smoothed = ewma_smooth(errors)
    pred_mask = smoothed > threshold
    precision, recall, f1 = precision_recall_f1(pred_mask, ground_truth_mask)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "smoothed_errors": smoothed,
    }


def evaluate_forecaster(
    forecaster: GRUForecaster,
    scaler: ChannelScaler,
    test_arr: np.ndarray,
    window_size: int = WINDOW_SIZE,
    horizon: int = FORECAST_HORIZON,
) -> dict:
    normalized = normalize_channel(test_arr, scaler)
    try:
        x, y = make_forecast_windows(normalized, window_size, horizon, stride=1)
    except ValueError:
        return {"forecast_mae": None}

    forecaster.eval()
    with torch.no_grad():
        preds = forecaster(torch.tensor(x, dtype=torch.float32)).numpy()

    preds_orig = scaler.inverse_transform_value(preds)
    y_orig = scaler.inverse_transform_value(y)
    return {"forecast_mae": forecast_mae(y_orig, preds_orig)}
