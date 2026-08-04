"""Live, single-point scoring: pushes a new raw telemetry point through the
trained AE + Isolation Forest + GRU forecaster and returns the same
detection+prediction result shape used by both the WebSocket replay stream
and REST historical-telemetry endpoints. Uses ml.models.ensemble as the sole
source of truth for score combination / health-state rules -- never
reimplemented here.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from ml.config import FORECAST_HORIZON, WINDOW_SIZE
from ml.data.windowing import RollingWindow, normalize_channel
from ml.forecast_scoring import score_forecast
from ml.models.ensemble import ensemble_score, get_health_state, is_anomaly
from ml.models.isolation_forest import build_engineered_features
from ml.thresholding import EWMA_SPAN

from app.services.model_registry import ChannelModel

ROLL_WINDOW = 20
RECENT_SCORES_MAXLEN = 60


@dataclass
class ChannelStreamState:
    channel_model: ChannelModel
    raw_window: RollingWindow = field(init=False)
    smoothed_error_buffer: deque = field(init=False)
    recent_scores: deque = field(init=False)
    _last_smoothed: float | None = field(default=None, init=False)

    def __post_init__(self):
        num_features = self.channel_model.metadata["num_features"]
        self.raw_window = RollingWindow(WINDOW_SIZE, num_features)
        self.smoothed_error_buffer = deque(maxlen=ROLL_WINDOW)
        self.recent_scores = deque(maxlen=RECENT_SCORES_MAXLEN)


def _ewma_update(state: ChannelStreamState, raw_error: float) -> float:
    alpha = 2.0 / (EWMA_SPAN + 1.0)
    if state._last_smoothed is None:
        smoothed = raw_error
    else:
        smoothed = alpha * raw_error + (1 - alpha) * state._last_smoothed
    state._last_smoothed = smoothed
    return smoothed


def score_point(
    state: ChannelStreamState, raw_point: np.ndarray, index: int, ground_truth_anomaly: bool
) -> dict:
    cm = state.channel_model
    state.raw_window.push(raw_point)

    base = {
        "index": index,
        "value": float(raw_point[0]),
        "ground_truth_anomaly": ground_truth_anomaly,
    }

    if not state.raw_window.is_full:
        return {
            **base,
            "reconstruction_error": 0.0,
            "ensemble_score": 0.0,
            "is_anomaly": False,
            "health_state": "normal",
            "forecast": None,
            "predicted_steps_to_anomaly": None,
        }

    raw_window = state.raw_window.get()
    normalized = normalize_channel(raw_window, cm.scaler)

    with torch.no_grad():
        batch = torch.tensor(normalized[np.newaxis, ...], dtype=torch.float32)
        raw_error = cm.ae_model.reconstruction_error(batch).numpy()[0, -1]

    smoothed_error = _ewma_update(state, float(raw_error))
    state.smoothed_error_buffer.append(smoothed_error)

    error_buf = np.array(state.smoothed_error_buffer)
    features = build_engineered_features(
        error_buf, cm.threshold["ae_mu"], cm.threshold["ae_sigma"]
    )
    current_features = features[-1]

    if_flag = bool(cm.if_detector.predict_is_anomaly(current_features.reshape(1, -1))[0])
    if_score = float(cm.if_detector.decision_score(current_features.reshape(1, -1))[0])

    ae_threshold = cm.threshold["ae_threshold"]
    if_threshold = cm.threshold["if_threshold"]

    anomaly = is_anomaly(smoothed_error, ae_threshold, if_flag)
    score = ensemble_score(smoothed_error, ae_threshold, if_score, if_threshold)

    forecast_result = score_forecast(
        forecaster=cm.forecaster,
        ae_model=cm.ae_model,
        scaler=cm.scaler,
        recent_window_raw=raw_window,
        ae_threshold=ae_threshold,
        window_size=WINDOW_SIZE,
        horizon=FORECAST_HORIZON,
    )

    health_state = get_health_state(
        score,
        anomaly_flag=anomaly,
        recent_scores=list(state.recent_scores),
        predicted_steps_to_anomaly=forecast_result["predicted_steps_to_anomaly"],
    )
    state.recent_scores.append(score)

    return {
        **base,
        "reconstruction_error": float(smoothed_error),
        "ensemble_score": float(score),
        "is_anomaly": bool(anomaly),
        "health_state": health_state,
        "forecast": forecast_result["forecast_values"],
        "predicted_steps_to_anomaly": forecast_result["predicted_steps_to_anomaly"],
    }
