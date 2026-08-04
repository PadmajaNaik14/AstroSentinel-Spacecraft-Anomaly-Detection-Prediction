"""Vectorized (fast) scoring of an entire test split, for REST/historical
endpoints. Computes the AE error series and Isolation Forest features in
batched calls instead of looping the live score_point() function per
timestep -- that function is designed for single-point streaming and would
be far too slow called thousands of times in a request. Forecast is a
live-only concept here (surfaced over the WebSocket), so historical points
carry forecast=None.
"""

import numpy as np

from ml.evaluation.evaluate import compute_error_series
from ml.models.ensemble import ensemble_score, get_health_state, is_anomaly
from ml.models.isolation_forest import build_engineered_features
from ml.thresholding import ewma_smooth

from app.services.model_registry import ChannelModel
from app.services.replay_engine import get_ground_truth_mask, get_test_data

_CACHE: dict[str, dict] = {}


def _compute_full_series(channel_model: ChannelModel) -> dict:
    chan_id = channel_model.chan_id
    if chan_id in _CACHE:
        return _CACHE[chan_id]

    test_arr = get_test_data(chan_id)
    gt_mask = get_ground_truth_mask(chan_id)

    raw_errors = compute_error_series(channel_model.ae_model, channel_model.scaler, test_arr)
    smoothed_errors = ewma_smooth(raw_errors)

    features = build_engineered_features(
        smoothed_errors, channel_model.threshold["ae_mu"], channel_model.threshold["ae_sigma"]
    )
    if_flags = channel_model.if_detector.predict_is_anomaly(features)
    if_scores = channel_model.if_detector.decision_score(features)

    ae_threshold = channel_model.threshold["ae_threshold"]
    if_threshold = channel_model.threshold["if_threshold"]

    n = len(smoothed_errors)
    scores = np.empty(n)
    anomalies = np.empty(n, dtype=bool)
    health_states: list[str] = [""] * n
    recent_scores: list[float] = []
    for i in range(n):
        score = ensemble_score(smoothed_errors[i], ae_threshold, if_scores[i], if_threshold)
        scores[i] = score
        anomaly_flag = is_anomaly(smoothed_errors[i], ae_threshold, bool(if_flags[i]))
        anomalies[i] = anomaly_flag
        health_states[i] = get_health_state(
            score, anomaly_flag=anomaly_flag, recent_scores=recent_scores
        )
        recent_scores.append(score)
        if len(recent_scores) > 60:
            recent_scores.pop(0)

    result = {
        "values": test_arr[:, 0],
        "smoothed_errors": smoothed_errors,
        "scores": scores,
        "anomalies": anomalies,
        "health_states": health_states,
        "ground_truth": gt_mask,
    }
    _CACHE[chan_id] = result
    return result


def get_series_slice(channel_model: ChannelModel, start: int, end: int) -> list[dict]:
    """Inclusive [start, end] slice of precomputed per-timestep results."""
    series = _compute_full_series(channel_model)
    points = []
    for idx in range(start, end + 1):
        points.append(
            {
                "index": idx,
                "value": float(series["values"][idx]),
                "reconstruction_error": float(series["smoothed_errors"][idx]),
                "ensemble_score": float(series["scores"][idx]),
                "is_anomaly": bool(series["anomalies"][idx]),
                "health_state": series["health_states"][idx],
                "ground_truth_anomaly": bool(series["ground_truth"][idx]),
                "forecast": None,
                "predicted_steps_to_anomaly": None,
            }
        )
    return points
