"""On-demand SHAP explanation (TreeExplainer on Isolation Forest features) +
a separate, explicitly-labeled reconstruction-error-per-timestep heatmap for
a given channel/index. Computed on request, not precomputed for every
timestep -- TreeExplainer on one sample is sub-millisecond.
"""

import numpy as np
import torch

from ml.config import WINDOW_SIZE
from ml.data.windowing import normalize_channel
from ml.evaluation.evaluate import compute_error_series
from ml.explain.shap_explainer import ShapExplainer
from ml.models.isolation_forest import build_engineered_features
from ml.thresholding import ewma_smooth

from app.services.model_registry import ChannelModel
from app.services.replay_engine import get_test_data

_ERROR_SERIES_CACHE: dict[str, np.ndarray] = {}
_EXPLAINER_CACHE: dict[str, ShapExplainer] = {}


def _get_smoothed_error_series(channel_model: ChannelModel) -> np.ndarray:
    chan_id = channel_model.chan_id
    if chan_id not in _ERROR_SERIES_CACHE:
        test_arr = get_test_data(chan_id)
        errors = compute_error_series(channel_model.ae_model, channel_model.scaler, test_arr)
        _ERROR_SERIES_CACHE[chan_id] = ewma_smooth(errors)
    return _ERROR_SERIES_CACHE[chan_id]


def _get_explainer(channel_model: ChannelModel) -> ShapExplainer:
    chan_id = channel_model.chan_id
    if chan_id not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[chan_id] = ShapExplainer(channel_model.if_detector.model)
    return _EXPLAINER_CACHE[chan_id]


def explain_index(channel_model: ChannelModel, index: int) -> dict:
    smoothed_errors = _get_smoothed_error_series(channel_model)
    index = max(0, min(index, len(smoothed_errors) - 1))

    error_window = smoothed_errors[: index + 1][-20:]
    features = build_engineered_features(
        error_window, channel_model.threshold["ae_mu"], channel_model.threshold["ae_sigma"]
    )
    current_features = features[-1]

    shap_result = _get_explainer(channel_model).explain(current_features)

    # Reconstruction-error-per-timestep heatmap for the WINDOW_SIZE-step
    # window ending at `index` -- explicitly NOT SHAP, see module docstring.
    test_arr = get_test_data(channel_model.chan_id)
    win_start = max(0, index - WINDOW_SIZE + 1)
    raw_window = test_arr[win_start : index + 1]
    if len(raw_window) < WINDOW_SIZE:
        pad = np.repeat(raw_window[:1], WINDOW_SIZE - len(raw_window), axis=0)
        raw_window = np.vstack([pad, raw_window])
    normalized = normalize_channel(raw_window, channel_model.scaler)

    channel_model.ae_model.eval()
    with torch.no_grad():
        batch = torch.tensor(normalized[np.newaxis, ...], dtype=torch.float32)
        per_step_error = channel_model.ae_model.reconstruction_error(batch).numpy()[0]

    max_err = float(per_step_error.max()) or 1.0
    heatmap = (per_step_error / max_err).tolist()

    return {
        "feature_names": shap_result["feature_names"],
        "shap_values": shap_result["shap_values"],
        "feature_values": shap_result["feature_values"],
        "reconstruction_error_heatmap": heatmap,
    }
