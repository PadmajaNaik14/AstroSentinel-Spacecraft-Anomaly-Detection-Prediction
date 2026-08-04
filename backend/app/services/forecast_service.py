"""On-demand REST forecast: wraps ml.forecast_scoring for a specific
historical index, reusing the same window-construction helper the replay
engine uses so live and on-demand forecasts are computed identically.
"""

from ml.config import FORECAST_HORIZON, WINDOW_SIZE
from ml.forecast_scoring import score_forecast

from app.services.model_registry import ChannelModel
from app.services.replay_engine import get_test_data, get_window_ending_at


def get_forecast(channel_model: ChannelModel, index: int | None = None) -> dict:
    test_arr = get_test_data(channel_model.chan_id)
    if index is None:
        index = test_arr.shape[0] - 1
    index = max(0, min(index, test_arr.shape[0] - 1))

    window = get_window_ending_at(channel_model.chan_id, index, WINDOW_SIZE)
    result = score_forecast(
        forecaster=channel_model.forecaster,
        ae_model=channel_model.ae_model,
        scaler=channel_model.scaler,
        recent_window_raw=window,
        ae_threshold=channel_model.threshold["ae_threshold"],
        window_size=WINDOW_SIZE,
        horizon=FORECAST_HORIZON,
    )
    return {"base_index": index, **result}
