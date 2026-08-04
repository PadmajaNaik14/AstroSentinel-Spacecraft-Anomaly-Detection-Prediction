"""Test-split caching and the per-connection async replay generator used by
the WebSocket endpoint. Each connection gets its own independent
ChannelStreamState and its own position in the series -- no cross-client
synchronization, which is the right tradeoff at this project's scale.
"""

import asyncio
from collections.abc import AsyncGenerator

import numpy as np

from ml.data.labels import build_ground_truth_mask
from ml.data.loader import load_channel

from app.services.inference import ChannelStreamState, score_point
from app.services.model_registry import ChannelModel

_TEST_DATA_CACHE: dict[str, np.ndarray] = {}
_GT_MASK_CACHE: dict[str, np.ndarray] = {}


def get_test_data(chan_id: str) -> np.ndarray:
    if chan_id not in _TEST_DATA_CACHE:
        _TEST_DATA_CACHE[chan_id] = load_channel(chan_id, "test")
    return _TEST_DATA_CACHE[chan_id]


def get_ground_truth_mask(chan_id: str) -> np.ndarray:
    if chan_id not in _GT_MASK_CACHE:
        arr = get_test_data(chan_id)
        _GT_MASK_CACHE[chan_id] = build_ground_truth_mask(chan_id, num_values=arr.shape[0])
    return _GT_MASK_CACHE[chan_id]


def get_window_ending_at(chan_id: str, end_index: int, window_size: int) -> np.ndarray:
    """Raw (window_size, F) window of test data ending at end_index (inclusive)."""
    arr = get_test_data(chan_id)
    start = max(0, end_index - window_size + 1)
    window = arr[start : end_index + 1]
    if len(window) < window_size:
        pad = np.repeat(window[:1], window_size - len(window), axis=0)
        window = np.vstack([pad, window])
    return window


async def stream(
    channel_model: ChannelModel, tick_seconds: float
) -> AsyncGenerator[dict, None]:
    """Yields one scored point at a time, looping back to index 0 at the end."""
    test_arr = get_test_data(channel_model.chan_id)
    gt_mask = get_ground_truth_mask(channel_model.chan_id)
    state = ChannelStreamState(channel_model)

    n = test_arr.shape[0]
    idx = 0
    while True:
        point = test_arr[idx]
        result = score_point(state, point, index=idx, ground_truth_anomaly=bool(gt_mask[idx]))
        yield result
        idx = (idx + 1) % n
        await asyncio.sleep(tick_seconds)
