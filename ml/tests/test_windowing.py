import numpy as np

from ml.data.windowing import (
    ChannelScaler,
    RollingWindow,
    make_forecast_windows,
    make_windows,
    normalize_channel,
)


def test_make_windows_shape():
    arr = np.arange(50 * 3).reshape(50, 3).astype(np.float64)
    windows = make_windows(arr, window_size=10, stride=1)
    assert windows.shape == (41, 10, 3)
    np.testing.assert_array_equal(windows[0], arr[0:10])
    np.testing.assert_array_equal(windows[-1], arr[40:50])


def test_make_windows_stride():
    arr = np.arange(50 * 3).reshape(50, 3).astype(np.float64)
    windows = make_windows(arr, window_size=10, stride=5)
    assert windows.shape == (9, 10, 3)


def test_make_forecast_windows_shapes():
    arr = np.arange(60 * 3).reshape(60, 3).astype(np.float64)
    x, y = make_forecast_windows(arr, window_size=10, horizon=5, stride=1)
    assert x.shape == (46, 10, 3)
    assert y.shape == (46, 5)
    np.testing.assert_array_equal(y[0], arr[10:15, 0])


def test_channel_scaler_roundtrip():
    train = np.random.randn(100, 2) * 5 + 10
    scaler = ChannelScaler.fit(train)
    normalized = normalize_channel(train, scaler)
    assert abs(normalized[:, 0].mean()) < 1e-6
    restored = scaler.inverse_transform_value(normalized[:, 0])
    np.testing.assert_allclose(restored, train[:, 0], atol=1e-6)


def test_channel_scaler_serialization():
    scaler = ChannelScaler(mean=3.0, std=2.0)
    d = scaler.to_dict()
    restored = ChannelScaler.from_dict(d)
    assert restored.mean == 3.0
    assert restored.std == 2.0


def test_rolling_window_push_and_fullness():
    rw = RollingWindow(size=3, num_features=2)
    assert not rw.is_full
    rw.push(np.array([1.0, 1.0]))
    rw.push(np.array([2.0, 2.0]))
    assert not rw.is_full
    rw.push(np.array([3.0, 3.0]))
    assert rw.is_full
    np.testing.assert_array_equal(rw.get()[-1], [3.0, 3.0])
    rw.push(np.array([4.0, 4.0]))
    np.testing.assert_array_equal(rw.get()[-1], [4.0, 4.0])
    np.testing.assert_array_equal(rw.get()[0], [2.0, 2.0])
