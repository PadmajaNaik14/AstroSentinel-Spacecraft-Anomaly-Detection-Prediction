"""Trains the GRU multi-step forecaster for a single channel on its train
split, reusing the same scaler the LSTM Autoencoder was fit with so the two
models' outputs stay directly comparable.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.config import FORECAST_HORIZON, WINDOW_SIZE
from ml.data.windowing import ChannelScaler, make_forecast_windows, normalize_channel
from ml.models.gru_forecaster import GRUForecaster


def train_forecaster(
    train_arr: np.ndarray,
    scaler: ChannelScaler,
    window_size: int = WINDOW_SIZE,
    horizon: int = FORECAST_HORIZON,
    epochs: int = 20,
    patience: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_fraction: float = 0.15,
    seed: int = 42,
    stride: int = 3,
) -> tuple[GRUForecaster, dict]:
    torch.manual_seed(seed)
    normalized = normalize_channel(train_arr, scaler)
    x, y = make_forecast_windows(normalized, window_size, horizon, stride=stride)
    num_features = train_arr.shape[1]

    n = len(x)
    n_val = max(1, int(n * val_fraction))
    x_train, y_train = x[: n - n_val], y[: n - n_val]
    x_val, y_val = x[n - n_val :], y[n - n_val :]

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    model = GRUForecaster(num_features=num_features, horizon=horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )

    best_val_loss = float("inf")
    best_state = None
    patience_left = patience
    history: dict = {"train_loss": [], "val_loss": []}

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(x_train_t)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val_t), y_val_t).item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
