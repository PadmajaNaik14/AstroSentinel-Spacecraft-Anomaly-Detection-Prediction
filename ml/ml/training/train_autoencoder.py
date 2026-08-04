"""Trains the LSTM Autoencoder for a single channel on its train split only
(assumed-normal data, per the Telemanom convention -- the model never sees
test-split anomalies during training).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.config import WINDOW_SIZE
from ml.data.windowing import ChannelScaler, make_windows, normalize_channel
from ml.models.lstm_autoencoder import LSTMAutoencoder


def train_autoencoder(
    train_arr: np.ndarray,
    window_size: int = WINDOW_SIZE,
    epochs: int = 20,
    patience: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_fraction: float = 0.15,
    seed: int = 42,
    stride: int = 3,
) -> tuple[LSTMAutoencoder, ChannelScaler, dict]:
    torch.manual_seed(seed)
    scaler = ChannelScaler.fit(train_arr)
    normalized = normalize_channel(train_arr, scaler)
    windows = make_windows(normalized, window_size, stride=stride)

    # SMAP channels are 25-dim, MSL channels are 55-dim -- infer per channel
    # rather than assuming a fixed width.
    num_features = train_arr.shape[1]

    n = len(windows)
    n_val = max(1, int(n * val_fraction))
    train_windows = windows[: n - n_val]
    val_windows = windows[n - n_val :]

    train_tensor = torch.tensor(train_windows, dtype=torch.float32)
    val_tensor = torch.tensor(val_windows, dtype=torch.float32)

    model = LSTMAutoencoder(num_features=num_features, window_size=window_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    patience_left = patience
    history: dict = {"train_loss": [], "val_loss": []}

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            loss = model.reconstruction_error(batch).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(train_tensor)

        model.eval()
        with torch.no_grad():
            val_loss = model.reconstruction_error(val_tensor).mean().item()

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

    return model, scaler, history
