"""LSTM Autoencoder: the primary anomaly detector.

Encoder consumes the full 25-feature window (value + one-hot context);
the decoder reconstructs only the value channel (column 0). High
reconstruction error on the value channel is the core anomaly signal.
"""

import torch
import torch.nn as nn

from ml.config import NUM_FEATURES, WINDOW_SIZE


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        window_size: int = WINDOW_SIZE,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.window_size = window_size
        self.latent_dim = latent_dim

        self.encoder_lstm1 = nn.LSTM(num_features, 64, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(64, latent_dim, batch_first=True)

        self.decoder_lstm1 = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(latent_dim, 64, batch_first=True)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_size, num_features) -> reconstruction (batch, window_size)."""
        enc_out, _ = self.encoder_lstm1(x)
        _, (h_n, _) = self.encoder_lstm2(enc_out)
        latent = h_n[-1]  # (batch, latent_dim)

        repeated = latent.unsqueeze(1).repeat(1, self.window_size, 1)
        dec_out, _ = self.decoder_lstm1(repeated)
        dec_out, _ = self.decoder_lstm2(dec_out)
        reconstruction = self.output_layer(dec_out)  # (batch, window_size, 1)
        return reconstruction.squeeze(-1)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-timestep squared error against the true value channel (column 0)."""
        recon = self.forward(x)
        target = x[:, :, 0]
        return (recon - target) ** 2
