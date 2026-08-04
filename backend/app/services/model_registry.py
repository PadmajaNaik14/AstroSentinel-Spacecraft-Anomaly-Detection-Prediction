"""Loads all trained model artifacts once at startup into an in-memory
registry (`app.state.model_registry`). Fails fast if artifacts are missing --
forces the training pre-step to actually happen before the app serves traffic.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import torch

from ml.data.windowing import ChannelScaler
from ml.models.gru_forecaster import GRUForecaster
from ml.models.isolation_forest import IsolationForestDetector
from ml.models.lstm_autoencoder import LSTMAutoencoder


@dataclass
class ChannelModel:
    chan_id: str
    spacecraft: str
    anomaly_class: str
    ae_model: LSTMAutoencoder
    forecaster: GRUForecaster
    if_detector: IsolationForestDetector
    scaler: ChannelScaler
    threshold: dict
    metadata: dict


class ModelRegistry:
    def __init__(self, channels: dict[str, ChannelModel]):
        self.channels = channels

    def get(self, chan_id: str) -> ChannelModel:
        if chan_id not in self.channels:
            raise KeyError(f"unknown channel {chan_id}")
        return self.channels[chan_id]

    def list_channel_ids(self) -> list[str]:
        return list(self.channels.keys())

    @classmethod
    def load(cls, artifacts_dir: Path) -> "ModelRegistry":
        manifest_path = artifacts_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"No trained artifacts found at {artifacts_dir} -- run the training "
                "pipeline first (`docker compose --profile train run --rm trainer` "
                "or `python -m ml.training.train_all`)."
            )
        manifest = json.loads(manifest_path.read_text())

        channels: dict[str, ChannelModel] = {}
        for chan_id, chan_info in manifest["channels"].items():
            chan_dir = artifacts_dir / chan_info["dir"]

            # Read metadata first -- num_features varies by spacecraft (SMAP
            # channels are 25-dim, MSL channels are 55-dim), so the model
            # shape must be reconstructed per channel, not assumed fixed.
            metadata = json.loads((chan_dir / "metadata.json").read_text())
            threshold = json.loads((chan_dir / "threshold.json").read_text())
            num_features = metadata["num_features"]

            ae_model = LSTMAutoencoder(
                num_features=num_features, window_size=metadata["window_size"]
            )
            ae_model.load_state_dict(torch.load(chan_dir / "lstm_ae.pt", map_location="cpu"))
            ae_model.eval()

            forecaster = GRUForecaster(
                num_features=num_features, horizon=metadata["forecast_horizon"]
            )
            forecaster.load_state_dict(
                torch.load(chan_dir / "gru_forecaster.pt", map_location="cpu")
            )
            forecaster.eval()

            if_detector = IsolationForestDetector.load(chan_dir / "isolation_forest.joblib")
            scaler = joblib.load(chan_dir / "scaler.joblib")

            channels[chan_id] = ChannelModel(
                chan_id=chan_id,
                spacecraft=chan_info["spacecraft"],
                anomaly_class=chan_info["anomaly_class"],
                ae_model=ae_model,
                forecaster=forecaster,
                if_detector=if_detector,
                scaler=scaler,
                threshold=threshold,
                metadata=metadata,
            )

        return cls(channels)
