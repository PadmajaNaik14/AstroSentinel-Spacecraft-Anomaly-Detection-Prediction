"""Raw channel loading. Column 0 is the real telemetry value; columns 1-24
are one-hot spacecraft command/event context (constant 0 in the train split).
"""

import numpy as np

from ml.config import TEST_DIR, TRAIN_DIR


def load_channel(chan_id: str, split: str) -> np.ndarray:
    """Load a channel's (T, 25) float64 array for split in {"train", "test"}."""
    base = TRAIN_DIR if split == "train" else TEST_DIR
    path = base / f"{chan_id}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/stage_data.py first"
        )
    return np.load(path)


def load_train_test(chan_id: str) -> tuple[np.ndarray, np.ndarray]:
    return load_channel(chan_id, "train"), load_channel(chan_id, "test")
