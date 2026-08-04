"""Central configuration: paths and the curated channel subset.

CURATED_CHANNELS is the single source of truth for which SMAP/MSL channels
AstroSentinel trains and serves. Extending coverage later is appending here
and re-running `python -m ml.training.train_all` -- nothing else changes.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Overridable via env vars so the same code resolves correctly both in local
# dev (paths computed relative to the repo) and inside Docker containers
# (paths pointing at mounted volumes).
DATA_RAW_DIR = Path(os.environ.get("ASTROSENTINEL_DATA_RAW_DIR", REPO_ROOT / "data" / "raw"))
DATA_PROCESSED_DIR = Path(
    os.environ.get("ASTROSENTINEL_DATA_PROCESSED_DIR", REPO_ROOT / "data" / "processed")
)
ARTIFACTS_DIR = Path(
    os.environ.get("ASTROSENTINEL_ARTIFACTS_DIR", REPO_ROOT / "ml" / "artifacts")
)

LABELED_ANOMALIES_CSV = DATA_RAW_DIR / "labeled_anomalies.csv"
TRAIN_DIR = DATA_RAW_DIR / "data" / "data" / "train"
TEST_DIR = DATA_RAW_DIR / "data" / "data" / "test"

WINDOW_SIZE = 100
FORECAST_HORIZON = 15
NUM_FEATURES = 25  # column 0 = value, columns 1-24 = one-hot context

# Verified against the real labeled_anomalies.csv (see plan.md).
# 10 SMAP + 10 MSL, mixed point / contextual / mixed anomaly classes.
CURATED_CHANNELS = [
    {"chan_id": "S-1", "spacecraft": "SMAP", "anomaly_class": "point"},
    {"chan_id": "E-2", "spacecraft": "SMAP", "anomaly_class": "point"},
    {"chan_id": "A-1", "spacecraft": "SMAP", "anomaly_class": "point"},
    {"chan_id": "D-1", "spacecraft": "SMAP", "anomaly_class": "point"},
    {"chan_id": "P-1", "spacecraft": "SMAP", "anomaly_class": "contextual"},
    {"chan_id": "A-2", "spacecraft": "SMAP", "anomaly_class": "contextual"},
    {"chan_id": "G-1", "spacecraft": "SMAP", "anomaly_class": "contextual"},
    {"chan_id": "T-1", "spacecraft": "SMAP", "anomaly_class": "mixed"},
    {"chan_id": "P-4", "spacecraft": "SMAP", "anomaly_class": "point"},
    {"chan_id": "B-1", "spacecraft": "SMAP", "anomaly_class": "point"},
    {"chan_id": "M-6", "spacecraft": "MSL", "anomaly_class": "point"},
    {"chan_id": "S-2", "spacecraft": "MSL", "anomaly_class": "point"},
    {"chan_id": "T-4", "spacecraft": "MSL", "anomaly_class": "point"},
    {"chan_id": "F-4", "spacecraft": "MSL", "anomaly_class": "point"},
    {"chan_id": "P-11", "spacecraft": "MSL", "anomaly_class": "point"},
    {"chan_id": "M-1", "spacecraft": "MSL", "anomaly_class": "contextual"},
    {"chan_id": "F-7", "spacecraft": "MSL", "anomaly_class": "contextual"},
    {"chan_id": "T-12", "spacecraft": "MSL", "anomaly_class": "contextual"},
    {"chan_id": "D-16", "spacecraft": "MSL", "anomaly_class": "contextual"},
    {"chan_id": "C-1", "spacecraft": "MSL", "anomaly_class": "mixed"},
]

CURATED_CHANNEL_IDS = [c["chan_id"] for c in CURATED_CHANNELS]
