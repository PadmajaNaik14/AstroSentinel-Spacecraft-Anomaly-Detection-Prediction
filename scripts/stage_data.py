"""Stage the kagglehub-downloaded SMAP/MSL dataset into the repo's gitignored data/raw/.

Copies (not symlinks -- for Windows/Docker-mount portability) only what AstroSentinel
needs: labeled_anomalies.csv plus train/test .npy files for the curated channel subset.
This keeps data/raw/ small and avoids hardcoding an absolute host path anywhere that
docker-compose.yml or other collaborators would depend on.
"""

import shutil
import sys
from pathlib import Path

import kagglehub

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ml"))
from ml.config import CURATED_CHANNEL_IDS  # noqa: E402

DATA_RAW = REPO_ROOT / "data" / "raw"


def main() -> None:
    cache_path = Path(
        kagglehub.dataset_download("patrickfleith/nasa-anomaly-detection-dataset-smap-msl")
    )
    print(f"Source cache: {cache_path}")

    dest_train = DATA_RAW / "data" / "data" / "train"
    dest_test = DATA_RAW / "data" / "data" / "test"
    dest_train.mkdir(parents=True, exist_ok=True)
    dest_test.mkdir(parents=True, exist_ok=True)

    src_csv = cache_path / "labeled_anomalies.csv"
    shutil.copy2(src_csv, DATA_RAW / "labeled_anomalies.csv")
    print("Copied labeled_anomalies.csv")

    src_train = cache_path / "data" / "data" / "train"
    src_test = cache_path / "data" / "data" / "test"

    missing = []
    for chan_id in CURATED_CHANNEL_IDS:
        for src_dir, dest_dir in ((src_train, dest_train), (src_test, dest_test)):
            src_file = src_dir / f"{chan_id}.npy"
            if not src_file.exists():
                missing.append(str(src_file))
                continue
            shutil.copy2(src_file, dest_dir / f"{chan_id}.npy")

    if missing:
        raise FileNotFoundError(f"Missing expected channel files: {missing}")

    print(f"Staged {len(CURATED_CHANNEL_IDS)} channels (train + test) into {DATA_RAW}")


if __name__ == "__main__":
    main()
