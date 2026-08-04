"""Download the NASA SMAP/MSL anomaly detection dataset via kagglehub.

Idempotent: kagglehub returns the existing local cache path if already downloaded.
Run this before scripts/stage_data.py.
"""

import kagglehub


def main() -> str:
    path = kagglehub.dataset_download("patrickfleith/nasa-anomaly-detection-dataset-smap-msl")
    print(f"Dataset available at: {path}")
    return path


if __name__ == "__main__":
    main()
