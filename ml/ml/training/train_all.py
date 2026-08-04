"""Full training pipeline entrypoint: trains the LSTM Autoencoder, GRU
forecaster, and Isolation Forest for every channel in CURATED_CHANNELS,
evaluates each, and writes model artifacts + manifest.json -- the contract
the backend's ModelRegistry loads at startup.

Usage: python -m ml.training.train_all
"""

import json
import time

import joblib
import torch

from ml.config import ARTIFACTS_DIR, CURATED_CHANNELS, FORECAST_HORIZON, WINDOW_SIZE
from ml.data.labels import build_ground_truth_mask
from ml.data.loader import load_train_test
from ml.evaluation.evaluate import compute_error_series, evaluate_detector, evaluate_forecaster
from ml.thresholding import ewma_smooth, grid_search_threshold, mu_sigma
from ml.training.train_autoencoder import train_autoencoder
from ml.training.train_forecaster import train_forecaster
from ml.training.train_isolation_forest import train_isolation_forest


def train_channel(chan_meta: dict) -> dict:
    chan_id = chan_meta["chan_id"]
    print(f"[{chan_id}] loading data...")
    train_arr, test_arr = load_train_test(chan_id)
    ground_truth_mask = build_ground_truth_mask(chan_id, num_values=test_arr.shape[0])

    print(f"[{chan_id}] training LSTM autoencoder...")
    ae_model, scaler, ae_history = train_autoencoder(train_arr)

    print(f"[{chan_id}] deriving threshold...")
    train_errors = compute_error_series(ae_model, scaler, train_arr)
    train_errors_smoothed = ewma_smooth(train_errors)
    train_mu, train_sigma = mu_sigma(train_errors_smoothed)

    test_errors = compute_error_series(ae_model, scaler, test_arr)
    test_errors_smoothed = ewma_smooth(test_errors)
    threshold_info = grid_search_threshold(
        test_errors_smoothed, ground_truth_mask, train_mu, train_sigma
    )

    print(f"[{chan_id}] training isolation forest...")
    if_detector, train_if_features = train_isolation_forest(
        train_errors_smoothed, train_mu, train_sigma
    )
    if_train_scores = if_detector.decision_score(train_if_features)
    if_mu, if_sigma = mu_sigma(if_train_scores)
    if_threshold = if_mu + 3.0 * if_sigma

    print(f"[{chan_id}] training GRU forecaster...")
    forecaster, forecaster_history = train_forecaster(train_arr, scaler)
    forecast_eval = evaluate_forecaster(forecaster, scaler, test_arr)

    detector_eval = evaluate_detector(
        ae_model, scaler, threshold_info["threshold"], test_arr, ground_truth_mask
    )

    # -- persist artifacts --
    chan_dir = ARTIFACTS_DIR / chan_id
    chan_dir.mkdir(parents=True, exist_ok=True)

    torch.save(ae_model.state_dict(), chan_dir / "lstm_ae.pt")
    torch.save(forecaster.state_dict(), chan_dir / "gru_forecaster.pt")
    if_detector.save(chan_dir / "isolation_forest.joblib")
    joblib.dump(scaler, chan_dir / "scaler.joblib")

    threshold_payload = {
        "z": threshold_info["z"],
        "ae_mu": train_mu,
        "ae_sigma": train_sigma,
        "ae_threshold": threshold_info["threshold"],
        "if_mu": if_mu,
        "if_sigma": if_sigma,
        "if_threshold": if_threshold,
    }
    (chan_dir / "threshold.json").write_text(json.dumps(threshold_payload, indent=2))

    metadata = {
        "chan_id": chan_id,
        "spacecraft": chan_meta["spacecraft"],
        "anomaly_class": chan_meta["anomaly_class"],
        "num_features": int(train_arr.shape[1]),
        "window_size": WINDOW_SIZE,
        "forecast_horizon": FORECAST_HORIZON,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_length": int(train_arr.shape[0]),
        "test_length": int(test_arr.shape[0]),
        "ae_final_train_loss": ae_history["train_loss"][-1],
        "ae_final_val_loss": ae_history["val_loss"][-1],
        "forecaster_final_train_loss": forecaster_history["train_loss"][-1],
        "forecaster_final_val_loss": forecaster_history["val_loss"][-1],
        "eval_precision": detector_eval["precision"],
        "eval_recall": detector_eval["recall"],
        "eval_f1": detector_eval["f1"],
        "forecast_mae": forecast_eval["forecast_mae"],
    }
    (chan_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(
        f"[{chan_id}] done. F1={detector_eval['f1']:.3f} "
        f"P={detector_eval['precision']:.3f} R={detector_eval['recall']:.3f} "
        f"forecast_MAE={forecast_eval['forecast_mae']}"
    )
    return metadata


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "channels": {}}

    for chan_meta in CURATED_CHANNELS:
        metadata = train_channel(chan_meta)
        manifest["channels"][chan_meta["chan_id"]] = {
            "dir": chan_meta["chan_id"],
            "spacecraft": chan_meta["spacecraft"],
            "anomaly_class": chan_meta["anomaly_class"],
            "eval_f1": metadata["eval_f1"],
        }

    (ARTIFACTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"\nWrote manifest for {len(manifest['channels'])} channels to "
        f"{ARTIFACTS_DIR / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
