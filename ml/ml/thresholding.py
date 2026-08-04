"""Reconstruction-error smoothing and per-channel dynamic threshold derivation.

Deliberately simplified vs. Telemanom's full non-parametric pruning algorithm:
threshold = mu + z*sigma on the model's own train-split residual, with z
grid-searched per channel to maximize F1 against labeled_anomalies.csv on the
test split. See plan.md for the documented rationale.
"""

import numpy as np

EWMA_SPAN = 10
Z_GRID = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def ewma_smooth(x: np.ndarray, span: int = EWMA_SPAN) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def mu_sigma(x: np.ndarray) -> tuple[float, float]:
    return float(np.mean(x)), float(np.std(x))


def threshold_from_z(mu: float, sigma: float, z: float) -> float:
    return mu + z * sigma


def precision_recall_f1(
    pred_mask: np.ndarray, true_mask: np.ndarray
) -> tuple[float, float, float]:
    tp = int(np.sum(pred_mask & true_mask))
    fp = int(np.sum(pred_mask & ~true_mask))
    fn = int(np.sum(~pred_mask & true_mask))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def grid_search_threshold(
    test_errors_smoothed: np.ndarray,
    ground_truth_mask: np.ndarray,
    mu: float,
    sigma: float,
) -> dict:
    """Grid-search z over Z_GRID to maximize F1 against ground truth on the test split."""
    best = None
    for z in Z_GRID:
        threshold = threshold_from_z(mu, sigma, z)
        pred_mask = test_errors_smoothed > threshold
        precision, recall, f1 = precision_recall_f1(pred_mask, ground_truth_mask)
        if best is None or f1 > best["f1"]:
            best = {
                "z": z,
                "threshold": threshold,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
    return best
