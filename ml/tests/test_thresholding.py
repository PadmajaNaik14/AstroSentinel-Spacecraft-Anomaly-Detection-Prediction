import numpy as np

from ml.thresholding import (
    ewma_smooth,
    grid_search_threshold,
    mu_sigma,
    precision_recall_f1,
    threshold_from_z,
)


def test_ewma_smooth_constant_series():
    x = np.full(20, 5.0)
    smoothed = ewma_smooth(x)
    np.testing.assert_allclose(smoothed, 5.0)


def test_ewma_smooth_reduces_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    smoothed = ewma_smooth(x)
    assert smoothed.var() < x.var()


def test_mu_sigma():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mu, sigma = mu_sigma(x)
    assert mu == 3.0
    assert sigma > 0


def test_threshold_from_z():
    assert threshold_from_z(mu=10.0, sigma=2.0, z=3.0) == 16.0


def test_precision_recall_f1_perfect():
    pred = np.array([True, True, False, False])
    true = np.array([True, True, False, False])
    p, r, f1 = precision_recall_f1(pred, true)
    assert p == 1.0 and r == 1.0 and f1 == 1.0


def test_precision_recall_f1_no_predictions():
    pred = np.array([False, False, False])
    true = np.array([True, False, False])
    p, r, f1 = precision_recall_f1(pred, true)
    assert p == 0.0 and r == 0.0 and f1 == 0.0


def test_grid_search_threshold_picks_best_f1():
    # errors spike exactly where the ground truth anomaly window is
    errors = np.concatenate([np.full(50, 1.0), np.full(10, 20.0), np.full(40, 1.0)])
    ground_truth = np.zeros(100, dtype=bool)
    ground_truth[50:60] = True

    mu, sigma = 1.0, 0.5
    result = grid_search_threshold(errors, ground_truth, mu, sigma)
    assert result["f1"] > 0.9
    pred_mask = errors > result["threshold"]
    np.testing.assert_array_equal(pred_mask, ground_truth)
