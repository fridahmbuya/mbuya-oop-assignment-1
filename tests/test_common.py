"""Tests for the shared forecasting utilities."""
import numpy as np
import pytest

from src.common.forecasting import (
    LinearTrendForecaster,
    bootstrap_prediction_intervals,
    mae,
    mape,
    rmse,
    walk_forward_predictions,
)


def test_metrics_match_hand_calculation():
    actual = [10, 20, 30]
    predicted = [12, 18, 33]  # errors: -2, 2, -3
    assert mae(actual, predicted) == pytest.approx(7 / 3)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert mape(actual, predicted) == pytest.approx((0.2 + 0.1 + 0.1) / 3 * 100)


def test_metrics_reject_empty_and_mismatched_input():
    with pytest.raises(ValueError):
        mae([], [])
    with pytest.raises(ValueError):
        rmse([1, 2], [1])


def test_mape_rejects_zero_actuals():
    with pytest.raises(ValueError):
        mape([0, 1], [1, 1])


def test_linear_trend_recovers_exact_line():
    y = 5 + 3 * np.arange(6)
    model = LinearTrendForecaster().fit(y)
    np.testing.assert_allclose(model.predict(2), [23, 26])
    np.testing.assert_allclose(model.residuals(), 0, atol=1e-9)


def test_predict_before_fit_and_bad_horizon():
    model = LinearTrendForecaster()
    with pytest.raises(RuntimeError):
        model.predict(1)
    model.fit([1, 2, 3])
    with pytest.raises(ValueError):
        model.predict(0)


def test_walk_forward_never_uses_the_future():
    y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    preds = walk_forward_predictions(LinearTrendForecaster, y, first_index=2)
    # Predictions for index 2, 3 come from perfect lines; the outlier at index 4 is unseen.
    np.testing.assert_allclose(preds, [3.0, 4.0, 5.0])


def test_bootstrap_interval_contains_point_forecast():
    rng = np.random.default_rng(0)
    y = 100 + 2 * np.arange(12) + rng.normal(0, 1, 12)
    point, lo, hi = bootstrap_prediction_intervals(LinearTrendForecaster, y, 3, n_boot=300)
    assert np.all(lo <= point) and np.all(point <= hi)
