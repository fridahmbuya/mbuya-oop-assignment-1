"""Tests for Mini-Project 5 (taxi routes, pricing and fleet)."""
import numpy as np
import pytest

from src.common.forecasting import LinearTrendForecaster
from src.taxi import (
    FleetPlanner,
    LinearMarket,
    MovingAverageForecaster,
    Route,
    SeasonalNaiveForecaster,
    SimpleExponentialSmoothing,
    backtest_mae,
    simulate_weekly_demand,
    tune_alpha,
)

NTINDA = [35, 40, 42, 50, 55, 60, 48, 52, 47, 45]


# --- normal cases -----------------------------------------------------------
def test_route_revenue_and_stats():
    r = Route("Kampala-Ntinda", NTINDA, fare=2000)
    assert r.total_revenue() == sum(NTINDA) * 2000  # 474 * 2000 = 948,000
    assert r.describe()["mean"] == pytest.approx(47.4)


def test_equilibrium_matches_hand_solution():
    # 120 - 0.02P = 10 + 0.03P  ->  110 = 0.05P  ->  P = 2200, Q = 76
    market = LinearMarket(a=120, b=0.02, c=10, d=0.03)
    p, q = market.equilibrium()
    assert p == pytest.approx(2200) and q == pytest.approx(76)
    assert market.excess_demand(2000) == pytest.approx(10)  # 80 demanded vs 70 supplied


def test_moving_average_forecast():
    assert MovingAverageForecaster(3).fit(NTINDA).predict(1)[0] == pytest.approx((52 + 47 + 45) / 3)


def test_ses_hand_calculation():
    # levels: 10 -> 0.5*20 + 0.5*10 = 15 -> 0.5*30 + 0.5*15 = 22.5
    assert SimpleExponentialSmoothing(0.5).fit([10, 20, 30]).predict(1)[0] == pytest.approx(22.5)


def test_ses_alpha_one_is_naive():
    assert SimpleExponentialSmoothing(1.0).fit(NTINDA).predict(1)[0] == NTINDA[-1]


def test_seasonal_naive_repeats_last_week():
    y = list(range(1, 15))  # two "weeks"
    np.testing.assert_array_equal(SeasonalNaiveForecaster(7).fit(y).predict(3), [8, 9, 10])


def test_backtest_uses_days_4_to_10():
    # A perfect straight line is forecast perfectly by the linear-trend model.
    y = np.arange(10) * 3 + 5
    assert backtest_mae(LinearTrendForecaster, y, first_day=4) == pytest.approx(0)


def test_tune_alpha_returns_grid_value():
    best, scores = tune_alpha(NTINDA, grid=[0.1, 0.5, 0.9])
    assert best in (0.1, 0.5, 0.9) and scores[best] == min(scores.values())


def test_fleet_rounds_up_with_buffer():
    fp = FleetPlanner(trips_per_day=8, seats=14, buffer=0.15)  # 112 seats per vehicle per day
    assert fp.vehicles_needed(100) == 2  # 115 / 112 = 1.03 -> 2
    assert fp.vehicles_needed(97) == 1  # 111.55 / 112 -> 1


def test_seasonal_naive_beats_moving_average_on_weekly_data():
    y = simulate_weekly_demand(50, days=60, seed=1)
    assert backtest_mae(SeasonalNaiveForecaster, y, 15) < backtest_mae(lambda: MovingAverageForecaster(3), y, 15)


# --- edge cases -------------------------------------------------------------
def test_route_rejects_bad_input():
    with pytest.raises(ValueError):
        Route("Empty", [], fare=2000)
    with pytest.raises(ValueError):
        Route("Neg", [10, -1], fare=2000)
    with pytest.raises(ValueError):
        Route("Free", [10, 12], fare=0)


def test_invalid_alpha_and_window():
    with pytest.raises(ValueError):
        SimpleExponentialSmoothing(0)
    with pytest.raises(ValueError):
        MovingAverageForecaster(0)


def test_zero_demand_needs_zero_vehicles():
    assert FleetPlanner().vehicles_needed(0) == 0


def test_moving_average_needs_enough_history():
    with pytest.raises(ValueError):
        MovingAverageForecaster(3).fit([10, 12])
