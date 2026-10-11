"""Tests for Mini-Project 1 (population forecaster)."""

import numpy as np
import pytest

from src.population import (
    CAGRForecaster,
    DistrictPopulation,
    FibonacciRatioForecaster,
    ModelComparison,
    classrooms_needed,
    fibonacci_ratios,
)

YEARS = list(range(2015, 2025))
KAMPALA = [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800]


@pytest.fixture
def kampala() -> DistrictPopulation:
    return DistrictPopulation("Kampala", YEARS, KAMPALA)


# --- normal cases -----------------------------------------------------------
def test_len_repr_and_year_lookup(kampala):
    assert len(kampala) == 10
    assert "Kampala" in repr(kampala)
    assert kampala[2020] == 1500


def test_cagr_matches_hand_calculation(kampala):
    # (1800 / 1200) ** (1/9) - 1 = 1.5 ** 0.1111 - 1 ≈ 0.04608
    assert kampala.cagr() == pytest.approx(0.04608, abs=1e-5)


def test_statistics_vs_numpy_variance_differ_by_ddof(kampala):
    s = kampala.stats_with_statistics_module()
    n = len(kampala)
    # statistics.variance divides by n-1; np.var (ddof=0) divides by n.
    assert s["variance"] * (n - 1) / n == pytest.approx(kampala.stats_with_numpy()["variance"])
    assert s["variance"] == pytest.approx(kampala.stats_with_numpy(ddof=1)["variance"])


def test_yoy_growth_first_value(kampala):
    assert kampala.yoy_growth()[0] == pytest.approx(50 / 1200)


def test_cagr_forecaster_continues_constant_growth():
    y = 100 * 1.1 ** np.arange(5)  # exactly 10 % growth
    fc = CAGRForecaster().fit(y).predict(2)
    np.testing.assert_allclose(fc, [y[-1] * 1.1, y[-1] * 1.21])


def test_fibonacci_ratios_and_forecast():
    np.testing.assert_allclose(fibonacci_ratios(4), [1, 2, 1.5, 5 / 3])
    np.testing.assert_allclose(FibonacciRatioForecaster().fit([10]).predict(3), [10, 20, 30])


def test_model_comparison_picks_lowest_rmse(kampala):
    mc = ModelComparison(kampala, last_train_year=2021)
    assert len(mc.test) == 3
    assert mc.best.rmse == min(s.rmse for s in mc.scores)


def test_classrooms_hand_calculation():
    # 100 thousand more people -> 18,000 more children -> 18000/53 = 339.6 -> 340 rooms
    assert classrooms_needed(1000, 1100) == 340


# --- edge cases -------------------------------------------------------------
def test_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        DistrictPopulation("Bad", [2015, 2016], [1, 2, 3])


def test_rejects_negative_population():
    with pytest.raises(ValueError):
        DistrictPopulation("Bad", [2015, 2016], [100, -5])


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        DistrictPopulation("Empty", [], [])


def test_classrooms_zero_for_shrinking_population():
    assert classrooms_needed(500, 450) == 0


def test_cagr_undefined_for_single_year():
    with pytest.raises(ValueError):
        DistrictPopulation("One", [2020], [100]).cagr()
