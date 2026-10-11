"""Tests for Mini-Project 3 (fish stock & export risk)."""
import numpy as np
import pytest

from src.fishstock import (
    FishStock,
    PriceModel,
    RiskAssessor,
    fibonacci,
    weekly_revenue,
)


# --- normal cases -----------------------------------------------------------
def test_fibonacci_sequence():
    assert fibonacci(7) == [1, 1, 2, 3, 5, 8, 13]


def test_one_logistic_step_by_hand():
    # N1 = 4000 + 0.4*4000*(1 - 0.4) - 0.1*4000 = 4000 + 960 - 400 = 4560
    traj = FishStock(r=0.4, K=10_000, n0=4_000, h=0.1).simulate(1)
    assert traj.stock[1] == pytest.approx(4560)
    assert traj.harvest[0] == pytest.approx(400)


def test_msy_and_equilibrium():
    fish = FishStock(r=0.4, K=10_000)
    assert fish.msy == pytest.approx(1000)
    assert fish.h_msy == pytest.approx(0.2)
    assert fish.equilibrium_stock(0.1) == pytest.approx(7500)
    assert fish.sustainable_yield(0.2) == pytest.approx(fish.msy)


def test_long_simulation_converges_to_equilibrium():
    fish = FishStock(h=0.1)
    assert fish.simulate(200).final_stock == pytest.approx(fish.equilibrium_stock(), rel=1e-6)


def test_price_path_is_reproducible_and_bounded():
    a = PriceModel(seed=1).simulate(52, n_paths=50)
    b = PriceModel(seed=1).simulate(52, n_paths=50)
    np.testing.assert_array_equal(a, b)
    assert a.min() >= 9_000 and a.max() <= 16_000
    assert np.all(a[:, 0] == 12_000)


def test_revenue_units():
    # 2 tonnes at 12,000 UGX/kg = 2,000 kg * 12,000 = 24,000,000 UGX
    assert weekly_revenue([2.0], [12_000])[0] == pytest.approx(24_000_000)


def test_risk_classes_and_scale_invariance():
    ra = RiskAssessor(low=0.10, high=0.25)
    assert [ra.classify(c) for c in (0.05, 0.10, 0.3)] == ["Low", "Moderate", "High"]
    rev = np.array([90.0, 100.0, 110.0, 120.0])
    # Changing currency units changes variance by 1000^2 but leaves CV unchanged.
    assert ra.describe(rev * 1000)["variance"] == pytest.approx(ra.describe(rev)["variance"] * 1e6)
    assert ra.describe(rev * 1000)["cv"] == pytest.approx(ra.describe(rev)["cv"])


def test_value_at_risk_on_known_distribution():
    outcomes = np.arange(1, 101)  # 1..100
    var = RiskAssessor.value_at_risk(outcomes, level=0.05)
    assert var["quantile"] == pytest.approx(np.percentile(outcomes, 5))
    assert var["var_vs_mean"] == pytest.approx(50.5 - var["quantile"])


def test_closed_season_means_zero_catch():
    mask = np.zeros(10, bool)
    mask[3:5] = True
    traj = FishStock().simulate(10, closed_weeks=mask)
    assert np.all(traj.harvest[3:5] == 0)


# --- edge cases -------------------------------------------------------------
def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        FishStock(K=0)
    with pytest.raises(ValueError):
        FishStock(h=1.2)
    with pytest.raises(ValueError):
        PriceModel(start=20_000)


def test_zero_initial_stock_stays_zero():
    traj = FishStock(n0=0).simulate(10)
    assert traj.total_harvest == 0 and traj.final_stock == 0


def test_empty_outcomes_rejected():
    with pytest.raises(ValueError):
        RiskAssessor.value_at_risk([])


def test_fibonacci_zero_and_negative():
    assert fibonacci(0) == []
    with pytest.raises(ValueError):
        fibonacci(-1)
