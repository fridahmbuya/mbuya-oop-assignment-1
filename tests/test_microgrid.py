"""Tests for Mini-Project 2 (solar micro-grid)."""
import numpy as np
import pytest

from src.microgrid import (
    HybridMicroGrid,
    MicroGrid,
    Tariff,
    generate_demand,
    generate_demand_csv,
    load_demand_csv,
    parse_positive_float,
    read_positive_float,
)


@pytest.fixture
def grid() -> MicroGrid:
    return MicroGrid()


# --- normal cases -----------------------------------------------------------
def test_determinant_and_condition(grid):
    assert grid.determinant() == pytest.approx(-5.0)  # 3*1 - 2*4
    assert grid.condition_number() == pytest.approx(3 + 2 * np.sqrt(2))  # ≈ 5.83
    assert grid.is_well_posed()


def test_solve_day_matches_hand_solution(grid):
    # Hand elimination: x = (2*D2 - D1)/5, y = (4*D1 - 3*D2)/5
    d1, d2 = 120.0, 140.0
    x, y = grid.solve_day(d1, d2)
    assert x == pytest.approx((2 * d2 - d1) / 5)  # 32
    assert y == pytest.approx((4 * d1 - 3 * d2) / 5)  # 12


def test_loop_and_vectorised_agree(grid):
    demand = generate_demand(30, seed=1)
    np.testing.assert_allclose(grid.solve_loop(demand), grid.solve_vectorised(demand))


def test_infeasible_day_is_flagged_and_repaired(grid):
    demand = np.array([[100.0, 120.0], [170.0, 140.0]])  # day 1: y = (400-510)/5 < 0
    result = grid.dispatch(demand)
    assert result.infeasible.tolist() == [True, False]
    assert np.all(result.energy >= 0)
    assert result.unmet[0] > 0 and result.unmet[1] == 0


def test_cost_hand_calculation(grid):
    result = grid.dispatch(np.array([[120.0], [140.0]]))  # x=32, y=12
    assert result.daily_cost(Tariff())[0] == pytest.approx(32 * 150 + 12 * 450)


def test_csv_round_trip(tmp_path):
    path = generate_demand_csv(tmp_path / "demand.csv", days=30, seed=3)
    np.testing.assert_allclose(load_demand_csv(path), generate_demand(30, seed=3))


def test_hybrid_is_3x3_and_solves():
    hybrid = HybridMicroGrid()
    x = hybrid.solve_day(150, 170, 60)
    np.testing.assert_allclose(hybrid.A @ x, [150, 170, 60])


# --- edge cases -------------------------------------------------------------
def test_negative_demand_rejected(grid):
    with pytest.raises(ValueError):
        grid.solve_day(-10, 50)


def test_dependent_equations_raise():
    # Third row = first row + second row -> singular
    dependent = HybridMicroGrid(coefficients=[[3, 2, 1], [4, 1, 2], [7, 3, 3]])
    assert dependent.rank() == 2
    with pytest.raises(np.linalg.LinAlgError):
        dependent.solve_day(1, 2, 3)


@pytest.mark.parametrize("bad", ["", "   ", "abc", "-5", "nan"])
def test_parse_rejects_bad_text(bad):
    with pytest.raises(ValueError):
        parse_positive_float(bad)


def test_interactive_reprompts_until_valid():
    answers = iter(["", "ten", "-3", "42.5"])
    messages = []
    value = read_positive_float("D1: ", input_func=lambda _: next(answers), print_func=messages.append)
    assert value == 42.5
    assert len(messages) == 3  # three rejections before success


def test_empty_csv_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("day,D1_kWh,D2_kWh\n")
    with pytest.raises(ValueError):
        load_demand_csv(path)
