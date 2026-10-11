"""Mini-Project 5: Taxi (matatu) Route Revenue, Pricing & Fleet Planner.

Classes
    Route                        passenger counts + fare for one route
    LinearMarket                 linear demand/supply and their equilibrium (2x2 system)
    MovingAverageForecaster      k-day moving average (the original method)
    SimpleExponentialSmoothing   SES with tunable alpha
    SeasonalNaiveForecaster      "same as this day last week" (extension)
    FleetPlanner                 converts a passenger forecast into a vehicle count

The abstract ``Forecaster`` base class and ``LinearTrendForecaster`` are shared with
Mini-Project 1 and live in ``src.common.forecasting``.
"""

from __future__ import annotations

import math
import statistics
from typing import Callable, Dict, Sequence, Tuple, Union

import numpy as np
from scipy import linalg

from src.common.forecasting import (
    Forecaster,
    LinearTrendForecaster,
    mae,
    walk_forward_predictions,
)

ArrayLike = Union[Sequence[float], np.ndarray]


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
class Route:
    """One taxi route.

    Args:
        name: e.g. "Kampala-Ntinda".
        passengers: Daily passenger counts (non-negative).
        fare: Fare per passenger in UGX (> 0).
        vehicle_capacity: Seats per vehicle (14 for a matatu).
    """

    def __init__(self, name: str, passengers: ArrayLike, fare: float, vehicle_capacity: int = 14) -> None:
        arr = np.asarray(passengers, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("passengers must be a non-empty 1-D sequence.")
        if np.any(arr < 0) or not np.all(np.isfinite(arr)):
            raise ValueError("Passenger counts must be finite and non-negative.")
        if fare <= 0:
            raise ValueError("Fare must be positive.")
        if vehicle_capacity <= 0:
            raise ValueError("Vehicle capacity must be positive.")
        self.name = name
        self.passengers = arr
        self.fare = float(fare)
        self.vehicle_capacity = vehicle_capacity

    def __repr__(self) -> str:
        return f"Route({self.name!r}, days={len(self)}, fare=UGX {self.fare:,.0f})"

    def __len__(self) -> int:
        return int(self.passengers.size)

    def daily_revenue(self) -> np.ndarray:
        """Revenue per day in UGX (passengers x fare)."""
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        return float(self.daily_revenue().sum())

    def describe(self) -> Dict[str, float]:
        """Mean, sample variance and sample SD of daily passengers (``statistics`` module)."""
        data = self.passengers.tolist()
        if len(data) < 2:
            raise ValueError("Need at least two days for variance.")
        return {
            "mean": statistics.mean(data),
            "variance": statistics.variance(data),
            "std": statistics.stdev(data),
        }


# ---------------------------------------------------------------------------
# Market equilibrium
# ---------------------------------------------------------------------------
class LinearMarket:
    """Linear demand Qd = a - b P and supply Qs = c + d P.

    Written as a linear system in the unknowns (Q, P):
        Q + b P = a
        Q - d P = c
    """

    def __init__(self, a: float, b: float, c: float, d: float) -> None:
        if b <= 0 or d <= 0:
            raise ValueError("Slopes must be positive (demand falls, supply rises with price).")
        self.a, self.b, self.c, self.d = a, b, c, d

    def __repr__(self) -> str:
        return f"LinearMarket(Qd = {self.a} - {self.b}P, Qs = {self.c} + {self.d}P)"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1.0, self.b], [1.0, -self.d]])

    @property
    def rhs(self) -> np.ndarray:
        return np.array([self.a, self.c], dtype=float)

    def equilibrium(self) -> Tuple[float, float]:
        """Return (P*, Q*) by solving the 2x2 system with ``scipy.linalg.solve``."""
        q, p = linalg.solve(self.matrix, self.rhs)
        return float(p), float(q)

    def demand(self, price: float) -> float:
        return self.a - self.b * price

    def supply(self, price: float) -> float:
        return self.c + self.d * price

    def excess_demand(self, price: float) -> float:
        """Positive = shortage (more riders than seats offered), negative = surplus."""
        return self.demand(price) - self.supply(price)


# ---------------------------------------------------------------------------
# Forecasters specific to this project
# ---------------------------------------------------------------------------
class MovingAverageForecaster(Forecaster):
    """Forecast = mean of the last ``window`` observations (flat for every horizon)."""

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("window must be at least 1.")
        self.window = window
        self.min_history = window
        self.name = f"{window}-day moving average"

    def _fit(self, y: np.ndarray) -> None:
        self.level = float(y[-self.window:].mean())

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level)


class SimpleExponentialSmoothing(Forecaster):
    """SES: level(t) = alpha * y(t) + (1 - alpha) * level(t-1); forecast = last level.

    alpha close to 1 -> reacts quickly (almost "yesterday's value");
    alpha close to 0 -> heavy smoothing (almost the long-run average).
    The level is initialised with the first observation.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = alpha
        self.name = f"SES (α={alpha:.2f})"
        self.min_history = 1

    def _fit(self, y: np.ndarray) -> None:
        levels = np.empty_like(y)
        levels[0] = y[0]
        for t in range(1, y.size):
            levels[t] = self.alpha * y[t] + (1 - self.alpha) * levels[t - 1]
        self.levels = levels

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.levels[-1])

    def fitted_values(self) -> np.ndarray:
        """One-step-ahead fits: the level *before* seeing each observation."""
        return np.concatenate([[self.levels[0]], self.levels[:-1]])


class SeasonalNaiveForecaster(Forecaster):
    """Forecast for a day = the value on the same weekday one season (7 days) earlier."""

    def __init__(self, season: int = 7) -> None:
        super().__init__()
        if season < 1:
            raise ValueError("season must be at least 1.")
        self.season = season
        self.min_history = season
        self.name = f"Seasonal naïve (m={season})"

    def _fit(self, y: np.ndarray) -> None:
        self.last_season = y[-self.season:].copy()

    def _predict(self, horizon: int) -> np.ndarray:
        return np.array([self.last_season[h % self.season] for h in range(horizon)])


def default_models(alpha: float = 0.5) -> Dict[str, Callable[[], Forecaster]]:
    """The three required forecasters as zero-argument factories."""
    return {
        "3-day moving average": lambda: MovingAverageForecaster(3),
        "SES": lambda: SimpleExponentialSmoothing(alpha),
        "Linear trend": LinearTrendForecaster,
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def backtest_mae(factory: Callable[[], Forecaster], y: ArrayLike, first_day: int = 4) -> float:
    """MAE of one-step-ahead walk-forward forecasts from ``first_day`` (1-based) to the end."""
    arr = np.asarray(y, dtype=float)
    first_index = first_day - 1
    preds = walk_forward_predictions(factory, arr, first_index)
    return mae(arr[first_index:], preds)


def tune_alpha(
    y: ArrayLike, grid: Sequence[float] = tuple(np.round(np.arange(0.05, 1.0001, 0.05), 2)), first_day: int = 4
) -> Tuple[float, Dict[float, float]]:
    """Grid search for the SES alpha with the lowest walk-forward MAE. Returns (best, all scores)."""
    if len(grid) == 0:
        raise ValueError("alpha grid is empty.")
    scores = {float(a): backtest_mae(lambda a=a: SimpleExponentialSmoothing(a), y, first_day) for a in grid}
    best = min(scores, key=scores.get)
    return best, scores


# ---------------------------------------------------------------------------
# Fleet planning
# ---------------------------------------------------------------------------
class FleetPlanner:
    """How many vehicles to deploy for a given passenger forecast.

    vehicles = ceil(forecast * (1 + buffer) / (trips_per_day * seats))

    Rounding *up* is deliberate: a fraction of a vehicle cannot be deployed, and rounding
    down would leave passengers stranded.
    """

    def __init__(self, trips_per_day: int = 8, seats: int = 14, buffer: float = 0.15) -> None:
        if trips_per_day <= 0 or seats <= 0:
            raise ValueError("trips_per_day and seats must be positive.")
        if buffer < 0:
            raise ValueError("buffer cannot be negative.")
        self.trips_per_day, self.seats, self.buffer = trips_per_day, seats, buffer

    @property
    def daily_capacity_per_vehicle(self) -> int:
        return self.trips_per_day * self.seats

    def vehicles_needed(self, passengers: float) -> int:
        if passengers < 0:
            raise ValueError("Passenger forecast cannot be negative.")
        return math.ceil(passengers * (1 + self.buffer) / self.daily_capacity_per_vehicle)

    def load_factor(self, passengers: float, vehicles: int) -> float:
        """Share of seats filled if ``vehicles`` are deployed."""
        if vehicles <= 0:
            raise ValueError("vehicles must be positive.")
        return passengers / (vehicles * self.daily_capacity_per_vehicle)


# ---------------------------------------------------------------------------
# Extension: synthetic 60-day data with a weekly pattern
# ---------------------------------------------------------------------------
WEEKDAY_FACTORS = np.array([1.00, 0.95, 1.00, 1.05, 1.35, 1.10, 0.60])  # Mon ... Sun


def simulate_weekly_demand(
    base: float, days: int = 60, noise_sd: float = 0.05, seed: int = 2026,
    factors: ArrayLike = WEEKDAY_FACTORS,
) -> np.ndarray:
    """Daily passengers = base x weekday factor x (1 + noise), rounded; day 1 is a Monday."""
    if base <= 0 or days <= 0:
        raise ValueError("base and days must be positive.")
    f = np.asarray(factors, dtype=float)
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    y = base * f[t % f.size] * (1 + rng.normal(0, noise_sd, days))
    return np.clip(np.round(y), 0, None)
