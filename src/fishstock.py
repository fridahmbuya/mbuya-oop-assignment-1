"""Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model (Jinja cooperative).

Classes
    StockTrajectory   result of a stock simulation (stock path + weekly harvest)
    FishStock         discrete logistic growth with proportional harvesting
    PriceModel        bounded, seeded random walk for the weekly fish price
    RiskAssessor      CV-based risk classification, Monte Carlo and Value-at-Risk

Functions
    fibonacci         the old "stock" baseline, for comparison only
    weekly_revenue    harvest (tonnes) x 1000 x price (UGX/kg)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


def fibonacci(n: int) -> List[int]:
    """First ``n`` Fibonacci numbers starting 1, 1, 2, 3, ..."""
    if n < 0:
        raise ValueError("n cannot be negative.")
    seq: List[int] = []
    a, b = 1, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


# ---------------------------------------------------------------------------
# Biology
# ---------------------------------------------------------------------------
@dataclass
class StockTrajectory:
    """Output of :meth:`FishStock.simulate`.

    Attributes:
        stock: Biomass at the start of each week, length ``weeks + 1`` (tonnes).
        harvest: Catch taken during each week, length ``weeks`` (tonnes).
    """

    stock: np.ndarray
    harvest: np.ndarray

    @property
    def final_stock(self) -> float:
        return float(self.stock[-1])

    @property
    def total_harvest(self) -> float:
        return float(self.harvest.sum())


class FishStock:
    """Discrete logistic growth with harvesting:

        N(t+1) = N(t) + r N(t) (1 - N(t)/K) - h N(t)

    Args:
        r: Intrinsic growth rate per week (> 0).
        K: Carrying capacity in tonnes (> 0).
        n0: Initial biomass in tonnes (0 <= n0).
        h: Fraction of the stock harvested each week (0 <= h < 1).
    """

    def __init__(self, r: float = 0.4, K: float = 10_000, n0: float = 4_000, h: float = 0.1) -> None:
        if r <= 0:
            raise ValueError("Growth rate r must be positive.")
        if K <= 0:
            raise ValueError("Carrying capacity K must be positive.")
        if n0 < 0:
            raise ValueError("Initial stock cannot be negative.")
        if not 0 <= h < 1:
            raise ValueError("Harvest rate h must be in [0, 1).")
        self.r, self.K, self.n0, self.h = float(r), float(K), float(n0), float(h)

    def __repr__(self) -> str:
        return f"FishStock(r={self.r}, K={self.K:,.0f}, n0={self.n0:,.0f}, h={self.h})"

    # -- theory -------------------------------------------------------------
    @property
    def msy(self) -> float:
        """Maximum sustainable yield rK/4 (tonnes per week), reached at N = K/2."""
        return self.r * self.K / 4

    @property
    def h_msy(self) -> float:
        """Harvest rate that achieves MSY: h = r/2."""
        return self.r / 2

    def equilibrium_stock(self, h: Optional[float] = None) -> float:
        """Long-run stock where growth exactly offsets harvest: N* = K(1 - h/r), floored at 0."""
        h = self.h if h is None else h
        return max(0.0, self.K * (1 - h / self.r))

    def sustainable_yield(self, h: Optional[float] = None) -> float:
        """Weekly catch at equilibrium: h * N*."""
        h = self.h if h is None else h
        return h * self.equilibrium_stock(h)

    # -- simulation ---------------------------------------------------------
    def simulate(self, weeks: int = 52, closed_weeks: Optional[ArrayLike] = None) -> StockTrajectory:
        """Simulate ``weeks`` steps.

        Args:
            weeks: Number of weekly steps.
            closed_weeks: Optional boolean mask of length ``weeks``; True = no fishing that week.
        """
        if weeks < 1:
            raise ValueError("weeks must be at least 1.")
        closed = np.zeros(weeks, dtype=bool) if closed_weeks is None else np.asarray(closed_weeks, bool)
        if closed.shape != (weeks,):
            raise ValueError("closed_weeks must have one entry per week.")
        stock = np.empty(weeks + 1)
        harvest = np.empty(weeks)
        stock[0] = self.n0
        for t in range(weeks):
            n = stock[t]
            harvest[t] = 0.0 if closed[t] else self.h * n
            growth = self.r * n * (1 - n / self.K)
            stock[t + 1] = max(0.0, n + growth - harvest[t])  # biomass cannot go negative
        return StockTrajectory(stock, harvest)


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------
class PriceModel:
    """Weekly export price (UGX/kg) as a bounded Gaussian random walk.

    p(t+1) = clip(p(t) + e(t), lower, upper),  e(t) ~ Normal(0, step_sd)

    Args:
        start: Price in week 1.
        lower, upper: Bounds (e.g. a floor set by local demand, a cap set by importers).
        step_sd: Standard deviation of the weekly change (UGX/kg).
        seed: Seed for ``np.random.default_rng``, for reproducibility.
    """

    def __init__(
        self,
        start: float = 12_000,
        lower: float = 9_000,
        upper: float = 16_000,
        step_sd: float = 400,
        seed: int = 2026,
    ) -> None:
        if not lower < upper:
            raise ValueError("lower bound must be below upper bound.")
        if not lower <= start <= upper:
            raise ValueError("start price must lie within the bounds.")
        if step_sd < 0:
            raise ValueError("step_sd cannot be negative.")
        self.start, self.lower, self.upper, self.step_sd = start, lower, upper, step_sd
        self.seed = seed

    def __repr__(self) -> str:
        return (f"PriceModel(start={self.start:,.0f}, bounds=[{self.lower:,.0f}, {self.upper:,.0f}], "
                f"step_sd={self.step_sd}, seed={self.seed})")

    def simulate(self, weeks: int = 52, n_paths: int = 1) -> np.ndarray:
        """Return prices of shape ``(n_paths, weeks)``; week 1 equals ``start``."""
        if weeks < 1 or n_paths < 1:
            raise ValueError("weeks and n_paths must be at least 1.")
        rng = np.random.default_rng(self.seed)
        prices = np.empty((n_paths, weeks))
        prices[:, 0] = self.start
        shocks = rng.normal(0, self.step_sd, size=(n_paths, weeks - 1))
        for t in range(1, weeks):
            prices[:, t] = np.clip(prices[:, t - 1] + shocks[:, t - 1], self.lower, self.upper)
        return prices


def weekly_revenue(harvest_tonnes: ArrayLike, price_ugx_per_kg: ArrayLike) -> np.ndarray:
    """Revenue in UGX = harvest (tonnes) x 1000 kg/tonne x price (UGX/kg). Broadcasts over paths."""
    h = np.asarray(harvest_tonnes, dtype=float)
    p = np.asarray(price_ugx_per_kg, dtype=float)
    if np.any(h < 0) or np.any(p < 0):
        raise ValueError("Harvest and price must be non-negative.")
    return h * 1000 * p


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------
class RiskAssessor:
    """Classify revenue risk by coefficient of variation (CV = SD / mean).

    Default rule (justified in the notebook):
        CV < 0.10          -> "Low"      (a typical week is within ±10 % of normal)
        0.10 <= CV < 0.25  -> "Moderate"
        CV >= 0.25         -> "High"     (about 1 week in 6 falls below 75 % of the average)

    Args:
        low: Upper CV limit of the Low class.
        high: Lower CV limit of the High class.
    """

    def __init__(self, low: float = 0.10, high: float = 0.25) -> None:
        if not 0 < low < high:
            raise ValueError("Thresholds must satisfy 0 < low < high.")
        self.low, self.high = low, high

    def __repr__(self) -> str:
        return f"RiskAssessor(low={self.low}, high={self.high})"

    @staticmethod
    def describe(revenue: ArrayLike) -> Dict[str, float]:
        """Mean, median, sample variance, sample SD and CV using the ``statistics`` module."""
        data = [float(v) for v in np.asarray(revenue).ravel()]
        if len(data) < 2:
            raise ValueError("Need at least two values to describe variability.")
        mean = statistics.mean(data)
        sd = statistics.stdev(data)
        return {
            "mean": mean,
            "median": statistics.median(data),
            "variance": statistics.variance(data),
            "std": sd,
            "cv": sd / mean if mean != 0 else float("inf"),
        }

    def classify(self, cv: float) -> str:
        """Return "Low", "Moderate" or "High" for a given CV."""
        if cv < 0 or np.isnan(cv):
            raise ValueError("CV must be a non-negative number.")
        if cv < self.low:
            return "Low"
        if cv < self.high:
            return "Moderate"
        return "High"

    def classify_revenue(self, revenue: ArrayLike) -> str:
        return self.classify(self.describe(revenue)["cv"])

    @staticmethod
    def monte_carlo_annual_revenue(
        trajectory: StockTrajectory, price_model: PriceModel, n_paths: int = 1000
    ) -> np.ndarray:
        """Annual (total) revenue for each of ``n_paths`` simulated price paths."""
        prices = price_model.simulate(weeks=trajectory.harvest.size, n_paths=n_paths)
        return weekly_revenue(trajectory.harvest, prices).sum(axis=1)

    @staticmethod
    def value_at_risk(outcomes: ArrayLike, level: float = 0.05) -> Dict[str, float]:
        """Value-at-Risk of an outcome distribution.

        Returns:
            ``quantile``: the ``level`` quantile (revenue is below this with probability ``level``);
            ``var_vs_mean``: shortfall of that quantile below the mean (the usual VaR figure);
            ``expected_shortfall``: average outcome in the worst ``level`` tail.
        """
        arr = np.asarray(outcomes, dtype=float)
        if arr.size == 0:
            raise ValueError("No outcomes supplied.")
        if not 0 < level < 1:
            raise ValueError("level must be between 0 and 1.")
        q = float(np.percentile(arr, level * 100))
        return {
            "quantile": q,
            "var_vs_mean": float(arr.mean() - q),
            "expected_shortfall": float(arr[arr <= q].mean()),
        }
