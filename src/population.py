"""Mini-Project 1: UBOS District Population Forecaster.

Classes
    DistrictPopulation      one district's population time series (the data object)
    CAGRForecaster          constant-percentage (exponential) growth model
    FibonacciRatioForecaster  previous cohort's Fibonacci-ratio model (kept for critique)
    ModelComparison         train/test evaluation of several models for one district

Functions
    classrooms_needed       turns a population change into a classroom count

The linear-trend model and the error metrics live in ``src.common.forecasting`` because
Mini-Project 5 uses them too.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Union

import numpy as np

from src.common.forecasting import (
    Forecaster,
    LinearTrendForecaster,
    mae,
    mape,
    rmse,
)

ArrayLike = Union[Sequence[float], np.ndarray]

# Data from the assignment brief (thousands of people, illustrative, 2015-2024) plus two
# extra districts of our choice. Mukono and Mbarara are ALSO illustrative: shaped to look
# like a fast-growing peri-urban district and a steadier regional-city district.
YEARS: np.ndarray = np.arange(2015, 2025)
DISTRICT_DATA: Dict[str, List[float]] = {
    "Kampala": [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800],
    "Wakiso": [950, 1000, 1070, 1150, 1220, 1300, 1390, 1480, 1570, 1670],
    "Gulu": [320, 330, 345, 360, 375, 390, 410, 430, 455, 480],
    "Mukono": [600, 632, 659, 698, 729, 776, 808, 857, 893, 948],
    "Mbarara": [470, 484, 496, 511, 523, 537, 549, 563, 577, 589],
}


# ---------------------------------------------------------------------------
# Data object
# ---------------------------------------------------------------------------
class DistrictPopulation:
    """Population time series for one district.

    Args:
        name: District name, e.g. ``"Kampala"``.
        years: Calendar years, strictly increasing.
        populations: Population for each year (thousands of people).

    Raises:
        ValueError: if the arrays are empty, of unequal length, contain negative or
            non-finite values, or the years are not strictly increasing.
    """

    def __init__(self, name: str, years: ArrayLike, populations: ArrayLike) -> None:
        years_arr = np.asarray(years, dtype=int)
        pop_arr = np.asarray(populations, dtype=float)
        if not name or not str(name).strip():
            raise ValueError("District name must be a non-empty string.")
        if years_arr.size == 0 or pop_arr.size == 0:
            raise ValueError("Years and populations must not be empty.")
        if years_arr.shape != pop_arr.shape:
            raise ValueError(
                f"years ({years_arr.size}) and populations ({pop_arr.size}) differ in length."
            )
        if not np.all(np.isfinite(pop_arr)):
            raise ValueError("Populations must be finite numbers.")
        if np.any(pop_arr < 0):
            raise ValueError("Populations cannot be negative.")
        if np.any(np.diff(years_arr) <= 0):
            raise ValueError("Years must be strictly increasing.")
        self.name = str(name)
        self.years = years_arr
        self.populations = pop_arr

    # -- dunder methods -----------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"DistrictPopulation(name={self.name!r}, years={self.years[0]}-{self.years[-1]}, "
            f"n={len(self)}, latest={self.populations[-1]:,.0f}k)"
        )

    def __len__(self) -> int:
        return int(self.populations.size)

    def __getitem__(self, year: int) -> float:
        """Population for a given calendar year, e.g. ``kampala[2020]``."""
        matches = np.flatnonzero(self.years == year)
        if matches.size == 0:
            raise KeyError(f"{self.name} has no data for {year}.")
        return float(self.populations[matches[0]])

    # -- descriptive statistics ---------------------------------------------
    def stats_with_statistics_module(self) -> Dict[str, float]:
        """Mean, median, *sample* variance and *sample* SD using the ``statistics`` module."""
        data = self.populations.tolist()
        return {
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "variance": statistics.variance(data),  # divides by n - 1
            "std": statistics.stdev(data),
        }

    def stats_with_numpy(self, ddof: int = 0) -> Dict[str, float]:
        """Same statistics with NumPy. ``ddof=0`` (default) divides by n, ``ddof=1`` by n-1."""
        p = self.populations
        return {
            "mean": float(np.mean(p)),
            "median": float(np.median(p)),
            "variance": float(np.var(p, ddof=ddof)),
            "std": float(np.std(p, ddof=ddof)),
        }

    # -- growth -------------------------------------------------------------
    def yoy_growth(self) -> np.ndarray:
        """Year-on-year growth rates as decimals (0.05 = 5 %). Length is ``len(self) - 1``."""
        if len(self) < 2:
            raise ValueError("Need at least two years to compute growth.")
        prev = self.populations[:-1]
        if np.any(prev == 0):
            raise ValueError("Growth is undefined when a previous year's population is 0.")
        return np.diff(self.populations) / prev

    def cagr(self) -> float:
        """Compound Annual Growth Rate between the first and last year.

        CAGR = (last / first) ** (1 / number_of_years_elapsed) - 1
        """
        if len(self) < 2:
            raise ValueError("Need at least two years to compute CAGR.")
        first, last = self.populations[0], self.populations[-1]
        if first <= 0:
            raise ValueError("CAGR is undefined when the first value is zero.")
        n_years = self.years[-1] - self.years[0]
        return float((last / first) ** (1 / n_years) - 1)

    # -- splitting ----------------------------------------------------------
    def split(self, last_train_year: int) -> tuple["DistrictPopulation", "DistrictPopulation"]:
        """Split into (train, test) where train ends at ``last_train_year`` inclusive."""
        mask = self.years <= last_train_year
        if mask.all() or not mask.any():
            raise ValueError("Split year must leave data on both sides.")
        return (
            DistrictPopulation(self.name, self.years[mask], self.populations[mask]),
            DistrictPopulation(self.name, self.years[~mask], self.populations[~mask]),
        )


# ---------------------------------------------------------------------------
# Forecasting models specific to this project
# ---------------------------------------------------------------------------
class CAGRForecaster(Forecaster):
    """Exponential growth at the history's compound annual growth rate.

    forecast(h) = last_value * (1 + g) ** h, where g is the CAGR of the history.
    """

    name = "Exponential (CAGR)"
    min_history = 2

    def __init__(self) -> None:
        super().__init__()
        self.growth_rate: float = float("nan")

    def _fit(self, y: np.ndarray) -> None:
        if y[0] <= 0:
            raise ValueError("CAGR model needs a positive first value.")
        self.growth_rate = float((y[-1] / y[0]) ** (1 / (y.size - 1)) - 1)

    def _predict(self, horizon: int) -> np.ndarray:
        steps = np.arange(1, horizon + 1)
        return self._y[-1] * (1 + self.growth_rate) ** steps  # type: ignore[index]

    def fitted_values(self) -> np.ndarray:
        # The exponential curve through the first point with the fitted growth rate.
        t = np.arange(self._y.size)  # type: ignore[union-attr]
        return self._y[0] * (1 + self.growth_rate) ** t  # type: ignore[index]


def fibonacci_ratios(count: int) -> np.ndarray:
    """First ``count`` ratios F(k+1)/F(k) with F(1) = F(2) = 1: 1, 2, 1.5, 1.667, 1.6, ..."""
    if count < 1:
        raise ValueError("count must be at least 1.")
    fib = [1, 1]
    while len(fib) < count + 1:
        fib.append(fib[-1] + fib[-2])
    return np.array([fib[k + 1] / fib[k] for k in range(count)])


class FibonacciRatioForecaster(Forecaster):
    """The previous cohort's model: multiply the last value by successive Fibonacci ratios.

    forecast(1) = last * r1, forecast(2) = forecast(1) * r2, ... with r = 1, 2, 1.5, 1.67, ...
    The ratios tend to the golden ratio (about 1.618), i.e. roughly 62 % growth per step,
    regardless of the data. It is implemented faithfully so it can be evaluated and
    critiqued, not because it is recommended.
    """

    name = "Fibonacci ratio"
    min_history = 1

    def _fit(self, y: np.ndarray) -> None:
        # Nothing is learned from the data -- which is itself the main criticism.
        pass

    def _predict(self, horizon: int) -> np.ndarray:
        return self._y[-1] * np.cumprod(fibonacci_ratios(horizon))  # type: ignore[index]

    def fitted_values(self) -> np.ndarray:
        # One-step-ahead in-sample fit: the first ratio is 1, so each fit is last year's value.
        y = self._y  # type: ignore[assignment]
        return np.concatenate([[y[0]], y[:-1]])


MODEL_FACTORIES: Dict[str, Callable[[], Forecaster]] = {
    LinearTrendForecaster.name: LinearTrendForecaster,
    CAGRForecaster.name: CAGRForecaster,
    FibonacciRatioForecaster.name: FibonacciRatioForecaster,
}


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------
@dataclass
class ModelScore:
    """Test-set errors for one model on one district."""

    model: str
    mae: float
    rmse: float
    mape: float
    predictions: np.ndarray


class ModelComparison:
    """Train on early years, test on later years, and pick the best model.

    Composition: a ModelComparison *has a* district and *has* several model factories.

    Args:
        district: The district to evaluate.
        last_train_year: Last year used for training (2021 in the brief).
        models: Mapping of model name -> factory. Defaults to the three required models.
        criterion: Metric used to choose the winner (``"rmse"``, ``"mae"`` or ``"mape"``).
    """

    def __init__(
        self,
        district: DistrictPopulation,
        last_train_year: int = 2021,
        models: Dict[str, Callable[[], Forecaster]] | None = None,
        criterion: str = "rmse",
    ) -> None:
        if criterion not in {"rmse", "mae", "mape"}:
            raise ValueError("criterion must be 'rmse', 'mae' or 'mape'.")
        self.district = district
        self.train, self.test = district.split(last_train_year)
        self.models = models or MODEL_FACTORIES
        self.criterion = criterion
        self.scores: List[ModelScore] = self._evaluate()

    def _evaluate(self) -> List[ModelScore]:
        scores = []
        actual = self.test.populations
        for name, factory in self.models.items():
            preds = factory().fit(self.train.populations).predict(len(self.test))
            scores.append(
                ModelScore(name, mae(actual, preds), rmse(actual, preds), mape(actual, preds), preds)
            )
        return scores

    @property
    def best(self) -> ModelScore:
        """The model with the lowest value of the chosen criterion."""
        return min(self.scores, key=lambda s: getattr(s, self.criterion))

    def table(self) -> List[Dict[str, object]]:
        """Rows ready for ``pandas.DataFrame`` (one row per model)."""
        return [
            {
                "District": self.district.name,
                "Model": s.model,
                "MAE (k)": round(s.mae, 2),
                "RMSE (k)": round(s.rmse, 2),
                "MAPE (%)": round(s.mape, 2),
                "Best": "✓" if s is self.best else "",
            }
            for s in self.scores
        ]


# ---------------------------------------------------------------------------
# Planning output
# ---------------------------------------------------------------------------
def classrooms_needed(
    population_now: float,
    population_future: float,
    school_age_share: float = 0.18,
    pupils_per_classroom: int = 53,
    unit: int = 1000,
) -> int:
    """Additional classrooms required for the growth in school-age children.

    additional = ceil(share * (future - now) * unit / pupils_per_classroom)

    ``unit`` converts the population figures to people (they are in thousands). A shrinking
    population returns 0 -- classrooms are not "negative-built".
    """
    if not 0 <= school_age_share <= 1:
        raise ValueError("school_age_share must be between 0 and 1.")
    if pupils_per_classroom <= 0:
        raise ValueError("pupils_per_classroom must be positive.")
    if population_now < 0 or population_future < 0:
        raise ValueError("Populations cannot be negative.")
    extra_pupils = school_age_share * (population_future - population_now) * unit
    return max(0, math.ceil(extra_pupils / pupils_per_classroom))


def load_districts() -> List[DistrictPopulation]:
    """Build a DistrictPopulation for every district in ``DISTRICT_DATA``."""
    return [DistrictPopulation(name, YEARS, pops) for name, pops in DISTRICT_DATA.items()]
