"""Shared forecasting tools used by Mini-Project 1 (population) and Mini-Project 5 (taxi).

Why a shared module?
    Both projects need the same idea: "fit a model to a history, then predict the next
    few values", and both need the same error metrics. Writing that once here avoids
    duplicated logic (a rubric criterion) and guarantees both projects measure error the
    same way.

Contents
    Forecaster              abstract base class (the common interface)
    LinearTrendForecaster   straight-line trend fitted with np.polyfit
    mae, rmse, mape         error metrics
    walk_forward_predictions  rolling-origin (walk-forward) backtest helper
    bootstrap_prediction_intervals  residual bootstrap for prediction intervals
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class Forecaster(ABC):
    """Common interface for every forecasting model.

    Subclasses must implement :meth:`_fit` (learn parameters from the history) and
    :meth:`_predict` (produce ``horizon`` future values). The public :meth:`fit` and
    :meth:`predict` methods do the input checking once, here, so no subclass has to
    repeat it (template-method pattern).

    Attributes:
        name: Short human-readable model name used in tables and plot legends.
        min_history: Smallest number of observations the model can be fitted on.
    """

    name: str = "Forecaster"
    min_history: int = 1

    def __init__(self) -> None:
        self._y: np.ndarray | None = None

    # -- public API ---------------------------------------------------------
    def fit(self, y: ArrayLike) -> "Forecaster":
        """Fit the model to a 1-D history and return ``self`` (so calls can be chained).

        Raises:
            ValueError: if ``y`` is empty, not 1-D, too short, or contains NaN/inf.
        """
        arr = np.asarray(y, dtype=float)
        if arr.ndim != 1:
            raise ValueError("History must be a 1-D sequence.")
        if arr.size < self.min_history:
            raise ValueError(
                f"{self.name} needs at least {self.min_history} observations, got {arr.size}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("History contains NaN or infinite values.")
        self._y = arr
        self._fit(arr)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Return an array of ``horizon`` forecasts following the fitted history."""
        if self._y is None:
            raise RuntimeError(f"{self.name} must be fitted before calling predict().")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        return np.asarray(self._predict(int(horizon)), dtype=float)

    def fitted_values(self) -> np.ndarray:
        """In-sample fitted values (same length as the training history).

        Needed for plotting "fitted" lines and for bootstrapping residuals. Models that
        have no natural in-sample fit may leave this unimplemented.
        """
        raise NotImplementedError(f"{self.name} does not define fitted values.")

    def residuals(self) -> np.ndarray:
        """Training residuals: actual minus fitted."""
        if self._y is None:
            raise RuntimeError("Model is not fitted.")
        return self._y - self.fitted_values()

    # -- hooks for subclasses ----------------------------------------------
    @abstractmethod
    def _fit(self, y: np.ndarray) -> None:
        """Learn model parameters from a validated history ``y``."""

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray:
        """Return ``horizon`` forecasts."""

    def __repr__(self) -> str:
        status = "fitted" if self._y is not None else "unfitted"
        return f"{type(self).__name__}({status})"


# ---------------------------------------------------------------------------
# A model both projects use
# ---------------------------------------------------------------------------
class LinearTrendForecaster(Forecaster):
    """Straight-line trend ``y = a + b*t`` fitted by least squares (``np.polyfit``).

    Time is indexed ``t = 0, 1, ..., n-1`` over the training history, so the forecast for
    step ``h`` ahead is evaluated at ``t = n - 1 + h``.
    """

    name = "Linear trend"
    min_history = 2

    def __init__(self) -> None:
        super().__init__()
        self.slope: float = float("nan")
        self.intercept: float = float("nan")

    def _fit(self, y: np.ndarray) -> None:
        t = np.arange(y.size)
        self.slope, self.intercept = np.polyfit(t, y, deg=1)

    def _predict(self, horizon: int) -> np.ndarray:
        n = self._y.size  # type: ignore[union-attr]
        t_future = np.arange(n, n + horizon)
        return self.intercept + self.slope * t_future

    def fitted_values(self) -> np.ndarray:
        t = np.arange(self._y.size)  # type: ignore[union-attr]
        return self.intercept + self.slope * t


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------
def _check_pair(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"Shape mismatch: actual {a.shape} vs predicted {p.shape}.")
    if a.size == 0:
        raise ValueError("Cannot compute an error metric on empty input.")
    return a, p


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean Absolute Error: average size of the miss, in the data's own units."""
    a, p = _check_pair(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Root Mean Squared Error: like MAE but punishes large misses more heavily."""
    a, p = _check_pair(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean Absolute Percentage Error, in percent. Undefined if any actual value is 0."""
    a, p = _check_pair(actual, predicted)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when an actual value is zero.")
    return float(np.mean(np.abs((a - p) / a)) * 100)


# ---------------------------------------------------------------------------
# Backtesting and uncertainty
# ---------------------------------------------------------------------------
def walk_forward_predictions(
    model_factory: Callable[[], Forecaster], y: ArrayLike, first_index: int
) -> np.ndarray:
    """One-step-ahead rolling-origin backtest.

    For every position ``t`` from ``first_index`` to ``len(y) - 1`` a *fresh* model is
    fitted on ``y[:t]`` (everything before day t) and asked to predict ``y[t]``. This
    mimics how the forecaster would really be used: it never sees the future.

    Args:
        model_factory: Zero-argument callable returning a new, unfitted model.
        y: The full series.
        first_index: 0-based index of the first value to forecast.

    Returns:
        Array of one-step-ahead predictions aligned with ``y[first_index:]``.
    """
    arr = np.asarray(y, dtype=float)
    if not 1 <= first_index < arr.size:
        raise ValueError("first_index must be between 1 and len(y) - 1.")
    preds = [model_factory().fit(arr[:t]).predict(1)[0] for t in range(first_index, arr.size)]
    return np.array(preds)


def bootstrap_prediction_intervals(
    model_factory: Callable[[], Forecaster],
    y: ArrayLike,
    horizon: int,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual-bootstrap prediction intervals.

    Recipe (for each of ``n_boot`` resamples):
        1. Fit the model to the real data and keep its fitted values and residuals.
        2. Build a synthetic history = fitted values + residuals drawn *with replacement*.
        3. Refit the model on that synthetic history (captures parameter uncertainty).
        4. Forecast, then add further resampled residuals (captures future noise).
    The lower/upper percentiles of all simulated forecasts form the interval.

    Returns:
        (point_forecast, lower, upper), each of length ``horizon``.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    base = model_factory().fit(y)
    fitted = base.fitted_values()
    resid = base.residuals()
    resid = resid - resid.mean()  # centre so resampling adds no bias
    sims = np.empty((n_boot, horizon))
    for b in range(n_boot):
        y_star = fitted + rng.choice(resid, size=resid.size, replace=True)
        path = model_factory().fit(y_star).predict(horizon)
        sims[b] = path + rng.choice(resid, size=horizon, replace=True)
    tail = (1 - level) / 2 * 100
    lower, upper = np.percentile(sims, [tail, 100 - tail], axis=0)
    return base.predict(horizon), lower, upper
