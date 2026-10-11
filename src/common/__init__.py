"""Shared utilities used by more than one mini-project."""
from src.common.forecasting import (
    Forecaster,
    LinearTrendForecaster,
    bootstrap_prediction_intervals,
    mae,
    mape,
    rmse,
    walk_forward_predictions,
)

__all__ = [
    "Forecaster",
    "LinearTrendForecaster",
    "bootstrap_prediction_intervals",
    "mae",
    "mape",
    "rmse",
    "walk_forward_predictions",
]
