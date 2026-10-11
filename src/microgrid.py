"""Mini-Project 2: Solar Micro-Grid Dispatch Planner (Kasese health centre).

The daily dispatch problem is a linear system  A @ [x, y] = [D1, D2]:

    3x + 2y = D1   (daytime load, kWh)
    4x +  y = D2   (critical-equipment load, kWh)

where x = energy from solar and y = energy from batteries.

Classes
    Tariff             prices per kWh for each source
    DispatchResult     the solution for many days, plus feasibility flags and costs
    MicroGrid          the 2x2 system: diagnostics, solving (loop / vectorised / NNLS)
    HybridMicroGrid    subclass adding a diesel generator (3x3 system) - extension

Functions
    generate_demand_csv / load_demand_csv   30-day demand file (seeded, weekly pattern)
    read_positive_float / prompt_day_demand interactive input with validation
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
from scipy import linalg, optimize

ArrayLike = Union[Sequence[Sequence[float]], np.ndarray]


# ---------------------------------------------------------------------------
# Small value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tariff:
    """Price per kWh (UGX) for each energy source, in the same order as the unknowns."""

    prices: Dict[str, float] = field(
        default_factory=lambda: {"solar": 150.0, "battery": 450.0}
    )

    def __post_init__(self) -> None:
        if any(p < 0 for p in self.prices.values()):
            raise ValueError("Prices cannot be negative.")

    def as_array(self, sources: Sequence[str]) -> np.ndarray:
        """Prices ordered to match ``sources``."""
        return np.array([self.prices[s] for s in sources], dtype=float)


@dataclass
class DispatchResult:
    """Energy drawn from each source for each day.

    Attributes:
        sources: Names of the sources, e.g. ("solar", "battery").
        energy: Array of shape (n_sources, n_days) in kWh.
        raw: The unconstrained linear-solve result (may contain negatives).
        infeasible: Boolean array (n_days,) -- True where the raw solution was negative.
        unmet: Residual demand (kWh) left unmet after the non-negative fix, per day.
    """

    sources: tuple
    energy: np.ndarray
    raw: np.ndarray
    infeasible: np.ndarray
    unmet: np.ndarray

    def daily_cost(self, tariff: Tariff) -> np.ndarray:
        """Cost per day in UGX: price vector · energy."""
        return tariff.as_array(self.sources) @ self.energy

    def total_cost(self, tariff: Tariff) -> float:
        """Total cost over all days in UGX."""
        return float(self.daily_cost(tariff).sum())


# ---------------------------------------------------------------------------
# The micro-grid
# ---------------------------------------------------------------------------
class MicroGrid:
    """A linear dispatch system ``A @ energy = demand``.

    Args:
        coefficients: Square coefficient matrix. Defaults to [[3, 2], [4, 1]].
        sources: Names of the unknowns (columns of ``A``).
        constraints: Names of the demand equations (rows of ``A``).
        singular_tol: |det| below this is treated as singular.
    """

    DEFAULT_A = np.array([[3.0, 2.0], [4.0, 1.0]])

    def __init__(
        self,
        coefficients: Optional[ArrayLike] = None,
        sources: Sequence[str] = ("solar", "battery"),
        constraints: Sequence[str] = ("daytime load D1", "critical load D2"),
        singular_tol: float = 1e-10,
    ) -> None:
        A = np.array(self.DEFAULT_A if coefficients is None else coefficients, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Coefficient matrix must be square.")
        if A.shape[1] != len(sources) or A.shape[0] != len(constraints):
            raise ValueError("Number of sources/constraints must match the matrix shape.")
        self.A = A
        self.sources = tuple(sources)
        self.constraints = tuple(constraints)
        self.singular_tol = singular_tol

    def __repr__(self) -> str:
        return f"{type(self).__name__}(sources={self.sources}, A={self.A.tolist()})"

    # -- diagnostics --------------------------------------------------------
    @property
    def size(self) -> int:
        return self.A.shape[0]

    def determinant(self) -> float:
        """det(A). Zero means no unique solution exists."""
        return float(np.linalg.det(self.A))

    def condition_number(self) -> float:
        """2-norm condition number: how much relative input errors can be amplified."""
        return float(np.linalg.cond(self.A))

    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.A))

    def is_well_posed(self, max_condition: float = 1e6) -> bool:
        """Unique solution exists (non-zero det) and the system is not badly conditioned."""
        return abs(self.determinant()) > self.singular_tol and self.condition_number() < max_condition

    def diagnostics(self) -> Dict[str, float]:
        return {
            "determinant": self.determinant(),
            "condition_number": self.condition_number(),
            "rank": self.rank(),
            "well_posed": self.is_well_posed(),
        }

    # -- solving ------------------------------------------------------------
    def _check_demand(self, demand: np.ndarray) -> np.ndarray:
        d = np.asarray(demand, dtype=float)
        if d.shape[0] != self.size:
            raise ValueError(f"Demand must have {self.size} rows, got shape {d.shape}.")
        if d.size == 0:
            raise ValueError("Demand is empty.")
        if not np.all(np.isfinite(d)):
            raise ValueError("Demand contains NaN or infinite values.")
        if np.any(d < 0):
            raise ValueError("Demand cannot be negative.")
        return d

    def _require_solvable(self) -> None:
        if abs(self.determinant()) <= self.singular_tol:
            raise np.linalg.LinAlgError(
                "Coefficient matrix is singular: the equations are linearly dependent, "
                "so there is no unique dispatch."
            )

    def solve_day(self, *demands: float) -> np.ndarray:
        """Solve one day, e.g. ``grid.solve_day(d1, d2)``. Returns energy per source (kWh)."""
        d = self._check_demand(np.array(demands, dtype=float))
        self._require_solvable()
        return linalg.solve(self.A, d)

    def solve_loop(self, demand: np.ndarray) -> np.ndarray:
        """Solve each day separately in a Python loop. ``demand`` has shape (n, days)."""
        d = self._check_demand(demand)
        self._require_solvable()
        return np.column_stack([linalg.solve(self.A, d[:, j]) for j in range(d.shape[1])])

    def solve_vectorised(self, demand: np.ndarray) -> np.ndarray:
        """Solve all days in ONE call by passing an (n x days) right-hand side."""
        d = self._check_demand(demand)
        self._require_solvable()
        return linalg.solve(self.A, d)

    def dispatch(self, demand: np.ndarray) -> DispatchResult:
        """Solve all days and repair physically impossible (negative) days.

        Strategy: solve exactly first; for any day with a negative source, re-solve that day
        with non-negative least squares (``scipy.optimize.nnls``). NNLS finds the
        non-negative dispatch that comes *closest* to meeting both loads; the remaining gap
        is reported as unmet demand so the operator knows the day needs attention.
        """
        d = self._check_demand(demand)
        raw = self.solve_vectorised(d)
        infeasible = np.any(raw < -1e-9, axis=0)
        energy = raw.copy()
        unmet = np.zeros(d.shape[1])
        for j in np.flatnonzero(infeasible):
            energy[:, j], unmet[j] = optimize.nnls(self.A, d[:, j])
        return DispatchResult(self.sources, energy, raw, infeasible, unmet)

    # -- sensitivity (extension) --------------------------------------------
    def monte_carlo_sensitivity(
        self, demand: Sequence[float], rel_perturbation: float = 0.05, n_draws: int = 1000, seed: int = 7
    ) -> Dict[str, np.ndarray]:
        """Perturb each demand uniformly by ±``rel_perturbation`` and re-solve.

        Returns the draws of demand and solutions, plus the observed worst-case
        amplification ‖Δx‖/‖x‖ ÷ ‖Δd‖/‖d‖, which can never exceed the condition number.
        """
        base_d = self._check_demand(np.asarray(demand, dtype=float))
        base_x = self.solve_day(*base_d)
        rng = np.random.default_rng(seed)
        factors = 1 + rng.uniform(-rel_perturbation, rel_perturbation, size=(self.size, n_draws))
        d_draws = base_d[:, None] * factors
        x_draws = self.solve_vectorised(d_draws)
        rel_dx = np.linalg.norm(x_draws - base_x[:, None], axis=0) / np.linalg.norm(base_x)
        rel_dd = np.linalg.norm(d_draws - base_d[:, None], axis=0) / np.linalg.norm(base_d)
        return {
            "demand_draws": d_draws,
            "solution_draws": x_draws,
            "base_solution": base_x,
            "amplification": rel_dx / rel_dd,
        }


class HybridMicroGrid(MicroGrid):
    """Extension: solar (x), battery (y) and diesel generator (z) -- a 3x3 system.

    Default equations (our choice):
        3x + 2y +  z = D1   daytime load (diesel can top up)
        4x +  y + 2z = D2   critical-equipment load
             y + 3z = D3   night-time load: solar cannot contribute after sunset
    """

    DEFAULT_A = np.array([[3.0, 2.0, 1.0], [4.0, 1.0, 2.0], [0.0, 1.0, 3.0]])

    def __init__(self, coefficients: Optional[ArrayLike] = None, **kwargs) -> None:
        kwargs.setdefault("sources", ("solar", "battery", "diesel"))
        kwargs.setdefault("constraints", ("daytime load D1", "critical load D2", "night load D3"))
        super().__init__(self.DEFAULT_A if coefficients is None else coefficients, **kwargs)


# ---------------------------------------------------------------------------
# Demand data: generation and loading
# ---------------------------------------------------------------------------
def generate_demand(days: int = 30, seed: int = 2026) -> np.ndarray:
    """Synthetic daily demand with a weekly pattern and noise. Shape (2, days), kWh.

    D1 (daytime load): ~120 kWh, busier on weekdays (outpatient clinics), quieter at weekends.
    D2 (critical load): ~140 kWh (fridges, oxygen, theatre), weakly tied to the week.
    """
    if days <= 0:
        raise ValueError("days must be positive.")
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    weekday = (t % 7) < 5
    d1 = 110 + 15 * weekday + rng.normal(0, 8, days)
    d2 = 138 + 6 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 12, days)
    return np.clip(np.vstack([d1, d2]), 0, None).round(1)


def generate_demand_csv(path: Union[str, Path], days: int = 30, seed: int = 2026) -> Path:
    """Write the synthetic demand to CSV with columns day, D1, D2."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    demand = generate_demand(days, seed)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["day", "D1_kWh", "D2_kWh"])
        for j in range(days):
            writer.writerow([j + 1, demand[0, j], demand[1, j]])
    return path


def load_demand_csv(path: Union[str, Path]) -> np.ndarray:
    """Read a demand CSV into a (2, days) array, validating every value."""
    rows = []
    with Path(path).open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            try:
                d1, d2 = float(row["D1_kWh"]), float(row["D2_kWh"])
            except (KeyError, TypeError, ValueError) as err:
                raise ValueError(f"Bad value on line {i} of {path}: {row}") from err
            if d1 < 0 or d2 < 0:
                raise ValueError(f"Negative demand on line {i} of {path}.")
            rows.append((d1, d2))
    if not rows:
        raise ValueError(f"{path} contains no data rows.")
    return np.array(rows).T


# ---------------------------------------------------------------------------
# Interactive input (the input function is injectable so it can be tested)
# ---------------------------------------------------------------------------
def parse_positive_float(text: str) -> float:
    """Convert user text to a non-negative float or raise ValueError with a helpful message."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty input - please type a number.")
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"'{text}' is not a number.") from None
    if not np.isfinite(value):
        raise ValueError("Value must be a finite number.")
    if value < 0:
        raise ValueError("Demand cannot be negative.")
    return value


def read_positive_float(
    prompt: str,
    input_func: Callable[[str], str] = input,
    print_func: Callable[[str], None] = print,
    max_attempts: int = 10,
) -> float:
    """Ask until the user types a valid non-negative number (re-prompts on bad input)."""
    for _ in range(max_attempts):
        try:
            return parse_positive_float(input_func(prompt))
        except ValueError as err:
            print_func(f"  Invalid: {err} Try again.")
    raise ValueError(f"No valid input after {max_attempts} attempts.")


def prompt_day_demand(
    input_func: Callable[[str], str] = input, print_func: Callable[[str], None] = print
) -> tuple:
    """Interactively read (D1, D2) for one day."""
    d1 = read_positive_float("Daytime load D1 (kWh): ", input_func, print_func)
    d2 = read_positive_float("Critical load D2 (kWh): ", input_func, print_func)
    return d1, d2
