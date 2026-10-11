"""Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser.

Classes
    Region               12 months of rainfall for one place, with summary methods
    CropRule             a crop's suitable monthly-rainfall band, with its source
    SimilarityAnalyser   pairwise cosine / Pearson / Euclidean matrices for many regions

Functions
    cosine_similarity    the *correct* vector cosine similarity (dot / product of norms)
    fetch_nasa_power_monthly, load_rainfall_history_csv   real-data extension helpers
"""

from __future__ import annotations

import calendar
import csv
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal import find_peaks

ArrayLike = Union[Sequence[float], np.ndarray]
MONTHS: List[str] = [calendar.month_abbr[m] for m in range(1, 13)]


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """cos(theta) = (a · b) / (||a|| ||b||).

    The previous cohort used ``math.cos()``, which takes an *angle* and returns its cosine.
    Here we compute the cosine of the angle *between two vectors* from their coordinates.
    """
    u = np.asarray(a, dtype=float)
    v = np.asarray(b, dtype=float)
    if u.shape != v.shape or u.ndim != 1 or u.size == 0:
        raise ValueError("Vectors must be non-empty, 1-D and the same length.")
    norm_product = np.linalg.norm(u) * np.linalg.norm(v)
    if norm_product == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return float(np.dot(u, v) / norm_product)


# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------
class Region:
    """Monthly rainfall (mm, Jan-Dec) for one region.

    Raises:
        ValueError: unless exactly 12 finite, non-negative values are given.
    """

    def __init__(self, name: str, rainfall_mm: ArrayLike) -> None:
        arr = np.asarray(rainfall_mm, dtype=float)
        if arr.shape != (12,):
            raise ValueError(f"Expected 12 monthly values, got {arr.size}.")
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError("Rainfall must be finite and non-negative.")
        self.name = name
        self.rainfall = arr

    def __repr__(self) -> str:
        return f"Region({self.name!r}, annual={self.annual_total():,.0f} mm, {self.regime()})"

    def __len__(self) -> int:
        return 12

    # -- summaries ----------------------------------------------------------
    def annual_total(self) -> float:
        return float(self.rainfall.sum())

    def mean_monthly(self) -> float:
        return float(self.rainfall.mean())

    def wettest_month(self) -> Tuple[str, float]:
        i = int(np.argmax(self.rainfall))
        return MONTHS[i], float(self.rainfall[i])

    def driest_month(self) -> Tuple[str, float]:
        i = int(np.argmin(self.rainfall))
        return MONTHS[i], float(self.rainfall[i])

    def coefficient_of_variation(self) -> float:
        """Sample SD / mean of monthly rainfall: how uneven the year is (0 = perfectly even)."""
        mean = self.rainfall.mean()
        if mean == 0:
            raise ValueError("CV undefined: no rainfall at all.")
        return float(self.rainfall.std(ddof=1) / mean)

    # -- seasonality ----------------------------------------------------------
    def rainy_season_peaks(self, prominence_frac: float = 0.25) -> List[int]:
        """Month indices (0 = Jan) of rainy-season peaks found by ``scipy.signal.find_peaks``.

        Two design choices (explained in the notebook):
            * The year is treated as *circular*: we tile it three times and keep peaks in the
              middle copy, so a December or January peak is not lost at the array edge.
            * A peak must stand out by at least ``prominence_frac`` of the annual range, so a
              small dip inside one long season is not counted as two seasons.
        """
        if not 0 <= prominence_frac < 1:
            raise ValueError("prominence_frac must be in [0, 1).")
        span = self.rainfall.max() - self.rainfall.min()
        if span == 0:
            return []
        tiled = np.tile(self.rainfall, 3)
        peaks, _ = find_peaks(tiled, prominence=prominence_frac * span)
        return sorted(int(p - 12) for p in peaks if 12 <= p < 24)

    def regime(self, prominence_frac: float = 0.25) -> str:
        n = len(self.rainy_season_peaks(prominence_frac))
        return {0: "no clear season", 1: "unimodal"}.get(n, "bimodal" if n == 2 else f"{n} peaks")


# ---------------------------------------------------------------------------
# Crops
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CropRule:
    """Suitable monthly rainfall band for a crop.

    Attributes:
        crop: Crop name.
        min_mm, max_mm: Suitable monthly rainfall range (mm).
        source: Citation for the thresholds.
        annual_range: Optional (min, max) annual total, useful for perennials like coffee.
    """

    crop: str
    min_mm: float
    max_mm: float
    source: str
    annual_range: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        if self.min_mm < 0 or self.min_mm >= self.max_mm:
            raise ValueError("Need 0 <= min_mm < max_mm.")

    def score(self, mm: float) -> int:
        """-1 = too dry, 0 = suitable, +1 = too wet."""
        if mm < 0:
            raise ValueError("Rainfall cannot be negative.")
        if mm < self.min_mm:
            return -1
        if mm > self.max_mm:
            return 1
        return 0

    def classify(self, mm: float) -> str:
        return {-1: "Drought risk", 0: f"Good for {self.crop}", 1: "Waterlogging risk"}[self.score(mm)]

    def classify_region(self, region: Region) -> List[str]:
        return [self.classify(v) for v in region.rainfall]

    def annual_suitable(self, region: Region) -> Optional[bool]:
        if self.annual_range is None:
            return None
        lo, hi = self.annual_range
        return lo <= region.annual_total() <= hi


FAO_WATER = (
    "FAO (Critchley & Siegert 1991), Water Harvesting, AGL/MISC/17/91, ch. 2, Table 2 (seasonal "
    "crop water needs) & Table 6 (growing period): https://www.fao.org/3/U3160E/u3160e04.htm"
)
DEFAULT_CROPS: List[CropRule] = [
    # 500-800 mm over a 125-180 day season (about 4-6 months) -> roughly 100-200 mm/month.
    CropRule("maize", 100, 200, FAO_WATER + "; waterlogging: FAO Land & Water, Crop information: Maize"),
    # 300-500 mm over a 95-110 day season (about 3.5 months) -> roughly 85-145 mm/month.
    CropRule("beans", 85, 145, FAO_WATER),
    # Robusta: 1,200-1,800 mm/yr "well distributed over a period of 9 months" (UCDA).
    # min = 1200/12 = 100 mm/month; max = 1800/9 = 200 mm/month.
    CropRule(
        "coffee (Robusta)",
        100,
        200,
        "Uganda Coffee Development Authority, Robusta Coffee Handbook "
        "(https://greeninguganda.com/images/product/coffee/robusta.pdf)",
        annual_range=(1200, 1800),
    ),
]


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------
class SimilarityAnalyser:
    """Build pairwise similarity / distance matrices for a list of regions (composition)."""

    METRICS = ("cosine", "pearson", "euclidean")

    def __init__(self, regions: Sequence[Region]) -> None:
        if len(regions) < 2:
            raise ValueError("Need at least two regions to compare.")
        self.regions = list(regions)
        self.names = [r.name for r in self.regions]

    def pair(self, a: Region, b: Region, metric: str) -> float:
        if metric == "cosine":
            return cosine_similarity(a.rainfall, b.rainfall)
        if metric == "pearson":
            return float(np.corrcoef(a.rainfall, b.rainfall)[0, 1])
        if metric == "euclidean":
            return float(np.linalg.norm(a.rainfall - b.rainfall))
        raise ValueError(f"Unknown metric {metric!r}; choose from {self.METRICS}.")

    def matrix(self, metric: str) -> np.ndarray:
        n = len(self.regions)
        out = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                out[i, j] = self.pair(self.regions[i], self.regions[j], metric)
        return out


# ---------------------------------------------------------------------------
# Extension: real data from NASA POWER
# ---------------------------------------------------------------------------
NASA_URL = (
    "https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=PRECTOTCORR"
    "&community=AG&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON"
)


def fetch_nasa_power_monthly(lat: float, lon: float, start: int, end: int, timeout: int = 60) -> Dict[int, List[float]]:
    """Download monthly precipitation totals (mm) per year from the NASA POWER API.

    NASA POWER reports PRECTOTCORR as a daily *rate* (mm/day) for monthly data, so each value
    is multiplied by the number of days in that month. Returns {year: [12 monthly totals]}.
    """
    url = NASA_URL.format(lat=lat, lon=lon, start=start, end=end)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = json.load(resp)
    values = payload["properties"]["parameter"]["PRECTOTCORR"]
    units = payload.get("parameters", {}).get("PRECTOTCORR", {}).get("units", "mm/day")
    out: Dict[int, List[float]] = {}
    for year in range(start, end + 1):
        row = []
        for m in range(1, 13):
            v = float(values[f"{year}{m:02d}"])
            if v < 0:  # NASA uses -999 for missing
                v = float("nan")
            days = calendar.monthrange(year, m)[1]
            row.append(v * days if "day" in units.lower() else v)
        out[year] = row
    return out


def save_rainfall_history_csv(path: Union[str, Path], region: str, history: Dict[int, List[float]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["region", "year", *MONTHS])
        for year, vals in sorted(history.items()):
            w.writerow([region, year, *[round(v, 1) for v in vals]])
    return path


def load_rainfall_history_csv(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Read the CSV written above into {region: array of shape (years, 12)}."""
    data: Dict[str, List[List[float]]] = {}
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            data.setdefault(row["region"], []).append([float(row[m]) for m in MONTHS])
    if not data:
        raise ValueError(f"{path} has no rows.")
    return {k: np.array(v) for k, v in data.items()}
