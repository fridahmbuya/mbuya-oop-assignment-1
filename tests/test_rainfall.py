"""Tests for Mini-Project 4 (rainfall & crop suitability)."""
import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cosine_distance

from src.rainfall import (
    CropRule,
    Region,
    SimilarityAnalyser,
    cosine_similarity,
    load_rainfall_history_csv,
    save_rainfall_history_csv,
)

KAMPALA = [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130]
GULU = [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15]
MBARARA = [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90]


# --- normal cases -----------------------------------------------------------
def test_region_summaries():
    k = Region("Kampala", KAMPALA)
    assert k.annual_total() == 1600
    assert k.wettest_month() == ("May", 220)
    assert k.driest_month() == ("Sep", 60)
    assert k.coefficient_of_variation() == pytest.approx(np.std(KAMPALA, ddof=1) / np.mean(KAMPALA))


def test_cosine_matches_scipy_and_hand_example():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0)          # perpendicular
    assert cosine_similarity([1, 2], [2, 4]) == pytest.approx(1)          # same direction
    assert cosine_similarity(KAMPALA, GULU) == pytest.approx(1 - scipy_cosine_distance(KAMPALA, GULU))


def test_cosine_ignores_magnitude_but_euclidean_does_not():
    dry = Region("Dry", MBARARA)
    wet = Region("Wet", np.array(MBARARA) * 3)  # three times wetter, same shape
    sa = SimilarityAnalyser([dry, wet])
    assert sa.matrix("cosine")[0, 1] == pytest.approx(1.0)
    assert sa.matrix("pearson")[0, 1] == pytest.approx(1.0)
    assert sa.matrix("euclidean")[0, 1] > 0


def test_season_detection_matches_uganda_zones():
    assert Region("Gulu", GULU).regime() == "unimodal"        # northern Uganda: one long season
    assert Region("Mbarara", MBARARA).regime() == "bimodal"   # south-west: two seasons
    assert Region("Mbarara", MBARARA).rainy_season_peaks() == [3, 9]  # April and October


def test_wrap_around_finds_december_peak():
    rain = [150, 100, 50, 30, 30, 30, 30, 30, 30, 50, 100, 200]  # peak in December
    assert Region("South", rain).rainy_season_peaks() == [11]


def test_crop_rule_boundaries():
    maize = CropRule("maize", 100, 200, "test")
    assert maize.classify(99) == "Drought risk"
    assert maize.classify(100) == "Good for maize"
    assert maize.classify(200) == "Good for maize"
    assert maize.classify(201) == "Waterlogging risk"


def test_history_csv_round_trip(tmp_path):
    path = tmp_path / "hist.csv"
    save_rainfall_history_csv(path, "X", {2020: list(range(12)), 2021: list(range(1, 13))})
    loaded = load_rainfall_history_csv(path)
    assert loaded["X"].shape == (2, 12)


# --- edge cases -------------------------------------------------------------
def test_region_rejects_wrong_length_and_negatives():
    with pytest.raises(ValueError):
        Region("Short", [10, 20])
    with pytest.raises(ValueError):
        Region("Neg", [-1] + [10] * 11)


def test_cosine_rejects_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0, 0], [1, 2, 3])


def test_flat_rainfall_has_no_season():
    assert Region("Flat", [100] * 12).rainy_season_peaks() == []


def test_crop_rule_rejects_inverted_range():
    with pytest.raises(ValueError):
        CropRule("bad", 200, 100, "none")
