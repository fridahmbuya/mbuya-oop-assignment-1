# OOP with Python: Mini-Projects (Advent 2026)

**Student:** _[MBUYA CHRISTINE FRIDAH, S26M19/029]_  **Programme:** _[MSDS]_  **Course:** OOP with Python

Five mini-projects set in a Ugandan context. Each one models its domain with classes, implements its methods rather than using black-box libraries, is unit-tested, and ends with a *Findings & Limitations* section.

| # | Notebook | Module | Topic |
|---|---|---|---|
| 1 | `project1_population.ipynb` | `src/population.py` | District population forecasting and classroom planning |
| 2 | `project2_microgrid.ipynb` | `src/microgrid.py` | Solar/battery dispatch as a linear system |
| 3 | `project3_fishstock.ipynb` | `src/fishstock.py` | Logistic fish stock, price random walk, CV risk and VaR |
| 4 | `project4_rainfall.ipynb` | `src/rainfall.py` | Rainfall regimes, similarity measures, crop suitability |
| 5 | `project5_taxi.ipynb` | `src/taxi.py` | Route revenue, fare equilibrium, forecasting, fleet sizing |

Shared code used by Projects 1 and 5 (abstract `Forecaster`, linear trend, MAE/RMSE/MAPE, walk-forward backtest, bootstrap intervals) lives in `src/common/forecasting.py`.

## Repository layout

```
├── README.md
├── requirements.txt
├── project1_population.ipynb … project5_taxi.ipynb
├── src/
│   ├── common/forecasting.py
│   ├── population.py  microgrid.py  fishstock.py  rainfall.py  taxi.py
├── tests/                  pytest unit tests (one file per module)
├── data/                   generated demand CSV (Project 2); optional NASA rainfall CSV (Project 4)
├── figures/                PNG copies of every chart (written by the notebooks)
└── scripts/fetch_nasa_rainfall.py   downloads real rainfall for the Project 4 extension
```

## Setup and running

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -q                            # run all unit tests
jupyter notebook                     # open any notebook -> Kernel -> Restart & Run All
```

Run the notebooks from the repository root (they import from `src/`). Every notebook runs top to bottom without errors. All randomness uses `np.random.default_rng(seed)` with fixed seeds, so results are reproducible. Timings in Project 2 vary slightly between machines.

To execute all notebooks headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace project*.ipynb
```

**Project 4 extension (optional, needs internet):** `python scripts/fetch_nasa_rainfall.py` downloads 2014–2023 monthly rainfall from NASA POWER into `data/nasa_power_rainfall.csv`. Re-running `project4_rainfall.ipynb` then adds box plots of year-to-year variability. Without the file, the notebook skips that step and prints instructions instead of failing.

## Summary of findings

**1. Population forecaster.** Four of five districts grow multiplicatively, so an exponential (CAGR) model beat a linear trend on the 2022–2024 hold-out (MAPE < 1 % for Kampala, Wakiso, Mukono). Mbarara grows linearly. Wakiso grows fastest (CAGR 6.47 %). By 2029 the districts need roughly 2,090 (Wakiso), 1,550 (Kampala), 930 (Mukono), 410 (Gulu) and 230 (Mbarara) extra classrooms, with 95 % bootstrap ranges. The Fibonacci-ratio model was wrong by 78–91 % because it ignores the data.

**2. Solar micro-grid.** The system is well-posed (det = −5, κ ≈ 5.83), but a day is only physically feasible when 0.5·D1 ≤ D2 ≤ 1.33·D1. Six of 30 synthetic days broke this and were repaired with non-negative least squares, leaving small amounts of unmet demand. The battery is the volatile (CV 0.83 vs 0.16) and expensive source: 27 % of energy, 52 % of the ~UGX 308,000 monthly cost. Vectorised solving is about 15–20× faster than a loop.

**3. Fish stock and risk.** The sustainable optimum is h = 0.20, which gives MSY = rK/4 = 1,000 t/week at a stock of 5,000 t. h = 0.30 overfishes, yielding the same 750 t/week as h = 0.10 from a third of the stock. The old "variance > 50,000" rule is unit-dependent and meaningless. CV classifies every scenario as Moderate risk, and the 5 % annual VaR at h = 0.10 is ≈ UGX 80 bn (18 % of mean). CV cannot detect biological risk, so it must be paired with the MSY check. An 8-week closed season raises 5-year revenue only when overfishing (+7 % at h = 0.30).

**4. Rainfall and crops.** Cosine similarity (implemented correctly and verified against SciPy) rates all regions as similar because it ignores amount. Pearson reveals that Kampala's and Gulu's wet months don't coincide. Peak detection with a circular year and a prominence threshold classifies Gulu as unimodal and Mbarara as bimodal, matching Uganda's climate zones. Kampala's illustrative data show only one clear season. Advisory: Mbarara farmers should grow beans, planting mid-February to early March and late August to early September.

**5. Taxi routes.** The Ntinda fare (UGX 2,000) is below the UGX 2,200 equilibrium, causing a shortage of 10 passengers per trip-hour. In walk-forward backtesting, tuned SES (α = 1, i.e. "tomorrow = today") beat the 3-day moving average on every route. One 14-seater per route covers day-11 demand with a 15 % buffer, which suggests the counts are per vehicle or per trip-hour. With a weekly pattern, a seasonal-naïve forecaster has about 4× lower error than the moving average.

## Data sources and assumptions

- District populations, rainfall, micro-grid demand structure, taxi counts and prices are **illustrative**, as given in the brief. Mukono and Mbarara population series (Project 1) and all micro-grid demand values (Project 2) are synthetic and were created for this assignment.
- Crop water requirements: FAO, *Water Harvesting* (Critchley & Siegert, 1991), ch. 2, Tables 2 and 6, <https://www.fao.org/3/U3160E/u3160e04.htm>; FAO Land & Water, *Crop Information: Maize*; Uganda Coffee Development Authority, *Robusta Coffee Handbook*.
- Optional real rainfall: NASA POWER API (<https://power.larc.nasa.gov>), parameter `PRECTOTCORR`.

## Testing

`tests/` contains one test file per module (74 tests). They cover hand-calculated normal cases (e.g. CAGR, the 2×2 solution, MSY, equilibrium fare) and edge cases (empty input, zero, negative values, singular matrices, invalid parameters).

## AI use declaration

>
>
> An AI assistant (Claude, by Anthropic) was used to help produce this repository. It drafted the class designs, module code, unit tests, notebooks and the first drafts of the written analysis. I reviewed all of the code, ran the tests and notebooks myself, checked the key results by hand (see the "hand check" cells), and rewrote the Findings & Limitations sections in my own words.I can explain every part of the code.
