"""Download 10 years of real monthly rainfall for the three regions (Mini-Project 4 extension).

Run once from the repository root, with an internet connection:

    python scripts/fetch_nasa_rainfall.py

It writes data/nasa_power_rainfall.csv, which project4_rainfall.ipynb picks up automatically.
Source: NASA POWER (https://power.larc.nasa.gov), parameter PRECTOTCORR (bias-corrected
precipitation), community AG.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rainfall import fetch_nasa_power_monthly, save_rainfall_history_csv  # noqa: E402

LOCATIONS = {  # (latitude, longitude) of the town centres
    "Kampala": (0.3476, 32.5825),
    "Gulu": (2.7724, 32.2881),
    "Mbarara": (-0.6072, 30.6545),
}
START, END = 2014, 2023
OUT = Path(__file__).resolve().parents[1] / "data" / "nasa_power_rainfall.csv"

if __name__ == "__main__":
    if OUT.exists():
        OUT.unlink()
    for name, (lat, lon) in LOCATIONS.items():
        print(f"Fetching {name} {START}-{END} ...")
        save_rainfall_history_csv(OUT, name, fetch_nasa_power_monthly(lat, lon, START, END))
    print("Saved", OUT)
