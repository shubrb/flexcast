"""
Offline development dataset — synthetic but schema-matched.

Generates data/raw-synthetic/... parquet files with the SAME shape the real
download produces, so every downstream engine (labeler, forecaster,
planner, replays, frontend) can be built and tested before the real
ERCOT download finishes — or anywhere without network.

The synthesis is physically sensible on purpose: summer evening peaks,
wind that dies in heat waves, prices that spike when net load is high.
That way label/forecast code "looks right" on it. It is NOT for slides.

    python -m engines.sample_data --years 2024 2026
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import synth_path

RNG = np.random.default_rng(7)
LOCS = ["HB_WEST", "HB_NORTH", "LZ_WEST", "LZ_NORTH"]
ZONES = ["coast", "east", "far_west", "north", "north_central",
         "south_central", "southern", "west"]


def _hours(year: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = min(pd.Timestamp(f"{year}-12-31 23:00", tz="UTC"),
              pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=2))
    return pd.date_range(start, end, freq="h")


def _system_shape(idx: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (load_mw, wind_mw, solar_mw) with realistic co-movement."""
    local = idx.tz_convert("America/Chicago")
    doy, hod = local.dayofyear.values, local.hour.values
    summer = np.exp(-0.5 * ((doy - 205) / 38.0) ** 2)          # July/Aug bump
    winter = 0.45 * np.exp(-0.5 * ((doy - 15) / 20.0) ** 2)
    diurnal = 0.55 + 0.45 * np.exp(-0.5 * ((hod - 17.5) / 3.2) ** 2)
    heat = summer * diurnal
    load = 42_000 + 40_000 * heat + 9_000 * winter * diurnal + RNG.normal(0, 1500, len(idx))
    # wind: anticorrelated with heat waves, stronger at night
    wind = (14_000 * (0.65 - 0.35 * summer) * (1.15 - 0.3 * np.sin((hod - 3) / 24 * 2 * np.pi))
            + RNG.normal(0, 2200, len(idx))).clip(500, 28_000)
    solar = (16_000 * np.maximum(0, np.sin((hod - 6.5) / 13.5 * np.pi)) * (0.7 + 0.3 * summer)
             + RNG.normal(0, 500, len(idx))).clip(0)
    return load.clip(30_000), wind, solar


def _prices(load, wind, solar):
    net = load - wind - solar
    q = (pd.Series(net).rank(pct=True)).values
    base = 22 + 60 * np.maximum(0, q - 0.55) / 0.45
    scarcity = np.where(q > 0.995, RNG.uniform(800, 4200, len(q)),
                np.where(q > 0.985, RNG.uniform(180, 900, len(q)), 0))
    rt = base + scarcity + RNG.normal(0, 6, len(q))
    dam = base + 0.35 * scarcity + RNG.normal(0, 5, len(q))  # smoother, partly anticipates
    return rt.clip(1), dam.clip(5), net


def gen_year(year: int) -> None:
    idx = _hours(year)
    load, wind, solar = _system_shape(idx)
    rt, dam, _net = _prices(load, wind, solar)

    # --- load (wide by weather zone + total, like the ERCOT archives)
    shares = np.array([0.30, 0.06, 0.06, 0.03, 0.17, 0.19, 0.12, 0.07])
    ldf = pd.DataFrame({"interval_start": idx})
    for z, s in zip(ZONES, shares):
        ldf[z] = load * s * RNG.normal(1, 0.015, len(idx))
    ldf["total"] = ldf[ZONES].sum(axis=1)
    ldf.to_parquet(synth_path("load", year), index=False)

    # --- DAM hourly SPP (long: one row per location per hour)
    dam_rows = []
    for loc in LOCS:
        mult = RNG.normal(1.0, 0.03)
        dam_rows.append(pd.DataFrame({
            "interval_start": idx, "location": loc,
            "location_type": "Trading Hub" if loc.startswith("HB") else "Load Zone",
            "market": "DAM", "spp": dam * mult}))
    pd.concat(dam_rows, ignore_index=True).to_parquet(synth_path("dam_spp", year), index=False)

    # --- RTM 15-min SPP
    idx15 = pd.date_range(idx[0], idx[-1] + pd.Timedelta(minutes=45), freq="15min", tz="UTC")
    rt15 = np.repeat(rt, 4)[: len(idx15)] * RNG.normal(1, 0.05, len(idx15))
    rtm_rows = []
    for loc in LOCS[:2]:
        rtm_rows.append(pd.DataFrame({
            "interval_start": idx15, "location": loc,
            "location_type": "Trading Hub", "market": "RTM",
            "spp": rt15 * RNG.normal(1.0, 0.02)}))
    pd.concat(rtm_rows, ignore_index=True).to_parquet(synth_path("rtm_spp", year), index=False)

    # --- ancillary prices (wide, hourly)
    pd.DataFrame({
        "interval_start": idx, "market": "DAM",
        "non-spinning_reserves": (0.08 * rt + RNG.normal(2, 1, len(idx))).clip(0.5),
        "regulation_up": (0.06 * rt + RNG.normal(3, 1, len(idx))).clip(0.5),
        "regulation_down": RNG.uniform(1, 8, len(idx)),
        "responsive_reserves": (0.09 * rt + RNG.normal(3, 1, len(idx))).clip(0.5),
        "ercot_contingency_reserve_service": (0.05 * rt + RNG.normal(2, 1, len(idx))).clip(0.3),
    }).to_parquet(synth_path("as_prices", year), index=False)

    # --- weather: actuals + "archived forecast" (= actual + horizon noise)
    local = idx.tz_convert("America/Chicago")
    summer = np.exp(-0.5 * ((local.dayofyear.values - 205) / 38.0) ** 2)
    temp = 12 + 26 * summer + 8 * np.sin((local.hour.values - 15) / 24 * 2 * np.pi) + RNG.normal(0, 2, len(idx))
    ws = (wind / 28_000) * 22 + RNG.normal(0, 1.2, len(idx))
    rad = solar / 16_000 * 900
    frames = []
    for kind, noise in [("actual", 0.0), ("forecast", 1.0)]:
        frames.append(pd.DataFrame({
            "interval_start": idx,
            "temperature_2m": temp + RNG.normal(0, 1.3 * noise, len(idx)),
            "wind_speed_100m": (ws + RNG.normal(0, 1.8 * noise, len(idx))).clip(0),
            "shortwave_radiation": (rad + RNG.normal(0, 60 * noise, len(idx))).clip(0),
            "kind": kind}))
    pd.concat(frames, ignore_index=True).to_parquet(synth_path("weather", year), index=False)

    print(f"[sample] generated {year}: {len(idx):,} hours across 5 datasets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"), default=[2024, 2026])
    a = ap.parse_args()
    for y in range(a.years[0], a.years[1] + 1):
        gen_year(y)
    print("[sample] done — downstream engines can now run offline. "
          "Replace with the real download before computing slide numbers.")


if __name__ == "__main__":
    main()
