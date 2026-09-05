"""
Phase 1 — build the grid's memory.

Downloads ERCOT history (via gridstatus, keyless) and site weather
(via Open-Meteo, keyless) into data/raw/<dataset>/<year>.parquet.

Resumable: a year that already has a parquet file is skipped, so you can
re-run after any failure and it continues where it left off. Run it the
first prep evening — it's the slowest step of the whole project.

Usage:
    python -m engines.download --years 2021 2026            # everything
    python -m engines.download --years 2024 2026 --datasets dam_spp rtm_spp
    python -m engines.download --dry-run --years 2021 2026  # print the plan only

Datasets:
    load       hourly system + weather-zone load (ERCOT load archives)
    dam_spp    day-ahead hourly settlement point prices, hubs + zones (2011+)
    rtm_spp    real-time 15-min settlement point prices, hubs + zones (2011+)
    as_prices  day-ahead ancillary clearing prices (recent ~30-day window —
               ERCOT's public archive expires these documents fast)
    weather    Open-Meteo actuals + archived FORECASTS for the site location
    renewables (optional, recent window only) hourly wind/solar reports
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import traceback
import zipfile

import certifi
import os as _os
_os.environ.setdefault("SSL_CERT_FILE", certifi.where())
_os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import pandas as pd
import requests

from .config import load_site, raw_path

UTC = "UTC"


# ---------------------------------------------------------------- helpers
def _log(msg: str) -> None:
    print(f"[download] {msg}", flush=True)


def _normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize on a UTC 'interval_start' column, keep everything else."""
    candidates = ["Interval Start", "interval_start", "Time", "time", "Date", "date"]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        raise ValueError(f"no time column found in {list(df.columns)[:8]}")
    out = df.copy()
    ts = pd.to_datetime(out[col], utc=False)
    # tz-aware -> convert; naive -> assume ERCOT local (America/Chicago)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(UTC)
    else:
        ts = ts.dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT").dt.tz_convert(UTC)
    # Drop the source time column BEFORE adding the normalized one — otherwise
    # lowercasing turns "Interval Start" + "interval_start" into duplicates.
    out = out.drop(columns=[col])
    out["interval_start"] = ts
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    # Defensive: if any duplicate labels remain (e.g. "Time" alongside "time"),
    # keep the first occurrence so label-based ops never see duplicates.
    out = out.loc[:, ~out.columns.duplicated()]
    out = out.dropna(subset=["interval_start"])
    return out.sort_values("interval_start").reset_index(drop=True)


def _save(df: pd.DataFrame, dataset: str, year: int) -> None:
    path = raw_path(dataset, year)
    df.to_parquet(path, index=False)
    _log(f"  saved {dataset}/{year}: {len(df):,} rows -> {path}")


def _retry(fn, tries: int = 3, wait: float = 20.0):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — we want to survive anything and retry
            if i == tries - 1:
                raise
            _log(f"  attempt {i+1} failed ({type(e).__name__}: {str(e)[:120]}) — retrying in {wait:.0f}s")
            time.sleep(wait)


# ---------------------------------------------------------------- ERCOT datasets
def _load_fallback(ercot, year: int) -> pd.DataFrame:
    """ERCOT sometimes updates the current-year Native_Load zip in place,
    leaving a member whose stored CRC doesn't match its bytes. Refetch and
    read the workbook with the CRC check disabled — openpyxl still fails
    loudly if the bytes are genuinely corrupt, so this can't silently
    produce garbage."""
    from bs4 import BeautifulSoup

    page = requests.get("https://www.ercot.com/gridinfo/load/load_hist", timeout=120)
    page.raise_for_status()
    soup = BeautifulSoup(page.content, "html.parser")
    link = None
    for a in soup.find_all("a"):
        href = a.get("href", "")
        if str(year) in a.get_text(strip=True) and (".zip" in href.lower() or ".xls" in href.lower()):
            link = href
            break
    if link is None:
        raise ValueError(f"no load archive link found for {year}")
    blob = requests.get(link, timeout=300)
    blob.raise_for_status()
    if link.lower().endswith(".zip"):
        zf = zipfile.ZipFile(io.BytesIO(blob.content))
        member = next(n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls")))
        with zf.open(member) as fh:
            fh._expected_crc = None  # skip CRC verification on the flaky member
            raw = fh.read()
    else:
        raw = blob.content
    df = pd.read_excel(io.BytesIO(raw))
    return _normalize_time(ercot._process_post_settlements_load_data(df))


def fetch_load(ercot, year: int) -> pd.DataFrame:
    # ERCOT load archives: one call per year
    try:
        return _normalize_time(ercot.get_hourly_load_post_settlements(f"{year}-01-01"))
    except zipfile.BadZipFile as e:
        _log(f"  ERCOT's {year} load zip failed its CRC check ({e}) — "
             "refetching with the CRC check disabled")
        return _load_fallback(ercot, year)


def fetch_dam_spp(ercot, year: int) -> pd.DataFrame:
    # Historical DAM SPPs for all hubs + load zones, whole year (2011+)
    return _normalize_time(ercot.get_dam_spp(year))


def _clean_rtm_frames(sheets: dict) -> pd.DataFrame:
    """The current-year RTM workbook ships one sheet per month, and the
    future months are empty. Concatenating them poisons 'Delivery Hour'
    into object dtype, which crashes gridstatus's parser on pandas>=2
    (astype('timedelta64[h]') no longer accepts object). Clean per sheet,
    then force numeric dtypes so parse_doc sees what it expects."""
    key = ["Delivery Hour", "Delivery Interval", "Settlement Point Price"]
    frames = []
    for s in sheets.values():
        s = s.rename(columns=lambda c: str(c).strip())
        if not set(key).issubset(s.columns):
            continue
        s = s.dropna(how="all")
        if len(s):
            frames.append(s)
    if not frames:
        raise ValueError("workbook has no data sheets")
    df = pd.concat(frames, ignore_index=True).dropna(subset=key, how="any")
    for c in ("Delivery Hour", "Delivery Interval"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Delivery Hour", "Delivery Interval"])
    df["Delivery Hour"] = df["Delivery Hour"].astype("int64")
    df["Delivery Interval"] = df["Delivery Interval"].astype("Int64")
    return df


def _rtm_spp_fallback(ercot, year: int) -> pd.DataFrame:
    """Same document gridstatus fetches, our cleaning, then gridstatus's
    own time/DST parsing and schema finalization — so the output matches
    the completed-year files exactly."""
    from gridstatus import Markets
    from gridstatus import utils as gs_utils
    from gridstatus.ercot import HISTORICAL_RTM_LOAD_ZONE_AND_HUB_PRICES_RTID

    doc = ercot._get_document(
        report_type_id=HISTORICAL_RTM_LOAD_ZONE_AND_HUB_PRICES_RTID,
        constructed_name_contains=f"{year}.zip",
        verbose=False,
    )
    sheets = pd.read_excel(gs_utils.get_zip_file(doc.url, verbose=False), sheet_name=None)
    df = ercot.parse_doc(_clean_rtm_frames(sheets), verbose=False)
    return ercot._finalize_spp_df(df, market=Markets.REAL_TIME_15_MIN, verbose=False)


def fetch_rtm_spp(ercot, year: int) -> pd.DataFrame:
    # Historical RTM 15-min SPPs for all hubs + load zones, whole year (2011+)
    try:
        return _normalize_time(ercot.get_rtm_spp(year))
    except (ValueError, TypeError) as e:
        _log(f"  gridstatus parser failed on the {year} RTM workbook "
             f"({type(e).__name__}: {str(e)[:90]}) — using fallback parser")
        return _normalize_time(_rtm_spp_fallback(ercot, year))


def fetch_as_prices(ercot, _year: int) -> pd.DataFrame:
    """ERCOT's archive for this report keeps only ~30 days of documents, so
    ancillary prices are a recent-window nice-to-have, not deep history."""
    end = pd.Timestamp.now(tz="America/Chicago").normalize() - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=25)
    return _normalize_time(ercot.get_as_prices(start.date(), end=end.date()))


def fetch_renewables_recent(ercot, _year: int) -> pd.DataFrame:
    """Hourly wind + solar reports keep only a short archive window —
    grab whatever is available and merge. Optional; the labeler degrades
    gracefully to load+price labels without it."""
    end = pd.Timestamp.now(tz="America/Chicago").normalize()
    start = end - pd.Timedelta(days=55)
    wind = _normalize_time(ercot.get_wind_actual_and_forecast_hourly(start.date(), end=end.date()))
    solar = _normalize_time(ercot.get_solar_actual_and_forecast_hourly(start.date(), end=end.date()))
    wind["kind"], solar["kind"] = "wind", "solar"
    return pd.concat([wind, solar], ignore_index=True)


# ---------------------------------------------------------------- weather (Open-Meteo)
def fetch_weather(site: dict, year: int) -> pd.DataFrame:
    """Actual weather AND archived forecasts for the site location.

    - archive-api: what actually happened (for labels/diagnostics)
    - historical-forecast-api: what the FORECAST said at the time
      (for honest as-of-time features in the 2-7 day model)
    """
    lat, lon = site["site"]["lat"], site["site"]["lon"]
    # Open-Meteo rejects future end dates — clamp the current year to ~2 days ago
    last = min(pd.Timestamp(f"{year}-12-31"),
               pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=2))
    end_date = last.strftime("%Y-%m-%d")
    frames = []
    for kind, host in [
        ("actual", "https://archive-api.open-meteo.com/v1/archive"),
        ("forecast", "https://historical-forecast-api.open-meteo.com/v1/forecast"),
    ]:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year}-01-01",
            "end_date": end_date,
            "hourly": "temperature_2m,wind_speed_100m,shortwave_radiation",
            "timezone": "UTC",
        }
        r = requests.get(host, params=params, timeout=120)
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame(h).rename(columns={"time": "interval_start"})
        df["interval_start"] = pd.to_datetime(df["interval_start"], utc=True)
        df["kind"] = kind
        frames.append(df)
        time.sleep(2)  # be polite
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- main
DATASETS = {
    "load": fetch_load,
    "dam_spp": fetch_dam_spp,
    "rtm_spp": fetch_rtm_spp,
    "as_prices": fetch_as_prices,
    "weather": None,  # handled separately (needs site, not ercot)
    "renewables": fetch_renewables_recent,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="FlexCast Phase-1 data cache")
    ap.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"), required=True)
    ap.add_argument("--datasets", nargs="+", default=["load", "dam_spp", "rtm_spp", "as_prices", "weather"],
                    choices=list(DATASETS))
    ap.add_argument("--dry-run", action="store_true", help="print the plan, download nothing")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch the LAST requested year even if cached — run before "
                         "the demo so the current year isn't weeks stale")
    args = ap.parse_args()

    years = list(range(args.years[0], args.years[1] + 1))
    site = load_site()

    recent_only = ("renewables", "as_prices")  # ERCOT keeps ~30 days for these
    plan = [(d, y) for d in args.datasets for y in years
            if not (d in recent_only and y != years[-1])]
    todo = [(d, y) for d, y in plan
            if not raw_path(d, y).exists() or (args.refresh and y == years[-1])]
    _log(f"plan: {len(plan)} dataset-years ({len(plan)-len(todo)} already cached, {len(todo)} to fetch)")
    for d, y in plan:
        state = ("SKIP (cached)" if (d, y) not in todo
                 else "REFRESH" if raw_path(d, y).exists() else "FETCH")
        _log(f"  {state:14s} {d}/{y}")
    if args.dry_run:
        return 0

    import gridstatus  # imported here so --dry-run works anywhere
    ercot = gridstatus.Ercot()

    failures = []
    for d, y in todo:
        _log(f"fetching {d}/{y} ...")
        try:
            if d == "weather":
                df = _retry(lambda: fetch_weather(site, y))
            else:
                df = _retry(lambda: DATASETS[d](ercot, y))
            _save(df, d, y)
        except Exception:
            failures.append((d, y))
            _log(f"  FAILED {d}/{y} — continuing (re-run later to resume)")
            traceback.print_exc(limit=1)
        time.sleep(3)  # be polite to ERCOT between yearly zips

    _log(f"done. {len(todo)-len(failures)} fetched, {len(failures)} failed"
         + (f" -> re-run to retry: {failures}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
