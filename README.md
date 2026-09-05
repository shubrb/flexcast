# FlexCast

The seven-day forecast and flight plan for power-flexible datacenters.
This repo starts at **Phase 0–1**: setup + the data layer. Later engines
(labeler, forecaster, planner, replay) build on the cache this creates.

## Quickstart (each teammate, ~5 min + download time)

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) See the download plan (no network needed):
python -m engines.download --dry-run --years 2021 2026

# 2) Run the real download (needs normal internet; NOT a corporate proxy).
#    Start it the FIRST prep evening — yearly zips take a while. Safe to
#    Ctrl-C and re-run anytime: finished years are skipped.
python -m engines.download --years 2021 2026

# 3) Verify the cache:
python -m engines.check

# 4) Label history + train/evaluate the forecaster (Phases 2-3):
python -m engines.labeler                # stress/calm + 4CP ground truth
python -m engines.forecast               # train + honest eval vs naive baselines
python -m engines.forecast --predict     # data/out/forecast.json (next 7 days)

# 5) Site model + weekly flight plan + backtest (Phases 4-6):
python -m engines.site
python -m engines.planner                # data/out/plan.json from the forecast
python -m engines.replay                 # 20-month backtest -> data/out/replay.json
python -m engines.gridsnap               # data/out/grid.json for the console's map

# 6) The console (Phase 7) — the FlexCast server (static app + live optimizer):
python -m engines.serve                  # then visit http://localhost:8000/app/
#    (?synthetic=1 shows the dev-cache version; plain `python -m http.server`
#     also works but the "Design your own site" live solver needs engines.serve)

# 7) Live ticker (run on a loop during the demo; survives wifi loss):
python -m engines.live

# Before the demo: re-pull the newest weeks of the current year, then re-run
# labeler + forecast --predict so features aren't stale:
python -m engines.download --years 2021 2026 --refresh
```

**Can't download yet / on a plane?** Generate the schema-matched synthetic
dataset and keep building — every later engine runs on it unchanged:

```bash
python -m engines.sample_data --years 2024 2026
python -m engines.check
```

Synthetic data is for development only — recompute every slide number on
the real cache.

## What gets downloaded (all keyless)

| dataset    | source                              | granularity | depth      |
|------------|-------------------------------------|-------------|------------|
| load       | ERCOT load archives (by weather zone)| hourly     | years      |
| dam_spp    | ERCOT historical DAM prices (hubs+zones) | hourly | 2011+      |
| rtm_spp    | ERCOT historical RTM prices         | 15-min      | 2011+      |
| as_prices  | ERCOT ancillary clearing prices     | hourly      | ~30 days (ERCOT retention) |
| weather    | Open-Meteo actuals + **archived forecasts** for the site | hourly | 2016+ (forecasts ~2021+) |
| renewables | ERCOT hourly wind/solar reports     | hourly      | recent window only (optional) |

Everything lands in `data/raw/<dataset>/<year>.parquet` with a UTC
`interval_start` column and snake_case names (see `_normalize_time`).

## Conventions (agree once, never debug twice)

- **Storage timezone is UTC.** Convert to `America/Chicago` only for
  analysis/plots. DST bugs are the classic way this project dies.
- **The site is `site.yaml`.** One demo site; every engine reads it.
  It also holds label thresholds, the flex supply curve, and the job queue.
- **Resumability over speed.** Any script can be killed and re-run.
- **`data/out/` is engine output, `data/live/` is the ticker.** The
  frontend only ever reads JSON from those two places.

## Known sharp edges

- `weather` uses two Open-Meteo endpoints; the **historical-forecast**
  one is the honesty-critical piece (features must be what was knowable
  at the time). If a year 404s, that's fine — forecast archive starts
  around 2021; the day-ahead-price model doesn't need it.
- `renewables` (wind/solar hourly reports) and `as_prices` (ancillary
  clearing prices) only reach back a few weeks — ERCOT's public archive
  expires those documents. Expected; the labeler's deep history comes
  from load + DAM/RTM prices, which do go back years.
- If ERCOT throttles you, the script backs off and retries; worst case
  re-run tomorrow — cached years are skipped.

## Roadmap (from the playbook)

- ✅ Phase 2 `engines/labeler.py` — mark every historical hour stress/calm + 4CP stream
- ✅ Phase 3 `engines/forecast.py` — two-horizon model + calibration + drivers
  (day-ahead uses archived weather forecasts; week-ahead uses climatology —
  honest about what's knowable at each horizon)
- Phase 4–5 `engines/site.py`, `engines/planner.py` — flex supply curve, weekly plan, dispatch
- ✅ Phase 6 `engines/replay.py` — weekly rolling backtest on the held-out
  years, scored on ACTUAL prices + actual coincident peaks; three strategies
  (flat / plan / plan+dispatch) -> data/out/replay.json scoreboard
- ✅ Phase 7 `app/index.html` — the console: tide chart + plan + battery on one
  shared-cursor timeline, replay scoreboard + scrubber, CP table, flex ladder.
  Single file, zero dependencies, offline-safe; reads only data/out + data/live.
