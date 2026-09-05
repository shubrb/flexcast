"""
Phase 7 helper — the grid-map snapshot.

Writes data/out/grid.json: the latest known load in each ERCOT weather zone,
each zone's trailing-year peak (so the map can show "how hard is this region
working right now"), the statewide total, and the demo site's location.
Reads only the local cache — no network, safe to re-run anytime.

    python -m engines.gridsnap [--synthetic]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from .config import OUT, RAW, RAW_SYNTH, load_site

ZONES = ["coast", "east", "far_west", "north", "north_central",
         "south", "south_central", "southern", "west"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Snapshot grid-map data for the console")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--site", default=None)
    a = ap.parse_args()
    root = RAW_SYNTH if a.synthetic else RAW
    tag = "_synthetic" if a.synthetic else ""

    files = sorted((root / "load").glob("[0-9]*.parquet"))[-2:]
    if not files:
        raise SystemExit(f"[gridsnap] no load cache under {root}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["interval_start"] = pd.to_datetime(df["interval_start"], utc=True)
    df = df.sort_values("interval_start")
    total_col = "ercot" if "ercot" in df.columns else "total"
    zcols = [z for z in ZONES if z in df.columns]
    last = df.dropna(subset=[total_col]).iloc[-1]
    year = df[df["interval_start"] >= last["interval_start"] - pd.Timedelta(days=365)]

    site = load_site(a.site)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of_local": last["interval_start"].tz_convert("America/Chicago").isoformat(),
        "total_mw": round(float(last[total_col])),
        "total_peak_mw": round(float(year[total_col].max())),
        "zones": {z: {"mw": round(float(last[z])), "peak_mw": round(float(year[z].max()))}
                  for z in zcols if pd.notna(last[z])},
        "site": {
            "name": site["site"]["name"],
            "zone": site["site"]["weather_zone"].lower(),
            "hub": site["site"]["price_location"],
            "total_mw": site["power"]["total_mw"],
        },
    }
    path = OUT / f"grid{tag}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"[gridsnap] {payload['total_mw']:,} MW statewide as of "
          f"{payload['as_of_local'][:16]} ({len(payload['zones'])} zones) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
