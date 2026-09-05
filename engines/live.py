"""
Phase 1 — the live ticker (the demo's ONLY live dependency, by design).

Fetches "right now" from ERCOT: current load, fuel mix, and today's
day-ahead hub price curve. Writes data/live/live.json for the frontend.

Failure-proof: if any call fails (venue wifi...), the previous JSON is
kept and re-stamped with {"stale": true} so the UI can show a subtle
"cached" badge instead of breaking. Run it on a loop during the demo:

    watch -n 300 python -m engines.live      # every 5 minutes
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from .config import LIVE, load_site

OUT = LIVE / "live.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_payload() -> dict:
    import gridstatus

    ercot = gridstatus.Ercot()
    site = load_site()
    hub = site["site"]["price_location"]

    load = ercot.get_load("today")
    fuel = ercot.get_fuel_mix("today")
    dam = ercot.get_spp("today", market="DAY_AHEAD_HOURLY", location_type="Trading Hub")

    latest_load = load.dropna(subset=["Load"]).iloc[-1]
    fuel_row = fuel.iloc[-1].to_dict()
    fuel_mix = {k: float(v) for k, v in fuel_row.items()
                if isinstance(v, (int, float)) and str(k).lower() not in ("time",)}

    hub_col = "Location" if "Location" in dam.columns else "location"
    spp_col = "SPP" if "SPP" in dam.columns else "spp"
    dam_hub = dam[dam[hub_col] == hub]
    price_curve = [
        {"hour": int(pd_ts.hour), "usd_mwh": float(p)}
        for pd_ts, p in zip(pd_to_local(dam_hub), dam_hub[spp_col])
    ]

    return {
        "fetched_at": _now(),
        "stale": False,
        "load_mw": float(latest_load["Load"]),
        "fuel_mix_mw": fuel_mix,
        "dam_price_today": price_curve,
        "hub": hub,
    }


def pd_to_local(df):
    import pandas as pd  # noqa: F401
    col = "Interval Start" if "Interval Start" in df.columns else "interval_start"
    return [t.tz_convert("America/Chicago") for t in df[col]]


def main() -> int:
    try:
        payload = build_payload()
        OUT.write_text(json.dumps(payload, indent=2))
        print(f"[live] ok — load {payload['load_mw']:.0f} MW, wrote {OUT}")
        return 0
    except Exception as e:  # noqa: BLE001 — never let the ticker kill a demo
        if OUT.exists():
            payload = json.loads(OUT.read_text())
            payload["stale"] = True
            payload["stale_since"] = _now()
            OUT.write_text(json.dumps(payload, indent=2))
            print(f"[live] fetch failed ({type(e).__name__}) — kept cached payload, marked stale")
            return 0
        print(f"[live] fetch failed and no cache exists yet: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
