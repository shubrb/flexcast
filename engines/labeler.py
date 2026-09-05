"""
Phase 2 — the labeler.

Walks every cached hour and stamps ground truth on it:

  is_stress          the hours a flexible site should have acted in, per the
                     declared, tunable definition in site.yaml (labels:) —
                     price spike at the site hub, OR top price tail of the
                     year, OR top load tail of the year
  is_4cp_actual      the hour that WAS the ERCOT monthly system peak in each
                     4CP-season month (hourly approximation of the 15-min CP)
  is_4cp_candidate   causal "be nervous now" flag: a 4CP-season afternoon
                     hour at/near the month-to-date load peak — what a real
                     operator could have known AT THE TIME

Honest simplification (v1): site.yaml calls the load rule net_load_top_pct.
True net load (load minus wind/solar) needs renewables history that ERCOT's
public archive doesn't retain, so total system load is the proxy here.

Outputs (source-separated so synthetic never contaminates real):
  data/out/labels[_synthetic]/<year>.parquet     one row per hour
  data/out/label_summary[_synthetic].json        for the frontend + humans

    python -m engines.labeler                # real cache, all years found
    python -m engines.labeler --synthetic    # dev cache
    python -m engines.labeler --years 2021 2026
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import OUT, RAW, RAW_SYNTH, load_site

LOCAL_TZ = "America/Chicago"     # storage stays UTC; calendar logic is local
FOUR_CP_NEAR = 0.96              # candidate when load >= 96% of month-to-date peak
FOUR_CP_WINDOW = (13, 20)        # local hours with real peak risk (ERCOT CPs land ~14-19)


# ---------------------------------------------------------------- loading

def _read_years(root, dataset: str, years: list[int]) -> pd.DataFrame | None:
    frames = []
    for y in years:
        p = root / dataset / f"{y}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["interval_start"] = pd.to_datetime(df["interval_start"], utc=True)
    return df


def build_hourly(root, years: list[int], site: dict) -> pd.DataFrame:
    """One row per UTC hour: hub prices + system/zone load.

    Phase 3 reuses this join as its feature spine — keep it engine-agnostic.
    """
    hub = site["site"]["price_location"]
    zone = site["site"]["weather_zone"].lower()

    rtm = _read_years(root, "rtm_spp", years)
    if rtm is None:
        raise SystemExit(f"[labeler] no rtm_spp parquet under {root} — run engines.download first")
    at_hub = rtm[rtm["location"] == hub]
    if at_hub.empty:
        raise SystemExit(f"[labeler] hub {hub!r} not in rtm_spp; available: "
                         f"{sorted(rtm['location'].unique())}")
    hour = at_hub["interval_start"].dt.floor("h")
    hourly = at_hub.groupby(hour)["spp"].agg(rt_price_mean="mean", rt_price_max="max")

    dam = _read_years(root, "dam_spp", years)
    if dam is not None:
        da = (dam[dam["location"] == hub]
              .groupby(dam["interval_start"].dt.floor("h"))["spp"].mean()
              .rename("da_price"))
        hourly = hourly.join(da, how="outer")

    load = _read_years(root, "load", years)
    if load is not None:
        total_col = "ercot" if "ercot" in load.columns else "total"
        cols = {total_col: "load_mw"}
        if zone in load.columns:
            cols[zone] = "zone_load_mw"
        ld = (load.groupby("interval_start")[list(cols)].mean().rename(columns=cols))
        hourly = hourly.join(ld, how="outer")
    for c in ("da_price", "load_mw", "zone_load_mw"):
        if c not in hourly.columns:
            hourly[c] = np.nan

    hourly.index.name = "interval_start"
    return hourly.sort_index().reset_index()


# ---------------------------------------------------------------- labels

def apply_stress(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stress = price spike OR yearly price tail OR yearly load tail."""
    spike_usd = float(cfg.get("rt_price_spike_usd", 200))
    price_pct = float(cfg.get("rt_price_top_pct", 0.005))
    load_pct = float(cfg.get("net_load_top_pct", 0.01))

    local_year = df["interval_start"].dt.tz_convert(LOCAL_TZ).dt.year
    df["stress_price_spike"] = df["rt_price_max"] >= spike_usd
    rt_cut = df.groupby(local_year)["rt_price_mean"].transform(
        lambda s: s.quantile(1 - price_pct))
    df["stress_price_tail"] = df["rt_price_mean"] >= rt_cut
    load_cut = df.groupby(local_year)["load_mw"].transform(
        lambda s: s.quantile(1 - load_pct))
    df["stress_load_tail"] = df["load_mw"] >= load_cut
    for c in ("stress_price_spike", "stress_price_tail", "stress_load_tail"):
        df[c] = df[c].fillna(False).astype(bool)
    df["is_stress"] = df[["stress_price_spike", "stress_price_tail",
                          "stress_load_tail"]].any(axis=1)
    return df


def apply_four_cp(df: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    """Hindsight monthly-peak flag + causal near-peak candidate flag."""
    local = df["interval_start"].dt.tz_convert(LOCAL_TZ)
    df["_y"], df["_m"], hod = local.dt.year, local.dt.month, local.dt.hour
    in_season = df["_m"].isin(months) & df["load_mw"].notna()

    df["is_4cp_actual"] = False
    season = df[in_season]
    if not season.empty:
        peak_idx = season.groupby(["_y", "_m"])["load_mw"].idxmax()
        df.loc[peak_idx, "is_4cp_actual"] = True

    # Month-to-date running peak (inclusive: a NEW monthly peak is exactly the
    # moment to be curtailing). No weekday filter on purpose — if a Saturday
    # sets the month max, you still wanted to duck it.
    mtd_peak = df["load_mw"].fillna(-np.inf).groupby([df["_y"], df["_m"]]).cummax()
    df["is_4cp_candidate"] = (
        in_season
        & (hod >= FOUR_CP_WINDOW[0]) & (hod < FOUR_CP_WINDOW[1])
        & (df["load_mw"] >= FOUR_CP_NEAR * mtd_peak)
    ).fillna(False).astype(bool)
    return df.drop(columns=["_y", "_m"])


# ---------------------------------------------------------------- outputs

def summarize(df: pd.DataFrame, months: list[int]) -> dict:
    local = df["interval_start"].dt.tz_convert(LOCAL_TZ)
    years: dict[str, dict] = {}
    last_load = df.loc[df["load_mw"].notna(), "interval_start"].max()
    for y, g in df.groupby(local.dt.year):
        worst = g.loc[g["rt_price_max"].idxmax()] if g["rt_price_max"].notna().any() else None
        cps = []
        for _, r in g[g["is_4cp_actual"]].iterrows():
            ts = r["interval_start"].tz_convert(LOCAL_TZ)
            month_end = (ts + pd.offsets.MonthEnd(0)).replace(hour=23)
            cps.append({"month": int(ts.month),
                        "ts_local": ts.isoformat(),
                        "load_mw": round(float(r["load_mw"])),
                        "complete": bool(last_load is not pd.NaT and
                                         last_load.tz_convert(LOCAL_TZ) >= month_end)})
        years[str(int(y))] = {
            "hours": int(len(g)),
            "stress_hours": int(g["is_stress"].sum()),
            "stress_pct": round(float(g["is_stress"].mean() * 100), 2),
            "by_rule": {
                "price_spike": int(g["stress_price_spike"].sum()),
                "price_tail": int(g["stress_price_tail"].sum()),
                "load_tail": int(g["stress_load_tail"].sum()),
            },
            "worst_hour": None if worst is None else {
                "ts_local": worst["interval_start"].tz_convert(LOCAL_TZ).isoformat(),
                "rt_price_max": round(float(worst["rt_price_max"]), 1),
            },
            "four_cp_actual": cps,
            "four_cp_candidate_hours": int(g["is_4cp_candidate"].sum()),
        }
    return years


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2: label history stress/calm + 4CP")
    ap.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"))
    ap.add_argument("--synthetic", action="store_true",
                    help="label the dev cache (outputs kept separate)")
    ap.add_argument("--site", default=None, help="alternate site.yaml")
    a = ap.parse_args()

    root = RAW_SYNTH if a.synthetic else RAW
    tag = "_synthetic" if a.synthetic else ""
    site = load_site(a.site)
    cfg = site.get("labels", {})
    months = list(cfg.get("four_cp_months", [6, 7, 8, 9]))

    found = sorted(int(p.stem) for p in (root / "rtm_spp").glob("[0-9]*.parquet"))
    if not found:
        raise SystemExit(f"[labeler] nothing cached under {root}")
    years = [y for y in found if a.years is None or a.years[0] <= y <= a.years[1]]

    print(f"[labeler] source={'SYNTHETIC' if a.synthetic else 'REAL'} "
          f"hub={site['site']['price_location']} years={years[0]}-{years[-1]}")
    df = build_hourly(root, years, site)
    # Keep only the local-calendar years asked for: UTC-aligned files spill
    # a few edge hours into the prior local year, which would otherwise get
    # its own stub row (and a meaningless 6-hour percentile).
    ly = df["interval_start"].dt.tz_convert(LOCAL_TZ).dt.year
    df = df[ly.between(years[0], years[-1])].reset_index(drop=True)
    df = apply_stress(df, cfg)
    df = apply_four_cp(df, months)

    out_dir = OUT / f"labels{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    local_year = df["interval_start"].dt.tz_convert(LOCAL_TZ).dt.year
    for y, g in df.groupby(local_year):
        g.to_parquet(out_dir / f"{int(y)}.parquet", index=False)

    years_summary = summarize(df, months)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "synthetic" if a.synthetic else "real",
        "site": site["site"]["name"],
        "hub": site["site"]["price_location"],
        "thresholds": {
            "rt_price_spike_usd": cfg.get("rt_price_spike_usd", 200),
            "rt_price_top_pct": cfg.get("rt_price_top_pct", 0.005),
            "net_load_top_pct": cfg.get("net_load_top_pct", 0.01),
            "load_rule_note": "total system load (renewables history unavailable)",
            "four_cp_months": months,
            "four_cp_near": FOUR_CP_NEAR,
            "four_cp_window_local": list(FOUR_CP_WINDOW),
        },
        "years": years_summary,
    }
    spath = OUT / f"label_summary{tag}.json"
    spath.write_text(json.dumps(summary, indent=2))

    print(f"{'year':6s} {'hours':>7s} {'stress':>7s} {'spike':>6s} {'p-tail':>6s} "
          f"{'l-tail':>6s}  {'worst hour (local)':>22s} {'max $':>8s}  4CP")
    print("-" * 88)
    partial = False
    for y, s in years_summary.items():
        w = s["worst_hour"] or {}
        cp = ",".join(str(c["month"]) + ("" if c["complete"] else "*") for c in s["four_cp_actual"])
        partial = partial or any(not c["complete"] for c in s["four_cp_actual"])
        print(f"{y:6s} {s['hours']:>7,} {s['stress_hours']:>7,} "
              f"{s['by_rule']['price_spike']:>6,} {s['by_rule']['price_tail']:>6,} "
              f"{s['by_rule']['load_tail']:>6,}  "
              f"{w.get('ts_local', '-')[:16]:>22s} {w.get('rt_price_max', 0):>8,.0f}  {cp}")
    print("-" * 88)
    print(f"[labeler] wrote {out_dir}/<year>.parquet + {spath.name}"
          f"{'   (* = month still in progress)' if partial else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
