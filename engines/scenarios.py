"""
Phase 7 helper — the what-if solver behind the console's simulator buttons.

Takes the current 7-day forecast, rigs the weather three different ways
(a heat dome, a winter-storm-style price shock, a windy cheap week), and
solves the SAME weekly LP the real planner uses for each one. The console
swaps between the precomputed solutions instantly, so the buttons feel
live while every number is a genuine optimization, not browser math.

    python -m engines.scenarios [--synthetic]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from .config import OUT, load_site
from .planner import DEFAULTS, flat_baseline, read_forecast, solve
from .site import load_model


def _hours_local(fc):
    return [int(ts[11:13]) for ts in fc["ts_local"]]


def rig_heat(fc):
    """Tue-Thu afternoons turn dangerous: scarcity odds spike, prices follow."""
    out = {k: list(v) if isinstance(v, list) else v for k, v in fc.items()}
    hrs = _hours_local(fc)
    for t in range(fc["H"]):
        day = t // 24
        if 2 <= day <= 4 and 13 <= hrs[t] < 21:
            out["p50"][t] = max(fc["p50"][t] * 6, 350.0)
            out["p90"][t] = max(fc["p90"][t] * 6, 1200.0)
            out["p_stress"][t] = max(fc["p_stress"][t], 0.85)
            out["watch"][t] = True
        elif 2 <= day <= 4:
            out["p50"][t] = fc["p50"][t] * 1.6
            out["p90"][t] = fc["p90"][t] * 1.8
    return out


def rig_storm(fc):
    """A 40-hour Uri-style shock mid-week: prices pinned near the cap."""
    out = {k: list(v) if isinstance(v, list) else v for k, v in fc.items()}
    for t in range(fc["H"]):
        if 60 <= t < 100:
            out["p50"][t] = max(fc["p50"][t] * 12, 1800.0)
            out["p90"][t] = max(fc["p90"][t] * 10, 4500.0)
            out["p_stress"][t] = 0.95
        elif 52 <= t < 60 or 100 <= t < 110:
            out["p50"][t] = fc["p50"][t] * 3
            out["p90"][t] = fc["p90"][t] * 3
            out["p_stress"][t] = max(fc["p_stress"][t], 0.4)
    return out


def rig_windy(fc):
    """West Texas wind floods the wires: power gets cheap, risk evaporates."""
    out = {k: list(v) if isinstance(v, list) else v for k, v in fc.items()}
    for t in range(fc["H"]):
        out["p50"][t] = max(fc["p50"][t] * 0.45, 4.0)
        out["p90"][t] = max(fc["p90"][t] * 0.5, 8.0)
        out["p_stress"][t] = fc["p_stress"][t] * 0.15
        out["watch"][t] = False
    return out


SCENARIOS = [
    ("base", "This week's real forecast", None,
     "The model's actual read of the next 7 days."),
    ("heat", "Heat dome", rig_heat,
     "Three mid-week afternoons hit scarcity pricing. Watch the plan pre-charge the battery and empty those afternoons."),
    ("storm", "Storm shock", rig_storm,
     "A 40-hour Uri-style price spike lands mid-week. Deadlines still have to hold."),
    ("windy", "Wind flood", rig_windy,
     "Wind crushes prices all week. The plan runs hard and barely bothers flexing."),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Precompute what-if weeks for the console")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--site", default=None)
    a = ap.parse_args()
    tag = "_synthetic" if a.synthetic else ""

    fpath = OUT / f"forecast{tag}.json"
    if not fpath.exists():
        raise SystemExit(f"[scenarios] {fpath} missing — run engines.forecast --predict first")
    base_fc = read_forecast(fpath)
    base_json = json.loads(fpath.read_text())
    p10_ratio = []
    for h in base_json["hours"]:
        p10_ratio.append((h.get("price_p10") or 0) / h["price_p50"]
                         if h.get("price_p50") else 0.6)

    model, warn, fatal = load_model(a.site)
    if fatal:
        raise SystemExit("[scenarios] site not schedulable: " + "; ".join(fatal))
    cfg = {**DEFAULTS, **model.planner_cfg}
    flat_jobs = flat_baseline(model, base_fc["H"])
    flat_grid = [model.floor_mw + sum(h.values()) for h in flat_jobs]

    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source": base_json.get("source"), "scenarios": []}
    for key, label, rig, blurb in SCENARIOS:
        fc = base_fc if rig is None else rig(base_fc)
        sol = solve(model, fc, cfg)
        e_plan = sum(sol["grid"][t] * sol["eprice"][t] for t in range(fc["H"]))
        cp_plan = sum(sol["grid"][t] * sol["cp"][t] for t in range(fc["H"]))
        e_flat = sum(flat_grid[t] * sol["eprice"][t] for t in range(fc["H"]))
        cp_flat = sum(flat_grid[t] * sol["cp"][t] for t in range(fc["H"]))
        extras = sol["friction_usd"] + sol["dvfs_usd"] + sol["batt_usd"]
        plan_cost, flat_cost = e_plan + cp_plan + extras, e_flat + cp_flat
        fhours, phours = [], []
        for t in range(fc["H"]):
            fhours.append({
                "ts_local": fc["ts_local"][t],
                "price_p10": round(fc["p50"][t] * p10_ratio[t], 2),
                "price_p50": round(fc["p50"][t], 2),
                "price_p90": round(fc["p90"][t], 2),
                "p_stress": round(fc["p_stress"][t], 4),
                "four_cp_watch": bool(fc["watch"][t]),
                "drivers": [],
            })
            jobs = {j.id: round(j.mw * sol["run"].get((j.id, t), 0.0), 1)
                    for j in model.jobs if sol["run"].get((j.id, t), 0.0) > 1e-4}
            phours.append({
                "ts_local": fc["ts_local"][t], "jobs_mw": jobs,
                "training_mw": round(sum(jobs.values()), 1),
                "battery_charge_mw": round(sol["ch"][t], 1),
                "battery_discharge_mw": round(sol["dis"][t], 1),
                "battery_soc_mwh": round(sol["soc"][t], 1),
                "dvfs_shed_mw": round(sol["dvfs"][t], 1),
                "grid_mw": round(sol["grid"][t], 1),
                "grid_flat_baseline_mw": round(flat_grid[t], 1),
            })
        out["scenarios"].append({
            "key": key, "label": label, "blurb": blurb,
            "fhours": fhours, "phours": phours,
            "totals": {
                "plan_cost_usd": round(plan_cost),
                "flat_baseline_cost_usd": round(flat_cost),
                "savings_usd": round(flat_cost - plan_cost),
                "savings_pct": round(100 * (flat_cost - plan_cost) / flat_cost, 1)
                if flat_cost else None,
            },
        })
        print(f"[scenarios] {label:24s} plan ${plan_cost:,.0f} vs flat ${flat_cost:,.0f} "
              f"-> saves ${flat_cost - plan_cost:,.0f}")
    path = OUT / f"scenarios{tag}.json"
    path.write_text(json.dumps(out))
    print(f"[scenarios] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
