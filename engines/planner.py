"""
Phase 5 — the planner: forecast in, flight plan out.

Reads the 7-day forecast (data/out/forecast.json) and the site model, and
solves ONE linear program for the whole week: when each training job runs,
when the battery charges/discharges, and when DVFS capping fires — so that
every job still lands by its deadline while the campus draws least during
the hours the forecast says are expensive, risky, or 4CP-watch.

The cost model (all knobs overridable via an optional `planner:` section
in site.yaml):
    energy      grid_mw x [P50 + P(stress) x (P90 - P50)]  per hour
    4CP         a shadow price per MWh drawn during 4CP-watch afternoons
    friction    checkpoint overhead charged on every job pause
    battery     cycling cost per MWh discharged (from the rung ladder)

The plan is solved to optimality by CBC (via pulp) in a few seconds — the
forecast was learned, the plan is SOLVED. Alongside the schedule, every
hour gets a dispatch "playbook": if stress hits NOW, the rung ladder with
the relief actually available in that hour's planned state.

    python -m engines.planner                  # real forecast -> plan.json
    python -m engines.planner --synthetic
    python -m engines.planner --forecast F.json --out P.json   # what-if runs
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import OUT, load_site
from .site import load_model

DEFAULTS = {
    "four_cp_shadow_usd_mwh": 300.0,     # expected value of ducking a watch-hour MW
    "checkpoint_friction_usd_mwh": 300.0,  # value of compute lost during a checkpoint
    "battery_initial_soc_frac": 0.5,
}


# ---------------------------------------------------------------- inputs

def read_forecast(path) -> dict:
    p = json.loads(path.read_text())
    hours = p["hours"]
    if not hours:
        raise SystemExit(f"[planner] {path.name} has no hours")

    def fill(key, fallback):
        vals, last = [], None
        for h in hours:
            v = h.get(key)
            if v is not None:
                last = v
            vals.append(last)
        med = sorted(v for v in vals if v is not None)
        med = med[len(med) // 2] if med else fallback
        return [v if v is not None else med for v in vals]

    p50 = fill("price_p50", 30.0)
    p90 = [max(a, b) for a, b in zip(fill("price_p90", 60.0), p50)]
    ps = [h.get("p_stress") or 0.0 for h in hours]
    return {
        "H": len(hours),
        "ts_utc": [h["ts_utc"] for h in hours],
        "ts_local": [h["ts_local"] for h in hours],
        "p50": p50, "p90": p90, "p_stress": ps,
        "watch": [bool(h.get("four_cp_watch")) for h in hours],
        "generated_utc": p.get("generated_utc"),
        "source": p.get("source"),
    }


# ---------------------------------------------------------------- the LP

def solve(model, fc, cfg) -> dict:
    import pulp

    H = fc["H"]
    B = model.battery
    eprice = [fc["p50"][t] + fc["p_stress"][t] * max(0.0, fc["p90"][t] - fc["p50"][t])
              for t in range(H)]
    cp = [cfg["four_cp_shadow_usd_mwh"] if fc["watch"][t] else 0.0 for t in range(H)]
    rung_cost = {name: usd for name, _, usd in model.rungs}
    dvfs_mw = next((mw for n, mw, _ in model.rungs if n == "dvfs_cap"), 0.0)
    dvfs_usd = rung_cost.get("dvfs_cap", 40.0)
    batt_usd = rung_cost.get("battery", 15.0)
    dvfs_ratio = dvfs_mw / model.flex_mw if model.flex_mw else 0.0
    soc0 = cfg["battery_initial_soc_frac"] * B.energy_mwh

    prob = pulp.LpProblem("flexcast_week", pulp.LpMinimize)
    V = pulp.LpVariable

    def hrz(j):  # hours in which job j may run
        return range(min(j.deadline_h, H))

    run = {(j.id, t): V(f"run_{i}_{t}", 0, 1)
           for i, j in enumerate(model.jobs) for t in hrz(j)}
    stop = {(j.id, t): V(f"stop_{i}_{t}", 0)
            for i, j in enumerate(model.jobs) for t in hrz(j)}
    ch = [V(f"ch_{t}", 0, B.power_mw) for t in range(H)]
    dis = [V(f"dis_{t}", 0, B.power_mw) for t in range(H)]
    soc = [V(f"soc_{t}", B.min_soc_mwh, B.energy_mwh) for t in range(H)]
    dvfs = [V(f"dvfs_{t}", 0, dvfs_mw) for t in range(H)]
    grid = [V(f"grid_{t}", 0) for t in range(H)]

    def train(t):
        return pulp.lpSum(j.mw * run[j.id, t] for j in model.jobs if (j.id, t) in run)

    for t in range(H):
        prob += grid[t] == model.floor_mw + train(t) + ch[t] - dis[t] - dvfs[t]
        prob += train(t) <= model.flex_mw
        prob += dvfs[t] <= dvfs_ratio * train(t)
        prev = soc0 if t == 0 else soc[t - 1]
        prob += soc[t] == prev + ch[t] * B.rte - dis[t]
    prob += soc[H - 1] >= soc0  # leave the tank the way you found it

    for j in model.jobs:
        prob += pulp.lpSum(run[j.id, t] for t in hrz(j)) == j.hours
        for t in hrz(j):
            prev = 1.0 if t == 0 else run[j.id, t - 1]  # jobs are running at t=-1
            prob += stop[j.id, t] >= prev - run[j.id, t]

    chk = {j.id: j.mw * (j.checkpoint_min / 60.0) * cfg["checkpoint_friction_usd_mwh"]
           for j in model.jobs}
    prob += (
        pulp.lpSum(grid[t] * (eprice[t] + cp[t]) for t in range(H))
        + pulp.lpSum(stop[j.id, t] * chk[j.id] for j in model.jobs for t in hrz(j))
        + pulp.lpSum(dvfs[t] * dvfs_usd for t in range(H))
        + pulp.lpSum(dis[t] * batt_usd for t in range(H))
    )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise SystemExit(f"[planner] LP status {pulp.LpStatus[status]} — check "
                         "engines.site feasibility report")

    val = lambda v: max(0.0, v.value() or 0.0)
    return {
        "eprice": eprice, "cp": cp,
        "run": {(jid, t): val(v) for (jid, t), v in run.items()},
        "stops": sum(val(v) for v in stop.values()),
        "ch": [val(v) for v in ch], "dis": [val(v) for v in dis],
        "soc": [val(v) for v in soc], "dvfs": [val(v) for v in dvfs],
        "grid": [val(v) for v in grid],
        "friction_usd": sum(val(stop[j.id, t]) * chk[j.id]
                            for j in model.jobs for t in hrz(j)),
        "dvfs_usd": sum(val(v) * dvfs_usd for v in dvfs),
        "batt_usd": sum(val(v) * batt_usd for v in dis),
        "dvfs_ratio": dvfs_ratio, "batt_cost_usd_mwh": batt_usd,
    }


# ---------------------------------------------------------------- baseline

def flat_baseline(model, H: int) -> list[dict]:
    """The dumb datacenter: run every job flat-out ASAP (earliest deadline
    first), no battery, no price awareness. Returns per-hour job MW."""
    remaining = {j.id: j.hours for j in model.jobs}
    out = []
    for t in range(H):
        used, hour = 0.0, {}
        for j in model.jobs:  # already deadline-sorted
            if remaining[j.id] <= 0 or t >= j.deadline_h:
                continue
            lvl = min(1.0, remaining[j.id])
            if used + j.mw * lvl > model.flex_mw + 1e-9:
                lvl = max(0.0, (model.flex_mw - used) / j.mw)
            if lvl > 1e-9:
                hour[j.id] = j.mw * lvl
                used += j.mw * lvl
                remaining[j.id] -= lvl
        out.append(hour)
    if any(v > 1e-6 for v in remaining.values()):
        raise SystemExit(f"[planner] flat baseline couldn't finish jobs: {remaining}")
    return out


# ---------------------------------------------------------------- outputs

def playbook_hour(model, sol, t) -> list[dict]:
    """If stress hits in hour t: relief available NOW, cheapest first."""
    B = model.battery
    batch = sum(j.mw * sol["run"].get((j.id, t), 0.0)
                for j in model.jobs if j.id.startswith("batch"))
    non_batch = sum(j.mw * sol["run"].get((j.id, t), 0.0)
                    for j in model.jobs if not j.id.startswith("batch"))
    train_mw = batch + non_batch
    batt = min(B.power_mw - sol["dis"][t] + sol["ch"][t],
               max(0.0, sol["soc"][t] - B.min_soc_mwh))
    avail = {"defer_batch": batch,
             "battery": max(0.0, batt),
             "dvfs_cap": max(0.0, sol["dvfs_ratio"] * train_mw - sol["dvfs"][t]),
             "checkpoint_pause": non_batch}
    ladder, cum = [], 0.0
    for name, _, usd in model.rungs:
        mw = round(avail.get(name, 0.0), 1)
        if mw <= 0:
            continue
        cum += mw
        ladder.append({"rung": name, "mw": mw, "usd_mwh": usd,
                       "cumulative_mw": round(cum, 1)})
    return ladder


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5: solve the weekly flight plan")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--forecast", default=None, help="alternate forecast json (what-ifs)")
    ap.add_argument("--out", default=None, help="alternate output path")
    ap.add_argument("--site", default=None)
    a = ap.parse_args()

    tag = "_synthetic" if a.synthetic else ""
    fpath = OUT / f"forecast{tag}.json" if a.forecast is None else Path(a.forecast)
    if not fpath.exists():
        raise SystemExit(f"[planner] {fpath} missing — run engines.forecast --predict first")

    model, warn, fatal = load_model(a.site)
    for w in warn:
        print(f"[planner] site WARN: {w}")
    if fatal:
        raise SystemExit("[planner] site config not schedulable:\n  " + "\n  ".join(fatal))
    cfg = {**DEFAULTS, **model.planner_cfg}

    fc = read_forecast(fpath)
    H = fc["H"]
    print(f"[planner] {model.name}: {H}h horizon from {fc['generated_utc']} "
          f"({fc['source']} forecast)")
    sol = solve(model, fc, cfg)

    base_jobs = flat_baseline(model, H)
    base_grid = [model.floor_mw + sum(b.values()) for b in base_jobs]
    e_plan = sum(sol["grid"][t] * sol["eprice"][t] for t in range(H))
    e_base = sum(base_grid[t] * sol["eprice"][t] for t in range(H))
    c_plan = sum(sol["grid"][t] * sol["cp"][t] for t in range(H))
    c_base = sum(base_grid[t] * sol["cp"][t] for t in range(H))
    extras = sol["friction_usd"] + sol["dvfs_usd"] + sol["batt_usd"]
    plan_cost = e_plan + c_plan + extras
    base_cost = e_base + c_base
    save = base_cost - plan_cost

    deadlines = []
    for j in model.jobs:
        done, done_by = 0.0, None
        for t in range(min(j.deadline_h, H)):
            done += sol["run"].get((j.id, t), 0.0)
            if done >= j.hours - 1e-6 and done_by is None:
                done_by = t
        deadlines.append({"id": j.id, "due_h": j.deadline_h, "done_by_h": done_by,
                          "met": done_by is not None and done_by < j.deadline_h})

    hours_out = []
    for t in range(H):
        jobs_mw = {j.id: round(j.mw * sol["run"].get((j.id, t), 0.0), 1)
                   for j in model.jobs if sol["run"].get((j.id, t), 0.0) > 1e-4}
        hours_out.append({
            "ts_utc": fc["ts_utc"][t], "ts_local": fc["ts_local"][t],
            "eprice_usd_mwh": round(sol["eprice"][t], 2),
            "p_stress": round(fc["p_stress"][t], 4),
            "four_cp_watch": fc["watch"][t],
            "jobs_mw": jobs_mw,
            "training_mw": round(sum(jobs_mw.values()), 1),
            "dvfs_shed_mw": round(sol["dvfs"][t], 1),
            "battery_charge_mw": round(sol["ch"][t], 1),
            "battery_discharge_mw": round(sol["dis"][t], 1),
            "battery_soc_mwh": round(sol["soc"][t], 1),
            "grid_mw": round(sol["grid"][t], 1),
            "grid_flat_baseline_mw": round(base_grid[t], 1),
            "playbook": playbook_hour(model, sol, t),
        })

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": fc["source"], "site": model.name,
        "forecast_generated_utc": fc["generated_utc"],
        "cost_model": {k: cfg[k] for k in DEFAULTS},
        "totals": {
            "plan_cost_usd": round(plan_cost), "flat_baseline_cost_usd": round(base_cost),
            "savings_usd": round(save),
            "savings_pct": round(100 * save / base_cost, 1) if base_cost else None,
            "breakdown_usd": {
                "energy_timing": round(e_base - e_plan),
                "four_cp_exposure": round(c_base - c_plan),
                "checkpoint_friction": -round(sol["friction_usd"]),
                "dvfs_throughput": -round(sol["dvfs_usd"]),
                "battery_cycling": -round(sol["batt_usd"]),
            },
            "battery_mwh_discharged": round(sum(sol["dis"]), 1),
            "job_pauses_equivalent": round(sol["stops"], 2),
        },
        "deadlines": deadlines,
        "hours": hours_out,
    }
    opath = OUT / f"plan{tag}.json" if a.out is None else Path(a.out)
    opath.write_text(json.dumps(payload, indent=2))

    met = sum(d["met"] for d in deadlines)
    risky = sorted(range(H), key=lambda t: -(fc["p_stress"][t]))[:3]
    print(f"[planner] plan ${plan_cost:,.0f} vs flat ${base_cost:,.0f} "
          f"-> saves ${save:,.0f} ({100 * save / base_cost:.1f}%)")
    b = payload["totals"]["breakdown_usd"]
    print(f"[planner]   energy timing +${b['energy_timing']:,} | 4CP +${b['four_cp_exposure']:,}"
          f" | friction -${-b['checkpoint_friction']:,} | dvfs -${-b['dvfs_throughput']:,}"
          f" | battery -${-b['battery_cycling']:,}")
    print(f"[planner]   deadlines {met}/{len(deadlines)} met | "
          f"battery {payload['totals']['battery_mwh_discharged']} MWh discharged | "
          f"{payload['totals']['job_pauses_equivalent']} job-pauses")
    for t in risky:
        if fc["p_stress"][t] > 0.05:
            print(f"[planner]   risky {fc['ts_local'][t][:16]} P(stress)={fc['p_stress'][t]:.0%}: "
                  f"grid {sol['grid'][t]:,.0f} MW (flat would draw {base_grid[t]:,.0f})")
    print(f"[planner] wrote {opath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
