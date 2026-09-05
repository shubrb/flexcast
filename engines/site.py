"""
Phase 4 — the site model.

Turns site.yaml into validated, typed objects every later engine shares:
the inference floor (never curtailed), the flexible training block, the
battery, the rung ladder (the "flex supply curve" — relief sorted
cheapest-first), and the training-job queue with deadlines.

Also the config linter: catches impossible sites BEFORE the planner
produces nonsense (a job that can't make its deadline even running alone,
rungs that shed more MW than exists, a floor bigger than the campus).

    python -m engines.site        # feasibility report + data/out/flex_curve.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from .config import OUT, load_site


@dataclass(frozen=True)
class Job:
    id: str
    mw: float
    hours: float           # GPU-hours still owed
    deadline_h: int        # hours from plan start
    checkpoint_min: float  # minutes to checkpoint safely


@dataclass(frozen=True)
class Battery:
    energy_mwh: float
    power_mw: float
    rte: float             # round-trip efficiency, applied on charge
    min_soc_mwh: float


@dataclass(frozen=True)
class SiteModel:
    name: str
    floor_mw: float        # inference — never curtailed
    flex_mw: float         # training block ceiling
    battery: Battery
    rungs: tuple           # (name, mw, usd_mwh), sorted cheapest first
    jobs: tuple            # Job, sorted by deadline
    planner_cfg: dict      # optional planner: section from site.yaml


def load_model(path=None) -> tuple[SiteModel, list[str], list[str]]:
    """Returns (model, warnings, fatals)."""
    return model_from_dict(load_site(path))


def model_from_dict(raw: dict) -> tuple[SiteModel, list[str], list[str]]:
    """Build + lint a SiteModel from a site.yaml-shaped dict (also used by
    the live-solve server, where the dict comes from the console's form)."""
    warn: list[str] = []
    fatal: list[str] = []

    p = raw.get("power", {})
    floor = float(p.get("inference_floor_mw", 0))
    flex = float(p.get("flexible_training_mw", 0))
    total = float(p.get("total_mw", floor + flex))
    if floor + flex > total + 1e-6:
        fatal.append(f"floor {floor} + flexible {flex} exceeds total {total} MW")

    b = raw.get("battery", {})
    batt = Battery(
        energy_mwh=float(b.get("energy_mwh", 0)),
        power_mw=float(b.get("power_mw", 0)),
        rte=float(b.get("round_trip_efficiency", 1.0)),
        min_soc_mwh=float(b.get("energy_mwh", 0)) * float(b.get("min_soc", 0)),
    )
    if not 0 < batt.rte <= 1:
        fatal.append(f"battery round_trip_efficiency {batt.rte} not in (0, 1]")

    rungs = tuple(sorted(
        ((r["name"], float(r["mw"]), float(r["cost_per_mwh"]))
         for r in raw.get("flex_rungs", [])),
        key=lambda r: r[2]))
    non_batt = sum(mw for name, mw, _ in rungs if name != "battery")
    if non_batt > flex + 1e-6:
        warn.append(f"non-battery rungs total {non_batt:.0f} MW > flexible block "
                    f"{flex:.0f} MW — ladder can't all fire at once")

    jobs = []
    for j in raw.get("jobs", []):
        job = Job(id=str(j["id"]), mw=float(j["mw"]), hours=float(j["hours"]),
                  deadline_h=int(j["deadline_days"] * 24),
                  checkpoint_min=float(j.get("checkpoint_min", 10)))
        if job.mw > flex:
            fatal.append(f"job {job.id}: draws {job.mw} MW > flexible block {flex} MW")
        if job.hours > job.deadline_h:
            fatal.append(f"job {job.id}: needs {job.hours}h of compute but deadline "
                         f"is only {job.deadline_h}h away — impossible even running nonstop")
        jobs.append(job)
    jobs.sort(key=lambda j: j.deadline_h)

    # Necessary schedulability condition (EDF): by every deadline D, the energy
    # owed by all jobs due <= D must fit in the flexible block's capacity to D.
    for cut in sorted({j.deadline_h for j in jobs}):
        owed = sum(j.mw * j.hours for j in jobs if j.deadline_h <= cut)
        cap = flex * cut
        if owed > cap + 1e-6:
            fatal.append(f"jobs due within {cut}h owe {owed:,.0f} MWh but the "
                         f"flexible block can only deliver {cap:,.0f} MWh by then")

    model = SiteModel(
        name=raw.get("site", {}).get("name", "site"),
        floor_mw=floor, flex_mw=flex, battery=batt,
        rungs=rungs, jobs=tuple(jobs),
        planner_cfg=dict(raw.get("planner", {})),
    )
    return model, warn, fatal


def flex_curve(model: SiteModel) -> list[dict]:
    cum = 0.0
    out = []
    for name, mw, usd in model.rungs:
        cum += mw
        out.append({"name": name, "mw": mw, "usd_mwh": usd, "cumulative_mw": cum})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4: site model + feasibility lint")
    ap.add_argument("--site", default=None)
    a = ap.parse_args()
    model, warn, fatal = load_model(a.site)

    print(f"[site] {model.name}: {model.floor_mw:.0f} MW floor + "
          f"{model.flex_mw:.0f} MW flexible; battery {model.battery.energy_mwh:.0f} MWh "
          f"/ {model.battery.power_mw:.0f} MW (rte {model.battery.rte:.0%})")
    print(f"[site] flex ladder (cheapest first):")
    for step in flex_curve(model):
        print(f"    {step['name']:18s} {step['mw']:>5.0f} MW @ ${step['usd_mwh']:>6.0f}/MWh"
              f"   (cumulative {step['cumulative_mw']:.0f} MW)")
    owed = sum(j.mw * j.hours for j in model.jobs)
    print(f"[site] job queue: {len(model.jobs)} jobs owing {owed:,.0f} MWh "
          f"(= {owed / model.flex_mw:.0f}h flat-out)")
    for j in model.jobs:
        print(f"    {j.id:12s} {j.mw:>4.0f} MW x {j.hours:>4.0f}h  due in {j.deadline_h:>4d}h"
              f"  (checkpoint {j.checkpoint_min:.0f} min)")
    for w in warn:
        print(f"[site] WARN: {w}")
    for f in fatal:
        print(f"[site] FATAL: {f}")

    payload = {
        "site": model.name, "floor_mw": model.floor_mw, "flex_mw": model.flex_mw,
        "battery": asdict(model.battery), "rungs": flex_curve(model),
        "jobs": [asdict(j) for j in model.jobs],
        "feasible": not fatal, "warnings": warn, "fatals": fatal,
    }
    (OUT / "flex_curve.json").write_text(json.dumps(payload, indent=2))
    print(f"[site] {'OK — site is schedulable' if not fatal else 'NOT SCHEDULABLE'} "
          f"-> {OUT / 'flex_curve.json'}")
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
