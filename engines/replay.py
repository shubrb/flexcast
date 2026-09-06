"""
Phase 6 — the replay: live through held-out history and count real dollars.

Every Monday 00:00 (local) from the start of the held-out years, the replay:
  1. forecasts the coming week USING ONLY what was knowable that midnight —
     Monday's hours get the day-ahead model (its lags reach back 24h, legal
     only for lead <= 24h); Tue-Sun get the week-ahead model (lags >= 168h).
     4CP-watch flags are rebuilt causally from PREDICTED load + the
     month-to-date peak known at plan time — never from hindsight labels.
  2. solves the same weekly LP as Phase 5 (battery state carried over),
  3. executes three strategies through what ACTUALLY happened:
       flat            run every job ASAP, no awareness (the baseline)
       plan            the committed weekly schedule
       plan+dispatch   plan + reflexes: an hour whose ACTUAL price crosses
                       the spike trigger pauses jobs that still have
                       deadline slack and dumps the battery; paused hours
                       are made up later; friction charged; any
                       dispatch-caused deadline miss is recorded, not hidden.

Scoring is on reality, not on the forecast: energy at actual realized RT
prices; 4CP exposure at each strategy's draw during the ACTUAL four
coincident peaks (labels' is_4cp_actual) x a real transmission rate.
Jobs renew each Monday, so ~85 weeks = ~500 deadline commitments audited.

    python -m engines.replay               # real held-out years
    python -m engines.replay --synthetic
    python -m engines.replay --max-weeks 8   # quick smoke run
"""
from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import OUT, RAW, RAW_SYNTH, load_site
from .forecast import FEATURES, _cal, add_features, apply_climatology, build_frame
from .labeler import FOUR_CP_NEAR, FOUR_CP_WINDOW, LOCAL_TZ
from .planner import DEFAULTS as PLANNER_DEFAULTS
from .planner import flat_baseline, solve
from .site import model_from_dict

REPLAY_DEFAULTS = {
    "four_cp_rate_usd_per_mw_year": 50_000.0,  # ERCOT-ish transmission charge
    "dispatch_trigger_usd_mwh": 250.0,         # actual RT price that fires reflexes
    "four_cp_watch_margin": 0.05,              # plan-time slack for load-model error
    "four_cp_dispatch_near": 0.98,             # live-load fraction of mtd max that sheds
}


# ---------------------------------------------------------------- forecasts

def precompute_predictions(df: pd.DataFrame, bundle: dict) -> dict:
    """Per-horizon predictions for EVERY hour, vectorized once. Validity by
    lead time is enforced later when weeks are assembled."""
    out = {}
    for hz, feats in FEATURES.items():
        X = df[feats]
        p = _cal(bundle["iso"][hz], bundle["models"][("stress", hz)].predict_proba(X)[:, 1])
        q50 = np.sinh(bundle["models"][("price_q50", hz)].predict(X))
        q90 = np.sinh(bundle["models"][("price_q90", hz)].predict(X))
        out[hz] = {
            "p_stress": pd.Series(p, index=df.index),
            "p50": pd.Series(q50, index=df.index),
            "p90": pd.Series(np.maximum(q50, q90), index=df.index),
            "load": pd.Series(bundle["models"][("load", hz)].predict(X), index=df.index),
        }
    return out


def mtd_seeds(df: pd.DataFrame, t0, window_loc, months: list[int],
              seed_frac: float = 0.92) -> dict:
    """Known month-to-date load peaks as of t0, per (year, month) in the
    window — floored at seed_frac x LAST year's same-month peak (also
    knowable at t0). Without the floor, every warm afternoon early in a
    month looks like a 'new record' and fires false alarms."""
    known = df.loc[df.index < t0, "load_mw"]
    kloc = known.index.tz_convert(LOCAL_TZ)
    seeds = {}
    for y, m in {(y, m) for y, m in zip(window_loc.year, window_loc.month) if m in months}:
        mk = known[(kloc.year == y) & (kloc.month == m)]
        mtd = float(mk.max()) if mk.notna().any() else 0.0
        prior = known[(kloc.year == y - 1) & (kloc.month == m)]
        floor = seed_frac * float(prior.max()) if prior.notna().any() else 0.0
        seeds[(y, m)] = max(mtd, floor)
    return seeds


def week_forecast(window: pd.DatetimeIndex, preds: dict, df: pd.DataFrame,
                  months: list[int], margin: float = 0.0) -> dict:
    """Assemble the causal forecast for one week starting at window[0]."""
    lead = np.arange(1, len(window) + 1)
    hz = np.where(lead <= 24, "1d", "7d")
    pick = lambda key: np.array([preds[h][key].get(ts, np.nan)
                                 for h, ts in zip(hz, window)])
    p50 = np.nan_to_num(pick("p50"), nan=30.0)
    p90 = np.maximum(np.nan_to_num(pick("p90"), nan=60.0), p50)
    ps = np.clip(np.nan_to_num(pick("p_stress"), nan=0.0), 0, 1)
    load_pred = pick("load")

    # causal 4CP-watch: predicted load vs (known month-to-date peak + predicted-
    # so-far), with a slack margin because the week-ahead load model errs by
    # a few GW — the cost of a false watch is small, a missed peak is not
    loc = window.tz_convert(LOCAL_TZ)
    running = mtd_seeds(df, window[0], loc, months)
    near = FOUR_CP_NEAR - margin
    watch = []
    for i, ts in enumerate(loc):
        w = False
        if ts.month in months and np.isfinite(load_pred[i]):
            key = (ts.year, ts.month)
            running[key] = max(running.get(key, 0.0), float(load_pred[i]))
            w = (FOUR_CP_WINDOW[0] <= ts.hour < FOUR_CP_WINDOW[1]
                 and load_pred[i] >= near * running[key])
        watch.append(w)
    return {"H": len(window), "p50": list(p50), "p90": list(p90),
            "p_stress": list(ps), "watch": watch}


# ---------------------------------------------------------------- execution

def executed_matrices(model, sol, H):
    """Plan solution -> per-job level matrix + battery arrays (copies)."""
    lvl = {j.id: np.array([sol["run"].get((j.id, t), 0.0) for t in range(H)])
           for j in model.jobs}
    return lvl, np.array(sol["ch"]), np.array(sol["dis"])


def apply_dispatch(model, lvl, ch, dis, soc0, actual_rt, trigger, chk_usd, eprice,
                   window_loc, actual_load, months, seeds, cp_near, watch):
    """Reflex layer on a committed week. Two triggers, both real-time
    observable: the ACTUAL price crossing the spike threshold, and the ACTUAL
    statewide load closing on the month-to-date record during a 4CP-season
    afternoon (how real 4CP chasers operate — off the live load feed).
    Returns (lvl, ch, dis, reflex_hours, cp_reflex_count, extra_friction,
    makeup_missing, missing_jobs)."""
    B = model.battery
    H = len(actual_rt)
    lvl = {k: v.copy() for k, v in lvl.items()}
    ch, dis = ch.copy(), dis.copy()
    seeds = dict(seeds)
    remaining = {j.id: j.hours for j in model.jobs}
    friction = 0.0
    reflex_hours, cp_count = [], 0
    soc = soc0
    # Capacity-aware pause budget. At frontier utilization the fleet has
    # little spare room, so "wall-clock slack" alone over-promises: every
    # pause must be repayable out of REAL spare fleet capacity in future,
    # non-watch hours before that job's deadline, with margin for the make-up
    # pass being imperfect. True records / price spikes get a lenient margin,
    # merely near-record afternoons a strict one — when the budget is tight,
    # it is spent on the hours that actually cost money.
    train_mw = [sum(jj.mw * lvl[jj.id][t] for jj in model.jobs) for t in range(H)]
    debt_mwh = 0.0
    for h in range(H):
        price_hot = np.isfinite(actual_rt[h]) and actual_rt[h] >= trigger
        ts = window_loc[h]
        cp_hot = record_hot = False
        if ts.month in months and np.isfinite(actual_load[h]):
            key = (ts.year, ts.month)
            prior = seeds.get(key, 0.0)  # exclusive: the record BEFORE this hour
            in_win = FOUR_CP_WINDOW[0] <= ts.hour < FOUR_CP_WINDOW[1]
            cp_hot = prior > 0 and in_win and actual_load[h] >= cp_near * prior
            # a strict new monthly record — every ACTUAL coincident peak is
            # one, so the battery is reserved for exactly these (near-record
            # afternoons get job pauses only, keeping the tank full)
            record_hot = in_win and actual_load[h] >= prior > 0
            seeds[key] = max(prior, float(actual_load[h]))
        hot = price_hot or cp_hot
        if cp_hot and not price_hot:
            cp_count += 1
        if hot:
            ch[h] = 0.0
            if price_hot or record_hot:
                dis[h] = B.power_mw
            # Pause jobs only for real money: an actual price spike, or a
            # true monthly record (every actual coincident peak is one).
            # Merely near-record afternoons get the charge freeze + battery,
            # never a pause — at frontier utilization those pauses are
            # friction the fleet cannot repay.
            if price_hot or record_hot:
                for j in model.jobs:
                    run_now = lvl[j.id][h]
                    if run_now <= 1e-6 or h >= j.deadline_h:
                        continue
                    slack_ok = (j.deadline_h - (h + 1)) >= (remaining[j.id] + 6)
                    if not slack_ok:
                        continue
                    need = run_now * j.mw
                    # repayment room THIS job can actually use: hours before
                    # its deadline where it is not already running flat-out
                    # AND the fleet has spare MW (watch afternoons excluded)
                    cap = sum(j.mw * min(1.0 - lvl[j.id][t],
                                         max(0.0, model.flex_mw - train_mw[t]) / j.mw)
                              for t in range(h + 1, j.deadline_h) if not watch[t])
                    if cap < (debt_mwh + need) * 1.15:
                        continue  # fleet too hot to repay this pause
                    friction += run_now * j.mw * (j.checkpoint_min / 60.0) * chk_usd
                    train_mw[h] -= run_now * j.mw
                    lvl[j.id][h] = 0.0
                    debt_mwh += need
            reflex_hours.append(h)
        # battery can only discharge energy it actually has — reflexes drained
        # earlier hours, so later PLANNED discharges must be re-checked too
        dis[h] = min(dis[h], B.power_mw, max(0.0, soc + ch[h] * B.rte - B.min_soc_mwh))
        soc = min(B.energy_mwh, soc + ch[h] * B.rte - dis[h])
        for j in model.jobs:
            remaining[j.id] -= lvl[j.id][h]
    # make up paused job-hours in spare capacity — forecast-cheapest hours
    # first (the forecast was knowable; actual prices would be hindsight)
    missing, missing_jobs = 0.0, 0
    reflex_set = set(reflex_hours)
    watch_set = {h for h in range(H) if watch[h]}
    # cheapest forecast hours first; watch-flagged afternoons only as a last
    # resort (making up INTO a likely 4CP hour would undo the whole point)
    by_cheap = sorted(range(H), key=lambda h: (h in watch_set, eprice[h]))
    for j in model.jobs:
        short = remaining[j.id]
        if short <= 1e-6:
            continue
        for h in by_cheap:
            if short <= 1e-6:
                break
            if h in reflex_set or h >= j.deadline_h:
                continue
            train_h = sum(jj.mw * lvl[jj.id][h] for jj in model.jobs)
            room = min(1.0 - lvl[j.id][h], (model.flex_mw - train_h) / j.mw)
            if room > 1e-6:
                add = min(short, room)
                lvl[j.id][h] += add
                short -= add
        if short > 1e-3:
            missing_jobs += 1
        missing += max(0.0, short)
    return lvl, ch, dis, reflex_hours, cp_count, friction, missing, missing_jobs


def grid_series(model, lvl, ch, dis, dvfs, H):
    train = np.zeros(H)
    for j in model.jobs:
        train += j.mw * lvl[j.id]
    return model.floor_mw + train + ch - dis - dvfs


# ---------------------------------------------------------------- main loop

def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6: weekly rolling backtest on held-out years")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-weeks", type=int, default=None)
    ap.add_argument("--site", default=None)
    a = ap.parse_args()

    tag = "_synthetic" if a.synthetic else ""
    root = RAW_SYNTH if a.synthetic else RAW
    bpath = OUT / f"models{tag}" / "bundle.pkl"
    if not bpath.exists():
        raise SystemExit(f"[replay] {bpath} missing — run engines.forecast first")
    with open(bpath, "rb") as fh:
        bundle = pickle.load(fh)

    site = load_site(a.site)
    proof_load = None
    if site.get("proof_jobs"):
        # Score the held-out period on the FRONTIER-utilization queue (a
        # realistically hot campus); the interactive demo keeps the lighter
        # queue. Disclosed in replay.json and on the Proof page.
        site = {**site, "jobs": site["proof_jobs"]}
        wk_mwh = sum(j["mw"] * j["hours"] for j in site["jobs"])
        avg_mw = site["power"]["inference_floor_mw"] + wk_mwh / 168.0
        fleet_pct = 100 * wk_mwh / (site["power"]["flexible_training_mw"] * 168.0)
        proof_load = {
            "jobs_mwh_per_week": round(wk_mwh),
            "avg_campus_mw": round(avg_mw),
            "pct_of_connection": round(100 * avg_mw / site["power"]["total_mw"], 1),
            "flex_fleet_pct_busy": round(fleet_pct, 1),
            "note": "held-out scoring uses this frontier-utilization queue; the "
                    "interactive demo campus runs the lighter site.yaml queue",
        }
        print(f"[replay] proof load: {wk_mwh:,.0f} MWh/wk owed, campus avg "
              f"{avg_mw:,.0f} MW ({fleet_pct:.0f}% of the flex fleet busy)")
    model, warn, fatal = model_from_dict(site)
    if fatal:
        raise SystemExit("[replay] site not schedulable: " + "; ".join(fatal))
    cfg_p = {**PLANNER_DEFAULTS, **model.planner_cfg}
    cfg_r = {**REPLAY_DEFAULTS, **model.planner_cfg}
    months = list(site.get("labels", {}).get("four_cp_months", [6, 7, 8, 9]))
    chk_usd = cfg_p["checkpoint_friction_usd_mwh"]

    print(f"[replay] building features + predictions ({bundle['source']} cache)...")
    df = build_frame(tag, root)
    df = add_features(df, float(site.get("labels", {}).get("rt_price_spike_usd", 200)))
    df = apply_climatology(df, bundle["climo"])
    preds = precompute_predictions(df, bundle)

    test_years = set(bundle["test_years"])
    loc_all = df.index.tz_convert(LOCAL_TZ)
    priced = df["rt_price_mean"].notna()
    # Mondays whose 168h window sits in test years with nearly-complete prices
    mondays = []
    for ts in df.index[(loc_all.dayofweek == 0) & (loc_all.hour == 0)]:
        if ts.tz_convert(LOCAL_TZ).year not in test_years:
            continue
        window = pd.date_range(ts, periods=168, freq="h")
        if window[-1] not in df.index:
            continue
        if priced.reindex(window).sum() >= 160:
            mondays.append(ts)
    if a.max_weeks:
        mondays = mondays[: a.max_weeks]
    if not mondays:
        raise SystemExit("[replay] no complete replay weeks found")
    print(f"[replay] {len(mondays)} weeks: {mondays[0].tz_convert(LOCAL_TZ).date()} "
          f"-> {mondays[-1].tz_convert(LOCAL_TZ).date()} "
          f"(models trained on {bundle['train_years']} — never on these)")

    flat_jobs = flat_baseline(model, 168)
    flat_grid_week = np.array([model.floor_mw + sum(h.values()) for h in flat_jobs])

    strat_grid = {s: {} for s in ("flat", "plan", "dispatch")}  # ts -> MW
    energy = dict.fromkeys(strat_grid, 0.0)
    op_cost = dict.fromkeys(strat_grid, 0.0)
    deadline_slots = deadline_met = 0
    reflex_total, makeup_missing = 0, 0.0
    weekly_rows = []
    # separate battery worlds: the plan world follows the LP exactly; the
    # dispatch world's reflexes drain extra, so it carries its OWN state and
    # its planned discharges are feasibility-clamped against it
    soc_plan = soc_disp = cfg_p["battery_initial_soc_frac"] * model.battery.energy_mwh

    cp_reflex_total = 0
    for wi, t0 in enumerate(mondays):
        window = pd.date_range(t0, periods=168, freq="h")
        window_loc = window.tz_convert(LOCAL_TZ)
        fc = week_forecast(window, preds, df, months,
                           margin=cfg_r["four_cp_watch_margin"])
        cfg_week = {**cfg_p,
                    "battery_initial_soc_frac": soc_plan / model.battery.energy_mwh}
        sol = solve(model, fc, cfg_week)

        actual_rt = df["rt_price_mean"].reindex(window).to_numpy(float)
        actual_load = df["load_mw"].reindex(window).to_numpy(float)
        price_paid = np.where(np.isfinite(actual_rt), actual_rt, np.array(fc["p50"]))
        dvfs = np.array(sol["dvfs"])

        lvl_p, ch_p, dis_p = executed_matrices(model, sol, 168)
        grid_p = grid_series(model, lvl_p, ch_p, dis_p, dvfs, 168)

        seeds = mtd_seeds(df, t0, window_loc, months)
        lvl_d, ch_d, dis_d, reflex, cp_n, fric_x, miss, miss_jobs = apply_dispatch(
            model, lvl_p, ch_p, dis_p, soc_disp, actual_rt,
            cfg_r["dispatch_trigger_usd_mwh"], chk_usd, sol["eprice"],
            window_loc, actual_load, months, seeds, cfg_r["four_cp_dispatch_near"],
            fc["watch"])
        cp_reflex_total += cp_n
        grid_d = grid_series(model, lvl_d, ch_d, dis_d, dvfs, 168)

        wk = {}
        for name, grid in (("flat", flat_grid_week), ("plan", grid_p), ("dispatch", grid_d)):
            cost = float(np.sum(grid * price_paid))
            energy[name] += cost
            wk[name] = round(cost)
            for ts, g in zip(window, grid):
                strat_grid[name][ts] = float(g)
        base_ops = sol["friction_usd"] + sol["dvfs_usd"] + sol["batt_usd"]
        op_cost["plan"] += base_ops
        extra_batt = float(np.maximum(dis_d - dis_p, 0).sum()) * sol["batt_cost_usd_mwh"]
        op_cost["dispatch"] += base_ops + fric_x + extra_batt

        deadline_slots += len(model.jobs)
        deadline_met += len(model.jobs) - miss_jobs
        reflex_total += len(reflex)
        makeup_missing += miss

        # carry each world's battery state into its next week
        soc_plan = float(sol["soc"][-1])
        soc = soc_disp
        for h in range(168):
            soc = min(model.battery.energy_mwh,
                      max(model.battery.min_soc_mwh,
                          soc + ch_d[h] * model.battery.rte - dis_d[h]))
        soc_disp = soc

        weekly_rows.append({
            "monday_local": str(t0.tz_convert(LOCAL_TZ).date()),
            "energy_usd": wk,
            "reflex_hours": len(reflex),
            "stress_hours_actual": int(df["is_stress"].reindex(window).fillna(False).sum()),
            "worst_actual_price": round(float(np.nanmax(actual_rt)), 1)
            if np.isfinite(actual_rt).any() else None,
        })
        if (wi + 1) % 20 == 0:
            print(f"[replay]   ...week {wi + 1}/{len(mondays)}")

    # ---------------- scoring on reality
    period = [mondays[0], pd.date_range(mondays[-1], periods=168, freq="h")[-1]]
    in_period = (df.index >= period[0]) & (df.index <= period[1])

    stress_ts = df.index[in_period & df["is_stress"].fillna(False).astype(bool)]
    stress_draw = {}
    for s, g in strat_grid.items():
        vals = [g[ts] for ts in stress_ts if ts in g]
        stress_draw[s] = round(float(np.mean(vals))) if vals else None

    cp_rows = []
    cp_ts = df.index[in_period & df["is_4cp_actual"].fillna(False).astype(bool)]
    for ts in cp_ts:
        loc_ts = ts.tz_convert(LOCAL_TZ)
        row = {"ts_local": loc_ts.isoformat(), "year": loc_ts.year, "month": loc_ts.month}
        for s, g in strat_grid.items():
            v = g.get(ts)
            row[f"draw_{s}"] = None if v is None else round(v)
        cp_rows.append(row)
    rate = cfg_r["four_cp_rate_usd_per_mw_year"]
    fourcp_cost = {}
    for s in strat_grid:
        per_year = {}
        for y in sorted({r["year"] for r in cp_rows}):
            draws = [r[f"draw_{s}"] for r in cp_rows
                     if r["year"] == y and r[f"draw_{s}"] is not None]
            if draws:
                per_year[str(y)] = round(float(np.mean(draws)) * rate)
        fourcp_cost[s] = per_year

    totals = {}
    for s in strat_grid:
        totals[s] = {
            "energy_usd": round(energy[s]),
            "four_cp_usd": round(sum(fourcp_cost[s].values())),
            "op_costs_usd": round(op_cost[s]),
        }
        totals[s]["total_usd"] = sum(totals[s].values())
    for s in ("plan", "dispatch"):
        totals[s]["savings_vs_flat_usd"] = totals["flat"]["total_usd"] - totals[s]["total_usd"]
        totals[s]["savings_vs_flat_pct"] = round(
            100 * totals[s]["savings_vs_flat_usd"] / totals["flat"]["total_usd"], 1)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": bundle["source"],
        "period_local": [str(p.tz_convert(LOCAL_TZ).date()) for p in period],
        "weeks": len(mondays),
        "trained_on": bundle["train_years"],
        "assumptions": {
            "four_cp_rate_usd_per_mw_year": rate,
            "dispatch_trigger_usd_mwh": cfg_r["dispatch_trigger_usd_mwh"],
            "note": "energy at actual realized hub prices; 4CP at actual coincident "
                    "peaks (year cost = rate x mean draw over that year's known CPs); "
                    "op costs are the planner's modeled friction/battery/DVFS charges",
            **({"proof_load": proof_load} if proof_load else {}),
        },
        "totals": totals,
        "stress_hours": {"count": int(len(stress_ts)), "avg_draw_mw": stress_draw},
        "four_cp_events": cp_rows,
        "compliance": {
            "deadline_slots": deadline_slots, "deadline_met": deadline_met,
            "makeup_shortfall_job_hours": round(makeup_missing, 2),
            "reflex_hours_total": reflex_total,
            "reflex_hours_four_cp": cp_reflex_total,
            "floor_mw_always_served": True,
        },
        "weekly": weekly_rows,
    }
    opath = OUT / f"replay{tag}.json"
    opath.write_text(json.dumps(payload, indent=2))

    f, p, d = (totals[s] for s in ("flat", "plan", "dispatch"))
    print(f"\n[replay] SCOREBOARD — {len(mondays)} weeks, "
          f"{payload['period_local'][0]} -> {payload['period_local'][1]}")
    print(f"{'':22s}{'flat':>14s}{'plan':>14s}{'plan+dispatch':>16s}")
    print(f"{'energy (actual $)':22s}{f['energy_usd']:>14,}{p['energy_usd']:>14,}{d['energy_usd']:>16,}")
    print(f"{'4CP exposure':22s}{f['four_cp_usd']:>14,}{p['four_cp_usd']:>14,}{d['four_cp_usd']:>16,}")
    print(f"{'op costs (modeled)':22s}{f['op_costs_usd']:>14,}{p['op_costs_usd']:>14,}{d['op_costs_usd']:>16,}")
    print(f"{'TOTAL':22s}{f['total_usd']:>14,}{p['total_usd']:>14,}{d['total_usd']:>16,}")
    print(f"{'savings vs flat':22s}{'':>14s}{p['savings_vs_flat_usd']:>14,}{d['savings_vs_flat_usd']:>16,}")
    print(f"{'':22s}{'':>14s}{p['savings_vs_flat_pct']:>13.1f}%{d['savings_vs_flat_pct']:>15.1f}%")
    if stress_draw:
        print(f"[replay] actual stress hours ({len(stress_ts)}): avg draw "
              f"flat {stress_draw['flat']} -> plan {stress_draw['plan']} "
              f"-> dispatch {stress_draw['dispatch']} MW")
    print(f"[replay] 4CP peaks in period: {len(cp_rows)}; deadlines "
          f"{deadline_met}/{deadline_slots} met; reflex fired {reflex_total} hrs "
          f"({cp_reflex_total} on live 4CP risk); unrecovered {makeup_missing:.1f} job-hrs")
    print(f"[replay] wrote {opath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
