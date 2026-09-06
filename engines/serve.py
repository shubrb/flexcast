"""
Phase 7 — the FlexCast server: static console + LIVE optimizer.

Serves the repo (so the console works exactly as with http.server) and adds
one endpoint the "Design your own site" panel calls:

    POST /api/plan   {name, total_mw, floor_mw, batt_mwh, batt_mw, work_scale}

The server scales the demo case's rung ladder and job queue to the posted
campus, lints it with the same rules as engines.site (impossible sites come
back as readable errors, not crashes), solves the SAME weekly LP the real
planner uses against the CURRENT forecast, and returns the plan — typically
in well under a second. Every number the console then shows for a custom
site is a genuine optimization, not front-end math.

    source .venv/bin/activate
    python -m engines.serve             # http://localhost:8000/app/
    python -m engines.serve --synthetic --port 8742
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from .config import OUT, REPO_ROOT, load_site
from .planner import DEFAULTS, flat_baseline, playbook_hour, read_forecast, solve
from .site import model_from_dict

STATE: dict = {}
REFRESH_LOCK = threading.Lock()


def refresh_outputs(synthetic: bool) -> dict:
    """Re-anchor the forecast to NOW and re-solve the plan + what-if scenarios.

    Runs the same engines the CLI does (so every number stays a real model
    output / real LP solve), then reloads the forecast the live optimizer
    uses. Needs no internet: weather falls back to climatology and the
    forecast honestly records how stale its input features are."""
    args = ["--synthetic"] if synthetic else []
    t0 = time.time()
    steps = [("forecast", ["-m", "engines.forecast", "--predict"] + args),
             ("planner", ["-m", "engines.planner"] + args),
             ("scenarios", ["-m", "engines.scenarios"] + args)]
    for name, cmd in steps:
        r = subprocess.run([sys.executable] + cmd, cwd=str(REPO_ROOT),
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
            raise RuntimeError(f"{name} failed: " + " | ".join(tail))
    tag = "_synthetic" if synthetic else ""
    STATE["fc"] = read_forecast(OUT / f"forecast{tag}.json")
    fc0 = json.loads((OUT / f"forecast{tag}.json").read_text())
    return {"ok": True, "seconds": round(time.time() - t0, 1),
            "starts": fc0["hours"][0]["ts_local"],
            "weather": fc0.get("weather_source"),
            "stale_hours": fc0.get("stale_hours")}


def custom_site_dict(p: dict) -> dict:
    """Scale the loaded demo case to the requested campus."""
    base = copy.deepcopy(STATE["base_site"])
    total = max(50.0, float(p.get("total_mw", 500)))
    floor = min(max(10.0, float(p.get("floor_mw", 150))), total - 20)
    flex = total - floor
    batt_mwh = max(0.0, float(p.get("batt_mwh", 200)))
    batt_mw = max(0.0, float(p.get("batt_mw", 100)))
    work = min(2.0, max(0.25, float(p.get("work_scale", 1.0))))
    base_flex = float(base["power"]["flexible_training_mw"]) or 1.0
    k = flex / base_flex

    base["site"]["name"] = str(p.get("name") or "Custom site")[:48]
    base["power"] = {"total_mw": total, "inference_floor_mw": floor,
                     "flexible_training_mw": flex}
    base["battery"] = {"energy_mwh": batt_mwh, "power_mw": batt_mw,
                       "round_trip_efficiency": .88, "min_soc": .10}
    for r in base["flex_rungs"]:
        r["mw"] = round(batt_mw if r["name"] == "battery" else r["mw"] * k, 1)
    for j in base["jobs"]:
        j["mw"] = round(j["mw"] * k, 1)
        # cap scaled hours below each job's own deadline window so the
        # workload slider can push utilization up without going infeasible
        j["hours"] = round(min(j["hours"] * work, j["deadline_days"] * 24 * 0.92), 1)
    return base


def solve_custom(p: dict) -> tuple[int, dict]:
    model, warn, fatal = model_from_dict(custom_site_dict(p))
    if fatal:
        return 422, {"errors": fatal, "warnings": warn}
    fc = STATE["fc"]
    cfg = {**DEFAULTS, **model.planner_cfg}
    sol = solve(model, fc, cfg)
    H = fc["H"]
    flat_jobs = flat_baseline(model, H)
    flat_grid = [model.floor_mw + sum(h.values()) for h in flat_jobs]
    e_plan = sum(sol["grid"][t] * sol["eprice"][t] for t in range(H))
    cp_plan = sum(sol["grid"][t] * sol["cp"][t] for t in range(H))
    e_flat = sum(flat_grid[t] * sol["eprice"][t] for t in range(H))
    cp_flat = sum(flat_grid[t] * sol["cp"][t] for t in range(H))
    extras = sol["friction_usd"] + sol["dvfs_usd"] + sol["batt_usd"]
    plan_cost, flat_cost = e_plan + cp_plan + extras, e_flat + cp_flat
    hours = []
    for t in range(H):
        jobs = {j.id: round(j.mw * sol["run"].get((j.id, t), 0.0), 1)
                for j in model.jobs if sol["run"].get((j.id, t), 0.0) > 1e-4}
        hours.append({
            "ts_local": fc["ts_local"][t], "jobs_mw": jobs,
            "training_mw": round(sum(jobs.values()), 1),
            "battery_charge_mw": round(sol["ch"][t], 1),
            "battery_discharge_mw": round(sol["dis"][t], 1),
            "battery_soc_mwh": round(sol["soc"][t], 1),
            "dvfs_shed_mw": round(sol["dvfs"][t], 1),
            "grid_mw": round(sol["grid"][t], 1),
            "grid_flat_baseline_mw": round(flat_grid[t], 1),
            "playbook": playbook_hour(model, sol, t),
        })
    return 200, {
        "site": model.name, "warnings": warn,
        "params": {"total_mw": model.floor_mw + model.flex_mw,
                   "floor_mw": model.floor_mw,
                   "batt_mwh": model.battery.energy_mwh,
                   "batt_mw": model.battery.power_mw},
        "totals": {
            "plan_cost_usd": round(plan_cost),
            "flat_baseline_cost_usd": round(flat_cost),
            "savings_usd": round(flat_cost - plan_cost),
            "savings_pct": round(100 * (flat_cost - plan_cost) / flat_cost, 1)
            if flat_cost else 0,
        },
        "hours": hours,
    }


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):  # keep the terminal quiet
        pass

    def do_POST(self):
        if self.path not in ("/api/plan", "/api/repredict"):
            self.send_error(404)
            return
        try:
            if self.path == "/api/repredict":
                if not REFRESH_LOCK.acquire(blocking=False):
                    code, payload = 409, {"errors": ["a re-predict is already running"]}
                else:
                    try:
                        payload = refresh_outputs(STATE["synthetic"])
                        code = 200
                        print(f"[serve] re-anchored to now in {payload['seconds']}s "
                              f"(week starts {payload['starts'][:16]})")
                    finally:
                        REFRESH_LOCK.release()
            else:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(min(n, 65536)) or b"{}")
                code, payload = solve_custom(body)
        except Exception as e:  # noqa: BLE001 — a demo server never dies
            code, payload = 400, {"errors": [f"{type(e).__name__}: {e}"]}
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _refresh_and_report(synthetic: bool) -> None:
    print("[serve] re-anchoring the forecast to now (models + cached data, no download)…")
    try:
        with REFRESH_LOCK:
            info = refresh_outputs(synthetic)
        print(f"[serve] week now starts {info['starts'][:16]} · weather={info['weather']}"
              + (f" · features {info['stale_hours']}h stale" if (info["stale_hours"] or 0) > 48 else "")
              + f" · {info['seconds']}s")
    except Exception as e:  # noqa: BLE001
        print(f"[serve] re-anchor failed ({e}) — serving the existing files")


def main() -> int:
    # On a cloud host (Render sets $PORT) bind the socket immediately and
    # re-anchor in a background thread, so the platform's health check and
    # first visitors aren't stuck behind a 60s model run.
    cloud = "PORT" in os.environ
    ap = argparse.ArgumentParser(description="FlexCast console + live-solve API")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--host", default=os.environ.get("HOST")
                    or ("0.0.0.0" if cloud else "127.0.0.1"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-refresh", action="store_true",
                    help="serve the existing forecast.json as-is (skip re-anchoring)")
    a = ap.parse_args()
    tag = "_synthetic" if a.synthetic else ""
    STATE["synthetic"] = a.synthetic
    STATE["base_site"] = load_site()
    fpath = OUT / f"forecast{tag}.json"
    if not a.no_refresh and not cloud:
        _refresh_and_report(a.synthetic)
    if "fc" not in STATE:
        if not fpath.exists():
            raise SystemExit(f"[serve] {fpath} missing — run engines.forecast --predict first")
        STATE["fc"] = read_forecast(fpath)
    if not a.no_refresh and cloud:
        threading.Thread(target=_refresh_and_report, args=(a.synthetic,), daemon=True).start()
    handler = partial(Handler, directory=str(REPO_ROOT))
    srv = ThreadingHTTPServer((a.host, a.port), handler)
    print(f"[serve] FlexCast console at http://localhost:{a.port}/app/"
          f"{'?synthetic=1' if a.synthetic else ''}")
    print("[serve] live optimizer ready at POST /api/plan — Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
