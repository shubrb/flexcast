"""
Phase 3 — the forecaster.

Learns from Phase 2's labeled history to predict, for every hour in the
next 7 days: probability of stress, price level (P10/P50/P90), expected
load, and 4CP watch — plus WHY (top feature drivers per hour).

Honesty rules baked in:
  * every feature is as-of-time causal: archived weather FORECASTS (not
    actuals), and realized history lagged >= the forecast horizon
  * two genuinely different horizons: day-ahead (fresh lags + weather
    forecast) and week-ahead (older lags + climatology — no pretending
    we have a good 7-day weather forecast)
  * chronological split: train on early years, grade on held-out recent
    years the model never saw, against naive baselines it must beat
  * probabilities are isotonic-calibrated on walk-forward out-of-fold
    predictions, so "30%" means 30%

    python -m engines.forecast                 # train + evaluate + save bundle
    python -m engines.forecast --predict       # write data/out/forecast.json
    python -m engines.forecast --synthetic     # same, on the dev cache
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import OUT, RAW, RAW_SYNTH, load_site
from .labeler import FOUR_CP_NEAR, FOUR_CP_WINDOW, LOCAL_TZ

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 7
HORIZON_LAG = {"1d": 24, "7d": 168}   # min hours between a feature and the target hour
DEFAULT_TEST_YEARS = [2025, 2026]

CAL = ["hour", "dow", "month", "doy_sin", "doy_cos", "is_weekend"]
F1 = CAL + ["fc_temp", "fc_wind", "fc_rad", "fc_temp_daymax",
            "rt_mean_l24", "rt_max_l24", "load_l24", "zone_load_l24", "da_l24",
            "rt_mean_l48", "rt_mean_l168", "load_l168",
            "rt_mean_7d_l24", "rt_max_7d_l24", "load_7d_l24", "spikes_7d_l24",
            "mtd_peak_l24"]
F7 = CAL + ["cl_temp", "cl_wind", "cl_rad",
            "rt_mean_l168", "rt_max_l168", "load_l168", "zone_load_l168", "da_l168",
            "rt_mean_l336", "load_l336",
            "rt_mean_7d_l168", "rt_max_7d_l168", "load_7d_l168", "spikes_7d_l168",
            "mtd_peak_l168"]
FEATURES = {"1d": F1, "7d": F7}

FRIENDLY = {
    "fc_temp": "forecast temp (°C)", "fc_wind": "forecast wind (km/h)",
    "fc_rad": "forecast solar (W/m²)", "fc_temp_daymax": "day's forecast high (°C)",
    "cl_temp": "seasonal-normal temp", "cl_wind": "seasonal-normal wind",
    "cl_rad": "seasonal-normal solar",
    "hour": "hour of day", "dow": "day of week", "month": "month",
    "doy_sin": "time of year", "doy_cos": "time of year", "is_weekend": "weekend",
    "rt_mean_l24": "RT price yesterday ($)", "rt_max_l24": "RT spike yesterday ($)",
    "rt_mean_l48": "RT price 2 days ago ($)",
    "rt_mean_l168": "RT price last week ($)", "rt_max_l168": "RT spike last week ($)",
    "rt_mean_l336": "RT price 2 weeks ago ($)",
    "load_l24": "load yesterday (MW)", "load_l168": "load last week (MW)",
    "load_l336": "load 2 weeks ago (MW)",
    "zone_load_l24": "West-zone load yesterday", "zone_load_l168": "West-zone load last week",
    "da_l24": "DA price yesterday ($)", "da_l168": "DA price last week ($)",
    "rt_mean_7d_l24": "avg RT price, past week ($)", "rt_mean_7d_l168": "avg RT price, prior week ($)",
    "rt_max_7d_l24": "worst RT spike, past week ($)", "rt_max_7d_l168": "worst RT spike, prior week ($)",
    "load_7d_l24": "avg load, past week (MW)", "load_7d_l168": "avg load, prior week (MW)",
    "spikes_7d_l24": "spike hours, past week", "spikes_7d_l168": "spike hours, prior week",
    "mtd_peak_l24": "month-to-date peak (MW)", "mtd_peak_l168": "month-to-date peak (MW)",
}


# ---------------------------------------------------------------- assembly

def load_labels(tag: str) -> pd.DataFrame:
    d = OUT / f"labels{tag}"
    files = sorted(d.glob("[0-9]*.parquet"))
    if not files:
        raise SystemExit(f"[forecast] no labels under {d} — run engines.labeler first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["interval_start"] = pd.to_datetime(df["interval_start"], utc=True)
    return df.sort_values("interval_start")


def load_weather(root) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (forecast_archive, actuals), each indexed by UTC hour."""
    files = sorted((root / "weather").glob("[0-9]*.parquet"))
    if not files:
        return pd.DataFrame(), pd.DataFrame()
    w = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    w["interval_start"] = pd.to_datetime(w["interval_start"], utc=True)
    cols = ["temperature_2m", "wind_speed_100m", "shortwave_radiation"]
    out = []
    for kind in ("forecast", "actual"):
        k = (w[w["kind"] == kind].groupby("interval_start")[cols].mean()
             if "kind" in w.columns else pd.DataFrame())
        out.append(k)
    return out[0], out[1]


def build_frame(tag: str, root, extend_to: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per hour on a complete hourly grid: targets + raw inputs."""
    lab = load_labels(tag).set_index("interval_start")
    fc, _ = load_weather(root)
    end = lab.index.max() if extend_to is None else max(lab.index.max(), extend_to)
    idx = pd.date_range(lab.index.min(), end, freq="h", tz="UTC")
    df = lab.reindex(idx)
    df.index.name = "interval_start"
    for src, dst in [("temperature_2m", "fc_temp"), ("wind_speed_100m", "fc_wind"),
                     ("shortwave_radiation", "fc_rad")]:
        df[dst] = fc[src].reindex(idx) if len(fc) else np.nan
    return df


def add_features(df: pd.DataFrame, spike_usd: float) -> pd.DataFrame:
    """Calendar + causally-lagged history features. Index must be hourly-complete."""
    idx = df.index
    loc = idx.tz_convert(LOCAL_TZ)
    doy = loc.dayofyear.values
    f: dict[str, object] = {
        "hour": loc.hour.values.astype("int16"),
        "dow": loc.dayofweek.values.astype("int16"),
        "month": loc.month.values.astype("int16"),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "is_weekend": (loc.dayofweek.values >= 5).astype("int8"),
        "fc_temp_daymax": pd.Series(df["fc_temp"].values, index=idx)
                            .groupby(loc.date).transform("max").values,
    }
    rt, rtx = df["rt_price_mean"], df["rt_price_max"]
    ld, zld, da = df["load_mw"], df["zone_load_mw"], df["da_price"]
    spike = (rtx >= spike_usd).astype("float")
    for lag in (24, 48, 168, 336):
        f[f"rt_mean_l{lag}"] = rt.shift(lag).values
    for lag in (24, 168):
        f[f"rt_max_l{lag}"] = rtx.shift(lag).values
        f[f"da_l{lag}"] = da.shift(lag).values
        f[f"zone_load_l{lag}"] = zld.shift(lag).values
        f[f"rt_mean_7d_l{lag}"] = rt.shift(lag).rolling(168, min_periods=24).mean().values
        f[f"rt_max_7d_l{lag}"] = rtx.shift(lag).rolling(168, min_periods=24).max().values
        f[f"load_7d_l{lag}"] = ld.shift(lag).rolling(168, min_periods=24).mean().values
        f[f"spikes_7d_l{lag}"] = spike.shift(lag).rolling(168, min_periods=24).sum().values
    for lag in (24, 168, 336):
        f[f"load_l{lag}"] = ld.shift(lag).values
    # month-to-date load peak, as known `lag` hours ahead of the target hour
    mo = pd.Series(loc.month.values, index=idx)
    yr = pd.Series(loc.year.values, index=idx)
    mtd = ld.groupby([yr.values, mo.values]).cummax()
    for lag in (24, 168):
        same_month = mo.shift(lag) == mo
        f[f"mtd_peak_l{lag}"] = mtd.shift(lag).where(same_month).values
    return pd.concat([df, pd.DataFrame(f, index=idx)], axis=1)


def climatology(actuals: pd.DataFrame, train_years: list[int]) -> pd.DataFrame:
    """(month, hour) -> mean actual temp/wind/rad, TRAIN years only (no leakage)."""
    if actuals.empty:
        return pd.DataFrame()
    loc = actuals.index.tz_convert(LOCAL_TZ)
    a = actuals[pd.Index(loc.year).isin(train_years)]
    loc = a.index.tz_convert(LOCAL_TZ)
    cl = a.groupby([pd.Index(loc.month, name="month"),
                    pd.Index(loc.hour, name="hour")]).mean()
    return cl.rename(columns={"temperature_2m": "cl_temp", "wind_speed_100m": "cl_wind",
                              "shortwave_radiation": "cl_rad"})


def apply_climatology(df: pd.DataFrame, cl: pd.DataFrame) -> pd.DataFrame:
    key = pd.MultiIndex.from_arrays([df["month"].values, df["hour"].values])
    for c in ("cl_temp", "cl_wind", "cl_rad"):
        df[c] = cl[c].reindex(key).values if len(cl) else np.nan
    return df


# ---------------------------------------------------------------- models

def _lgb_cls():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          min_child_samples=40, subsample=0.9, subsample_freq=1,
                          colsample_bytree=0.9, random_state=SEED, verbose=-1)


def _lgb_reg(alpha: float | None = None):
    from lightgbm import LGBMRegressor
    kw = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
              min_child_samples=40, subsample=0.9, subsample_freq=1,
              colsample_bytree=0.9, random_state=SEED, verbose=-1)
    if alpha is not None:
        kw.update(objective="quantile", alpha=alpha)
    return LGBMRegressor(**kw)


def walk_forward_calibration(X: pd.DataFrame, y: pd.Series, halves: pd.Series):
    """Out-of-fold probs via expanding half-year folds -> isotonic map."""
    from sklearn.isotonic import IsotonicRegression
    uniq = sorted(halves.unique())
    oof = pd.Series(np.nan, index=X.index)
    for i in range(1, len(uniq)):
        tr, te = halves.isin(uniq[:i]), halves == uniq[i]
        if y[tr].sum() < 10 or y[tr].nunique() < 2:
            continue
        m = _lgb_cls().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    ok = oof.notna()
    if ok.sum() < 200 or y[ok].sum() < 30:
        return None  # not enough evidence to calibrate — use raw probs
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(oof[ok].values, y[ok].values)
    return iso


def _cal(iso, p: np.ndarray) -> np.ndarray:
    return iso.predict(p) if iso is not None else p


# ---------------------------------------------------------------- training

def train(df: pd.DataFrame, actuals: pd.DataFrame, test_years: list[int],
          spike_usd: float, cfg_months: list[int], tag: str) -> dict:
    loc_year = pd.Series(df.index.tz_convert(LOCAL_TZ).year, index=df.index)
    have_target = df["is_stress"].notna()
    years_present = sorted(int(y) for y in loc_year[have_target].unique())
    test_years = [y for y in test_years if y in years_present]
    train_years = [y for y in years_present if y not in test_years and
                   (not test_years or y < min(test_years))]
    if not train_years:
        raise SystemExit(f"[forecast] no training years left (present: {years_present})")
    print(f"[forecast] train {train_years} -> test {test_years or 'none'}")

    cl = climatology(actuals, train_years)
    df = apply_climatology(df, cl)

    tr_mask = loc_year.isin(train_years)
    halves = loc_year * 10 + (df["month"].values > 6)
    y_stress = df["is_stress"].astype("float")
    y_price = np.arcsinh(df["rt_price_mean"])
    y_load = df["load_mw"]

    models: dict[tuple[str, str], object] = {}
    iso: dict[str, object] = {}
    for hz, feats in FEATURES.items():
        X = df[feats]
        m_s = tr_mask & y_stress.notna()
        iso[hz] = walk_forward_calibration(X[m_s], y_stress[m_s], halves[m_s])
        models[("stress", hz)] = _lgb_cls().fit(X[m_s], y_stress[m_s])
        m_p = tr_mask & y_price.notna()
        for q in (0.1, 0.5, 0.9):
            models[(f"price_q{int(q*100)}", hz)] = _lgb_reg(q).fit(X[m_p], y_price[m_p])
        m_l = tr_mask & y_load.notna()
        models[("load", hz)] = _lgb_reg().fit(X[m_l], y_load[m_l])
        print(f"[forecast]   {hz}: fitted stress/priceq10-50-90/load on "
              f"{int(m_s.sum()):,} hrs (calibration: {'isotonic' if iso[hz] else 'raw'})")

    bundle = {
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "synthetic" if tag else "real",
        "train_years": train_years, "test_years": test_years,
        "features": FEATURES, "models": models, "iso": iso, "climo": cl,
        "spike_usd": spike_usd, "cfg_months": cfg_months,
    }
    bundle["eval"] = evaluate(df, bundle, loc_year) if test_years else {}
    return bundle


# ---------------------------------------------------------------- evaluation

def evaluate(df: pd.DataFrame, bundle: dict, loc_year: pd.Series) -> dict:
    from sklearn.metrics import roc_auc_score
    test = loc_year.isin(bundle["test_years"])
    y_s = df["is_stress"].astype("float")
    rt = df["rt_price_mean"]
    ld = df["load_mw"]
    # climatological stress rate from TRAIN years, mapped onto test hours
    tr = loc_year.isin(bundle["train_years"]) & y_s.notna()
    climo_rate = y_s[tr].groupby([df.loc[tr, "month"].values,
                                  df.loc[tr, "hour"].values]).mean()
    key = pd.MultiIndex.from_arrays([df["month"].values, df["hour"].values])
    p_climo = pd.Series(climo_rate.reindex(key).values, index=df.index)

    out: dict[str, dict] = {}
    for hz, feats in bundle["features"].items():
        X = df[feats]
        m = test & y_s.notna()
        yb = y_s[m].values
        p = _cal(bundle["iso"][hz],
                 bundle["models"][("stress", hz)].predict_proba(X[m])[:, 1])
        naive = y_s.shift(168)[m]
        nv = naive.notna().values
        k = int(yb.sum())
        top = np.argsort(-p)[:k]
        res = {
            "stress": {
                "test_hours": int(m.sum()), "stress_hours": k,
                "auc": round(float(roc_auc_score(yb, p)), 3),
                "brier": round(float(np.mean((p - yb) ** 2)), 4),
                "brier_naive_lastweek": round(float(np.mean((naive[nv].values - yb[nv]) ** 2)), 4),
                "brier_climo": round(float(np.mean((p_climo[m].fillna(0).values - yb) ** 2)), 4),
                "catch_at_budget": round(float(yb[top].sum() / max(k, 1)), 3),
                "catch_naive": round(float(
                    (naive[nv].values * yb[nv]).sum() / max(yb[nv].sum(), 1)), 3),
            }}
        mp = test & rt.notna()
        q = {f"q{n}": np.sinh(bundle["models"][(f"price_q{n}", hz)].predict(X[mp]))
             for n in (10, 50, 90)}
        lo = np.minimum(np.minimum(q["q10"], q["q50"]), q["q90"])
        hi = np.maximum(np.maximum(q["q10"], q["q50"]), q["q90"])
        naive_p = rt.shift(168)[mp]
        npv = naive_p.notna().values
        res["price"] = {
            "mae_p50": round(float(np.mean(np.abs(q["q50"] - rt[mp].values))), 2),
            "mae_naive_lastweek": round(float(
                np.mean(np.abs(naive_p[npv].values - rt[mp][npv].values))), 2),
            "p10_p90_coverage": round(float(
                np.mean((rt[mp].values >= lo) & (rt[mp].values <= hi))), 3),
        }
        ml = test & ld.notna()
        pl = bundle["models"][("load", hz)].predict(X[ml])
        naive_l = ld.shift(168)[ml]
        nlv = naive_l.notna().values
        res["load"] = {
            "mae_mw": round(float(np.mean(np.abs(pl - ld[ml].values)))),
            "mae_naive_lastweek": round(float(
                np.mean(np.abs(naive_l[nlv].values - ld[ml][nlv].values)))),
        }
        out[hz] = res
    return out


def print_eval(ev: dict) -> None:
    if not ev:
        print("[forecast] no held-out years — skipped evaluation")
        return
    name = {"1d": "day-ahead ", "7d": "week-ahead"}
    for hz, r in ev.items():
        s, p, l = r["stress"], r["price"], r["load"]
        print(f"== {name[hz]}  ({s['test_hours']:,} held-out hrs, {s['stress_hours']} stress)")
        print(f"   stress  AUC {s['auc']:.3f}   Brier {s['brier']:.4f} "
              f"(last-week {s['brier_naive_lastweek']:.4f}, climo {s['brier_climo']:.4f})   "
              f"catch@budget {s['catch_at_budget']:.0%} (naive {s['catch_naive']:.0%})")
        print(f"   price   P50 MAE ${p['mae_p50']:.2f} (naive ${p['mae_naive_lastweek']:.2f})   "
              f"P10-P90 coverage {p['p10_p90_coverage']:.0%}")
        print(f"   load    MAE {l['mae_mw']:,.0f} MW (naive {l['mae_naive_lastweek']:,.0f} MW)")


def print_importance(bundle: dict, k: int = 8) -> None:
    for hz in bundle["features"]:
        m = bundle["models"][("stress", hz)]
        imp = sorted(zip(bundle["features"][hz], m.feature_importances_),
                     key=lambda t: -t[1])[:k]
        print(f"   {hz} stress drivers: " + ", ".join(f"{FRIENDLY.get(f, f)}" for f, _ in imp))


# ---------------------------------------------------------------- predict

def fetch_live_weather(site: dict) -> pd.DataFrame | None:
    """Open-Meteo 7-day forecast for the site. Returns hourly df or None."""
    import requests
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": site["site"]["lat"], "longitude": site["site"]["lon"],
            "hourly": "temperature_2m,wind_speed_100m,shortwave_radiation",
            "forecast_days": 8, "timezone": "UTC"}, timeout=30)
        r.raise_for_status()
        h = r.json()["hourly"]
        w = pd.DataFrame(h).rename(columns={
            "time": "interval_start", "temperature_2m": "fc_temp",
            "wind_speed_100m": "fc_wind", "shortwave_radiation": "fc_rad"})
        w["interval_start"] = pd.to_datetime(w["interval_start"], utc=True)
        return w.set_index("interval_start")
    except Exception as e:  # noqa: BLE001 — forecast must degrade, not die
        print(f"[forecast] live weather fetch failed ({type(e).__name__}) — using climatology")
        return None


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def predict(bundle: dict, tag: str, root, site: dict) -> dict:
    now = pd.Timestamp.now(tz="UTC").floor("h")
    future = pd.date_range(now + pd.Timedelta(hours=1), periods=168, freq="h")
    df = build_frame(tag, root, extend_to=future[-1])

    live = fetch_live_weather(site)
    weather_src = "climatology"
    if live is not None:
        for c in ("fc_temp", "fc_wind", "fc_rad"):
            df.loc[df.index >= now, c] = live[c].reindex(df.index[df.index >= now]).values
        weather_src = "open-meteo-live"
    df = add_features(df, bundle["spike_usd"])
    df = apply_climatology(df, bundle["climo"])
    if weather_src == "climatology":  # no live fetch: climatology stands in for fc_*
        fut = df.index >= now
        for a, b in (("fc_temp", "cl_temp"), ("fc_wind", "cl_wind"), ("fc_rad", "cl_rad")):
            df.loc[fut, a] = df.loc[fut, a].fillna(df.loc[fut, b])
        df.loc[fut, "fc_temp_daymax"] = (
            df.loc[fut, "fc_temp"].groupby(df.index[fut].tz_convert(LOCAL_TZ).date)
              .transform("max"))

    fdf = df.loc[future]
    lead = ((future - now) / pd.Timedelta(hours=1)).astype(int)
    hz_of = np.where(lead <= 48, "1d", "7d")

    p_stress = np.full(len(fdf), np.nan)
    price = {n: np.full(len(fdf), np.nan) for n in (10, 50, 90)}
    load_pred = np.full(len(fdf), np.nan)
    drivers: list[list[dict]] = [[] for _ in range(len(fdf))]
    for hz in ("1d", "7d"):
        sel = hz_of == hz
        if not sel.any():
            continue
        X = fdf.loc[sel, bundle["features"][hz]]
        p_stress[sel] = _cal(bundle["iso"][hz],
                             bundle["models"][("stress", hz)].predict_proba(X)[:, 1])
        for n in (10, 50, 90):
            price[n][sel] = np.sinh(bundle["models"][(f"price_q{n}", hz)].predict(X))
        load_pred[sel] = bundle["models"][("load", hz)].predict(X)
        contrib = bundle["models"][("stress", hz)].booster_.predict(X, pred_contrib=True)[:, :-1]
        for row_i, i in enumerate(np.where(sel)[0]):
            order = np.argsort(-np.abs(contrib[row_i]))[:3]
            drivers[i] = [{
                "feature": bundle["features"][hz][j],
                "label": FRIENDLY.get(bundle["features"][hz][j], bundle["features"][hz][j]),
                "value": _f(X.iloc[row_i, j]),
                "push": "up" if contrib[row_i][j] > 0 else "down",
            } for j in order]
    qlo = np.nanmin([price[10], price[50], price[90]], axis=0)
    qhi = np.nanmax([price[10], price[50], price[90]], axis=0)

    # 4CP watch: causal candidate logic on PREDICTED load vs known + predicted peaks
    loc = future.tz_convert(LOCAL_TZ)
    hist_loc = df.index[df.index <= now].tz_convert(LOCAL_TZ)
    watch = np.zeros(len(fdf), bool)
    months = list(bundle["cfg_months"])
    running: dict[tuple[int, int], float] = {}
    known_load = df.loc[df.index <= now, "load_mw"]
    for (y, m), grp in known_load.groupby([hist_loc.year, hist_loc.month]):
        if m in months and grp.notna().any():
            running[(y, m)] = float(grp.max())
    for i, ts in enumerate(loc):
        if ts.month not in months or not np.isfinite(load_pred[i]):
            continue
        key = (ts.year, ts.month)
        running[key] = max(running.get(key, 0.0), load_pred[i])
        if (FOUR_CP_WINDOW[0] <= ts.hour < FOUR_CP_WINDOW[1]
                and load_pred[i] >= FOUR_CP_NEAR * running[key]):
            watch[i] = True

    last_rt = df.loc[df["rt_price_mean"].notna()].index.max()
    last_ld = df.loc[df["load_mw"].notna() & (df.index <= now)].index.max()
    return {
        "generated_utc": now.isoformat(),
        "source": bundle["source"], "site": site["site"]["name"],
        "hub": site["site"]["price_location"],
        "weather_source": weather_src,
        "trained_utc": bundle["trained_utc"],
        "feature_data_through": {
            "rt_price": None if pd.isna(last_rt) else last_rt.isoformat(),
            "load": None if pd.isna(last_ld) else last_ld.isoformat(),
        },
        "stale_hours": None if pd.isna(last_rt) else int((now - last_rt) / pd.Timedelta(hours=1)),
        "hours": [{
            "ts_utc": future[i].isoformat(),
            "ts_local": loc[i].isoformat(),
            "lead_h": int(lead[i]), "model": str(hz_of[i]),
            "p_stress": _f(round(p_stress[i], 4)) if np.isfinite(p_stress[i]) else None,
            "price_p10": _f(round(qlo[i], 2)) if np.isfinite(qlo[i]) else None,
            "price_p50": _f(round(price[50][i], 2)) if np.isfinite(price[50][i]) else None,
            "price_p90": _f(round(qhi[i], 2)) if np.isfinite(qhi[i]) else None,
            "load_pred_mw": _f(round(load_pred[i])) if np.isfinite(load_pred[i]) else None,
            "four_cp_watch": bool(watch[i]),
            "drivers": drivers[i],
        } for i in range(len(fdf))],
    }


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3: two-horizon stress/price/load forecaster")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--test-years", nargs="+", type=int, default=DEFAULT_TEST_YEARS)
    ap.add_argument("--predict", action="store_true",
                    help="write data/out/forecast.json for the next 168 hours")
    ap.add_argument("--retrain", action="store_true", help="ignore saved bundle")
    ap.add_argument("--site", default=None)
    a = ap.parse_args()

    tag = "_synthetic" if a.synthetic else ""
    root = RAW_SYNTH if a.synthetic else RAW
    site = load_site(a.site)
    cfg = site.get("labels", {})
    spike_usd = float(cfg.get("rt_price_spike_usd", 200))
    bpath = OUT / f"models{tag}" / "bundle.pkl"

    bundle = None
    if a.predict and bpath.exists() and not a.retrain:
        with open(bpath, "rb") as fh:
            bundle = pickle.load(fh)
        print(f"[forecast] loaded bundle trained {bundle['trained_utc']} "
              f"(train {bundle['train_years']})")

    if bundle is None:
        df = build_frame(tag, root)
        df = add_features(df, spike_usd)
        _, actuals = load_weather(root)
        bundle = train(df, actuals, a.test_years, spike_usd,
                       list(cfg.get("four_cp_months", [6, 7, 8, 9])), tag)
        bpath.parent.mkdir(parents=True, exist_ok=True)
        with open(bpath, "wb") as fh:
            pickle.dump(bundle, fh)
        (OUT / f"forecast_eval{tag}.json").write_text(json.dumps(bundle["eval"], indent=2))
        print_eval(bundle["eval"])
        print_importance(bundle)
        print(f"[forecast] bundle -> {bpath}")

    if a.predict:
        payload = predict(bundle, tag, root, site)
        fpath = OUT / f"forecast{tag}.json"
        fpath.write_text(json.dumps(payload, indent=2))
        risky = [h for h in payload["hours"] if (h["p_stress"] or 0) >= 0.25]
        cp = [h for h in payload["hours"] if h["four_cp_watch"]]
        print(f"[forecast] wrote {fpath.name}: 168 hrs, {len(risky)} hrs P(stress)>=25%, "
              f"{len(cp)} 4CP-watch hrs, weather={payload['weather_source']}"
              + (f", features {payload['stale_hours']}h stale" if payload["stale_hours"]
                 and payload["stale_hours"] > 48 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
