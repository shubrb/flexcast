"""
Cache health check — run after (or during) the download.

Scans BOTH data roots and labels them clearly:
  data/raw            REAL downloaded history   (slide numbers come from here)
  data/raw-synthetic  synthetic dev data        (never for slides)

    python -m engines.check
"""
from __future__ import annotations

import pandas as pd

from .config import RAW, RAW_SYNTH


def _scan(root, label: str) -> tuple[int, int]:
    """Returns (files, problems)."""
    files = sorted(root.glob("*/*.parquet")) if root.exists() else []
    if not files:
        print(f"== {label}: empty")
        return 0, 0
    print(f"== {label}")
    print(f"{'dataset':12s} {'year':6s} {'rows':>10s}  {'from':>12s} {'to':>12s}  note")
    print("-" * 72)
    problems = 0
    for f in files:
        df = pd.read_parquet(f, columns=["interval_start"])
        ts = pd.to_datetime(df["interval_start"], utc=True).sort_values()
        uniq = ts.drop_duplicates()  # long formats repeat timestamps per location
        note = ""
        if len(ts) == 0:
            note, problems = "EMPTY", problems + 1
        else:
            step = uniq.diff().median()
            if step is not pd.NaT and pd.Timedelta(0) < step <= pd.Timedelta(hours=1):
                expected = int((uniq.iloc[-1] - uniq.iloc[0]) / step) + 1
                missing = expected - len(uniq)
                if missing > max(24, expected * 0.02):
                    note, problems = f"~{missing} intervals missing", problems + 1
        print(f"{f.parent.name:12s} {f.stem:6s} {len(df):>10,}  "
              f"{str(ts.iloc[0].date()) if len(ts) else '-':>12s} "
              f"{str(ts.iloc[-1].date()) if len(ts) else '-':>12s}  {note}")
    print("-" * 72)
    return len(files), problems


def main() -> int:
    n_real, p_real = _scan(RAW, "REAL data (data/raw)")
    print()
    n_syn, p_syn = _scan(RAW_SYNTH, "SYNTHETIC dev data (data/raw-synthetic)")
    print()
    if n_real == 0:
        print("[check] no REAL data yet — run engines.download on a normal connection")
    problems = p_real + p_syn
    print(f"[check] {'OK — caches look healthy' if problems == 0 else f'{problems} problem(s) — see notes above'}")
    return 0 if problems == 0 and n_real + n_syn > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
