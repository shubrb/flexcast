"""Shared config + paths for all FlexCast engines."""
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
RAW = DATA / "raw"                    # REAL cached history (parquet per dataset-year)
RAW_SYNTH = DATA / "raw-synthetic"    # schema-matched synthetic data for offline dev ONLY
LIVE = DATA / "live"                  # live ticker json
OUT = DATA / "out"                    # engine outputs (labels, forecasts, plans, replays)

for p in (RAW, RAW_SYNTH, LIVE, OUT):
    p.mkdir(parents=True, exist_ok=True)


def load_site(path: Path | None = None) -> dict:
    """Load site.yaml — the single source of truth for the demo site."""
    with open(path or REPO_ROOT / "site.yaml") as f:
        return yaml.safe_load(f)


def raw_path(dataset: str, year: int) -> Path:
    d = RAW / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{year}.parquet"


def synth_path(dataset: str, year: int):
    d = RAW_SYNTH / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{year}.parquet"
