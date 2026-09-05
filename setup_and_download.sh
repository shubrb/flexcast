#!/bin/bash
# FlexCast one-command setup + real ERCOT download.
# Run from your normal macOS Terminal:
#   bash ~/Documents/flexcast/setup_and_download.sh
# Safe to re-run anytime: finished years are skipped automatically.
set -e
cd "$(dirname "$0")"

echo "==> [1/4] Python environment"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
# macOS Pythons often ship without CA certs wired up; use certifi's bundle
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

echo "==> [2/4] Quick probe (one real yearly price file) to fail fast"
python -m engines.download --years 2024 2024 --datasets dam_spp

echo "==> [3/4] Full download 2021-2026 (15-30 min; Ctrl-C safe, re-run resumes)"
python -m engines.download --years 2021 2026

echo "==> [4/4] Verifying the cache"
python -m engines.check

echo ""
echo "All done. Data is in $(pwd)/data/raw — tell Claude it's finished."
