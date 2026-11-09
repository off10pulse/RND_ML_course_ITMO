#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.eda --data-root "${1:-data/raw}" --output outputs/eda
python -m src.train --data-root "${1:-data/raw}" --config configs/baseline.yaml --output outputs/baseline
python -m src.report --run-dir outputs/baseline
