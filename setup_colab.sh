#!/usr/bin/env bash
# Setup Google Colab environment (16GB-24GB VRAM) for Baseline (SD2) & Proposed (Open-World Amodal SD2).
# Usage on Colab:
#   bash setup_colab.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 scripts/setup_colab.py "$@"

