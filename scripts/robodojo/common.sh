#!/usr/bin/env bash
set -euo pipefail
vpp2_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$vpp2_repo"
export PYTHONPATH="${vpp2_repo}/src${PYTHONPATH:+:${PYTHONPATH}}"
python_bin="${PYTHON_BIN:-python}"
