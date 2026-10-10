#!/usr/bin/env bash
# Prepare condition images and enhance their instructions with Qwen.
set -euo pipefail
vpp2_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$vpp2_repo"
export PYTHONPATH="${vpp2_repo}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN:-python}" -m vpp2.enhance_prompt "$@"
