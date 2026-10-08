#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exec "$python_bin" -m vpp2.libero.cli audit "$@"
