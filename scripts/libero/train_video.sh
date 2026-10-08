#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ "${1:-}" == --help || "${1:-}" == help ]]; then
  echo 'Usage: bash scripts/libero/train_video.sh [--dry-run] [key=value ...]'
  echo 'Default: configs/libero/train_video.yaml; NPROC_PER_NODE=8.'
  exit 0
fi
export MASTER_PORT="${MASTER_PORT:-29611}"
export VPP2_TRAIN_MODULE=vpp2.libero.cli
export VPP2_ACCELERATE_CONFIG=configs/libero/accelerate_zero2.yaml
exec bash scripts/train.sh --config configs/libero/train_video.yaml "$@"
