#!/usr/bin/env bash
# One 0-100k launch on each GPU-bearing role, sharing paths and output_dir.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ "${1:-}" == --help || "${1:-}" == help ]]; then
  echo 'Usage: bash scripts/robodojo/train.sh [--dry-run] [key=value ...]'
  echo 'Default: configs/robodojo/train_100k.yaml; continuous joint training to 100k.'
  exit 0
fi
# Managed-platform WORLD_SIZE/RANK are node counts/ranks before Accelerate launch.
export NNODES="${NNODES:-${WORLD_SIZE:-12}}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if [[ -z "${NODE_RANK:-}" && -n "${RANK:-}" ]]; then export NODE_RANK="$RANK"; fi
export VPP2_TRAIN_MODULE=vpp2.cli
export VPP2_ACCELERATE_CONFIG=configs/robodojo/accelerate.yaml
export REQUIRE_RDMA="${REQUIRE_RDMA:-1}"
exec bash scripts/train.sh --config configs/robodojo/train_100k.yaml "$@"
