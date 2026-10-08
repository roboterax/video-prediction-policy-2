#!/usr/bin/env bash
# Run on every GPU-bearing role: one master plus NNODES-1 workers.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-python}"
train_module="${VPP2_TRAIN_MODULE:-vpp2.cli}"
accelerate_config="${VPP2_ACCELERATE_CONFIG:-configs/robodojo/accelerate.yaml}"
export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
nodes="${NNODES:-1}"
per_node="${NPROC_PER_NODE:-8}"
node_rank="${NODE_RANK:-0}"
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29500}"
for number in "$nodes" "$per_node"; do
  [[ "$number" =~ ^[1-9][0-9]*$ ]] || { echo 'GPU and node counts must be positive integers' >&2; exit 2; }
done
[[ "$node_rank" =~ ^[0-9]+$ ]] && (( node_rank < nodes )) || { echo 'Invalid NODE_RANK' >&2; exit 2; }
for argument in "$@"; do
  if [[ "$argument" == --dry-run ]]; then
    exec env WORLD_SIZE="$((nodes * per_node))" "$python_bin" -m "$train_module" train "$@"
  fi
done
if (( nodes > 1 )); then
  : "${NODE_RANK:?Set NODE_RANK on every GPU-bearing role}"
  : "${MASTER_ADDR:?Set the reachable master address}"
fi
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
tmux list-sessions 2>/dev/null || true
"$python_bin" - "$per_node" <<'PY'
import sys, torch
expected = int(sys.argv[1])
observed = torch.cuda.device_count()
if observed != expected:
    raise SystemExit(f"Expected {expected} visible GPUs on this role; found {observed}")
PY
if [[ "${REQUIRE_RDMA:-0}" == 1 ]]; then
  [[ -d /dev/infiniband ]] || { echo 'RDMA devices are missing' >&2; exit 1; }
  ibv_devinfo >/dev/null
fi
if [[ -n "${ARNOLD_RDMA_DEVICE:-}" ]]; then
  export NCCL_IB_HCA="${ARNOLD_RDMA_DEVICE}"
fi
if [[ "$train_module" == vpp2.cli ]]; then
  export VPP2_TORCH_LOAD_MMAP=1
fi
exec "$python_bin" -m accelerate.commands.launch \
  --config_file "$accelerate_config" \
  --num_machines "$nodes" --num_processes "$((nodes * per_node))" \
  --machine_rank "$node_rank" --main_process_ip "$master_addr" \
  --main_process_port "$master_port" \
  --deepspeed_multinode_launcher standard --same_network \
  -m "$train_module" train "$@"
