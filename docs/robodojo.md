# RoboDojo: joint Action2B training and testing

The training recipe starts from **RoboDojo history-conditioned Video-10k** and
trains joint Video + a freshly initialized Action2B for 100,000 steps.

The published inference checkpoint is **joint Video + Action2B step 100,000**, evaluated with
**shift 1, 10 Euler steps, seed 1, horizon 32 / execute 24, history 8 / stride 25**.
Joint refers to training; inference prefills the Video cache once and denoises Action.

## Entry points

Install the dependencies, then run these Bash scripts from the checkout or use
their absolute paths from another directory. They load `src/vpp2` directly; no
project package installation is needed. Relative data/config paths resolve from
the repository root.
Set `PYTHON_BIN` when the policy interpreter is not the current `python`.

| Script | Operation |
|---|---|
| `scripts/robodojo/prepare.sh` | Validate converted EE16 data, fixed split and critical sampling index |
| `scripts/robodojo/text_cache.sh` | Cache language embeddings |
| `scripts/robodojo/init_action.sh` | Initialize the fresh Action2B expert |
| `scripts/robodojo/install_adapter.sh` | Copy the VPP2 adapter into RoboDojo/XPolicyLab |
| `scripts/robodojo/train.sh` | Continuous joint 0–100k using `configs/robodojo/train_100k.yaml` |
| `scripts/robodojo/export.sh` | Export paired Video/Action weights, stats and manifest; default step 100k |
| `scripts/robodojo/configure_eval.sh` | Generate a local, one-client, full 54-entry evaluation configuration |
| `scripts/robodojo/server.sh` | Validate bundle and start the XPolicyLab VPP2 policy server |
| `scripts/robodojo/eval.sh` | Run native-layout closed-loop evaluation and exact official metric audit |

`train`, `server`, and `eval` accept `--dry-run`. Train dry runs do not query or
allocate GPUs; server dry runs check the supplied bundle and installed adapter
without loading tensor weights. Configuration generation refuses an existing file.
Training runs and new evaluations must use distinct output directories and IDs.

## 1. Install and download

### Policy environment

Use Linux, Python 3.10 and an NVIDIA driver compatible with the CUDA/PyTorch
build. The following uses the versions from the tested policy environment:

```bash
conda create -n vpp2 python=3.10 -y
conda activate vpp2
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-train.txt -c environment-reference.txt
python -m pip check
bash scripts/robodojo/train.sh --help
```

For inference only, use `requirements.txt`. Install `wandb>=0.19` if using W&B.
[environment-reference.txt](../environment-reference.txt) records the observed
dependency versions. Training also needs compiler/CUDA development tools for
DeepSpeed; distributed workers need shared storage and reachable rendezvous ports.

Install RoboDojo/Isaac Sim in a separate environment using the
[upstream installation guide](https://robodojo-benchmark.com/doc/usage/install-and-download/)
and the [pinned simulator setup](evaluation.md#simulator-and-policy-installation).
Keep the RGB ordering/reset fixes, assets and seed-1 layouts consistent. Simulator
hosts need `tmux` and `ffmpeg`; remote scheduling also needs noninteractive SSH.

### Checkpoint downloads

The released weights are available on public
[Hugging Face](https://huggingface.co/Haodong082399/VPP2) and
[ModelScope](https://www.modelscope.cn/models/haodong123/VPP2_preview),
with identical checkpoint paths.
Use a ModelScope account with access. Install the ModelScope client in a download
environment if needed (`python -m pip install modelscope`).

For evaluation, use the **paired joint Video + Action2B 100k checkpoint**:

| Path in the model repository | Required for evaluation |
|---|---|
| `checkpoints/joint2b_s100000/video.pt` | Joint-trained Video backbone, step 100k |
| `checkpoints/joint2b_s100000/action.pt` | Matching Action2B, step 100k |
| `checkpoints/joint2b_s100000/` support files | Bundled statistics, configuration and manifest |
| `checkpoints/Wan2.1-I2V-14B-480P/` | Shared VAE, CLIP, UMT5 and tokenizer |

Download the complete bundle and shared encoders with Hugging Face
(`python -m pip install -U huggingface_hub` in a download environment):

```bash
hf download Haodong082399/VPP2 --local-dir weights/robodojo_release \
  --include 'checkpoints/joint2b_s100000/**' \
  --include 'checkpoints/Wan2.1-I2V-14B-480P/**'
```

Or use ModelScope:

```bash
modelscope download --model haodong123/VPP2_preview \
  --local_dir weights/robodojo_release \
  --include 'checkpoints/joint2b_s100000/**' \
            'checkpoints/Wan2.1-I2V-14B-480P/**'
```

Then set the local paths:

```bash
export VPP2_BUNDLE="$PWD/weights/robodojo_release/checkpoints/joint2b_s100000"
export VPP2_WAN_ROOT="$PWD/weights/robodojo_release/checkpoints/Wan2.1-I2V-14B-480P"
```

The paired Action/Video files total about 36.85 GB, excluding the shared Wan encoders.
Inference bundles contain no optimizer state. Continue to
[closed-loop evaluation](#5-configure-and-run-closed-loop-testing).

Joint + Action2B training starts directly from **RoboDojo history-conditioned
Video-10k**. This initializer is available on
[Hugging Face](https://huggingface.co/Haodong082399/VPP2/tree/main/checkpoints/initialization)
and [ModelScope](https://modelscope.cn/models/haodong123/VPP2_preview), revision `master`.

| Checkpoint | Path in the model repository | Local path |
|---|---|---|
| RoboDojo history-conditioned Video-10k | `checkpoints/initialization/robodojo_his10k.pt` | `weights/checkpoints/initialization/robodojo_his10k.pt` |

Use `weights/checkpoints/initialization/robodojo_his10k.pt` as `VPP2_VIDEO_INIT`.
The robot-video pretrained backbone checkpoint is a planned release; see the
[TODO list](../README.md#todo).

For training, download Video-10k together with the shared Wan encoders:

```bash
hf download Haodong082399/VPP2 --local-dir weights \
  --include 'checkpoints/initialization/**' --include 'checkpoints/Wan2.1-I2V-14B-480P/**'
```

Or use ModelScope:

```bash
modelscope download --model haodong123/VPP2_preview \
  --local_dir weights \
  --include 'checkpoints/initialization/**' 'checkpoints/Wan2.1-I2V-14B-480P/**'
```

Then set the encoder path:

```bash
export VPP2_WAN_ROOT="$PWD/weights/checkpoints/Wan2.1-I2V-14B-480P"
```

Converted 3500-episode training data is a separate artifact; its download link is
pending.

## 2. Prepare data and initialization

The [training data contract](training.md#required-artifacts) defines the converted
metadata, RGB videos and native frame mapping. State/action is dual-arm absolute
EE16: `[xyz, qwxyz, gripper] × 2`. Use the supplied z-score statistics for all 16
channels, including grippers. Set paths in your shell:

```bash
export VPP2_MEDIA_ROOT=/data/robodojo
export VPP2_PREPARED="$PWD/data/robodojo"
export VPP2_VIDEO_INIT="$PWD/weights/checkpoints/initialization/robodojo_his10k.pt"
export VPP2_ACTION_INIT="$PWD/weights/init/action2b_from_his10k.pt"
bash scripts/robodojo/prepare.sh \
  --metadata "$VPP2_MEDIA_ROOT/full_episode_metadata.csv" \
  --media-root "$VPP2_MEDIA_ROOT" --output "$VPP2_PREPARED"
CUDA_VISIBLE_DEVICES=0 bash scripts/robodojo/text_cache.sh --data "$VPP2_PREPARED" --wan "$VPP2_WAN_ROOT"
bash scripts/robodojo/init_action.sh --video "$VPP2_VIDEO_INIT" --output "$VPP2_ACTION_INIT"
COMMON_ARGS=(
  "paths.wan=$VPP2_WAN_ROOT" "paths.video_init=$VPP2_VIDEO_INIT"
  "paths.action_init=$VPP2_ACTION_INIT" "paths.data=$VPP2_PREPARED"
  "paths.media=$VPP2_MEDIA_ROOT"
)
```

`prepare` retains 3466 training and 34 held-out episodes. It does not convert raw
RoboDojo recordings. Text caching uses a GPU; inspect live GPU processes and tmux
before launching it. Initialization is a large CPU operation.

## 3. Train joint 0–100k

Use the self-contained [100k YAML](../configs/robodojo/train_100k.yaml) and one
launcher. Video loads the released his10k checkpoint; Action2B is initialized by
interpolating that same Video backbone, with fresh action/proprio projections.
Train both together in one run.

| Setting | Value |
|---|---:|
| Global batch size | 288 |
| Training steps | 100,000 |

Set the launcher environment (`NNODES`, `NPROC_PER_NODE`, `NODE_RANK`,
`MASTER_ADDR`, `MASTER_PORT`, `REQUIRE_RDMA`) for your runtime. Adjust
`batch_size` and `gradient_accumulation_steps` to retain global batch 288.
All training processes use the same data, weights and output directory.

```bash
bash scripts/robodojo/train.sh --dry-run "${COMMON_ARGS[@]}"
bash scripts/robodojo/train.sh "${COMMON_ARGS[@]}" \
  output_dir=runs/robodojo_joint2b_100k
```

Paths may instead be edited directly in the YAML's `paths` section. The script
uses that YAML by default, with bf16 and ZeRO-1. Confirm the effective global
batch in the dry-run output before launching training.

For the separate preflight probe, use the same script with these overrides, then
start the formal run fresh after inspecting `probe.json` and checkpoint loading:

```bash
bash scripts/robodojo/train.sh "${COMMON_ARGS[@]}" \
  output_dir=runs/robodojo_joint2b_probe max_steps=20 \
  eval_at_start=false eval_every=0 save_every=20 \
  preflight.enabled=true preflight.expected_steps=20
```

| Interval in the same run | Video LR | Action/proprio LR | Schedule |
|---|---|---|---|
| 0→80k | 5e-6 → 5e-8 | 1e-4 → 1e-6 | 500-step warmup, cosine to floor 0.01 |
| 80k→100k | 5e-8 → 5e-9 | 1e-6 → 1e-7 | Low-LR cosine to ratio 0.001 |

The YAML's `lr_tail` switches the schedule automatically at step 80,000, retaining
Adam moments and the running RNG/data position. A short probe keeps the same
schedule timescale.
To resume an interrupted run, add a full training-state directory to the same command:

```bash
bash scripts/robodojo/train.sh "${COMMON_ARGS[@]}" \
  resume=runs/robodojo_joint2b_100k/checkpoints/state/step_090000
```

The trainer checks restored step and LR against the schedule. A weight-only `.pt`
file or inference bundle cannot restore Adam.

Held-out validation runs at startup and every 1000 steps: 34 episodes × 16 windows,
10 denoising steps, **shift 3**. Results are in `eval/heldout_action.jsonl`, with
normalized action MSE/L1, denormalized L1 and error counts. These are offline
diagnostics; simulator inference uses **shift 1**.

## 4. Export the paired 100k bundle

```bash
bash scripts/robodojo/export.sh \
  --checkpoint runs/robodojo_joint2b_100k/checkpoints/weights/step_100000.pt \
  --config runs/robodojo_joint2b_100k/config.yaml \
  --stats "$VPP2_PREPARED/dataset_stats.json" \
  --output weights/joint2b_s100000
export VPP2_BUNDLE="$PWD/weights/joint2b_s100000"
```

Keep `action.pt`, `video.pt`, `dataset_stats.json`, `manifest.json` together.
The shared Wan directory supplies the VAE, CLIP, UMT5 and tokenizer needed for
training and inference. Export refuses an existing destination and validates the
full checkpoint step.
It strips training paths and uses a relative paired Video reference.

## 5. Configure and run closed-loop testing

Choose two available GPUs and a free policy port. The following configuration
uses GPU 0 for policy, GPU 1 for Isaac Sim, and one simulator client. It retains
**54 entries / 42 official cells / 2100 episodes** but runs more slowly than the
reference 27-GPU / 19-server / 54-client topology. The distributed template stays
available in `configs/robodojo/eval_reference54.json` with portable host placeholders.

```bash
export VPP2_SIM_ROOT=/path/to/RoboDojo
export VPP2_SIM_PYTHON=/path/to/robodojo-env/bin/python
bash scripts/robodojo/install_adapter.sh --robodojo-root "$VPP2_SIM_ROOT"
bash scripts/robodojo/configure_eval.sh \
  --robodojo-root "$VPP2_SIM_ROOT" --sim-python "$VPP2_SIM_PYTHON" \
  --run-id joint2b100k_seed1_shift1_run1 --sim-gpu 1 --policy-port 23000 \
  --output configs/robodojo/local.eval100k.json

bash scripts/robodojo/server.sh \
  --robodojo-root "$VPP2_SIM_ROOT" --bundle "$VPP2_BUNDLE" --wan "$VPP2_WAN_ROOT" \
  --gpu 0 --port 23000 --eval-config configs/robodojo/local.eval100k.json --dry-run
bash scripts/robodojo/eval.sh --config configs/robodojo/local.eval100k.json --dry-run

# Terminal 1: start policy and wait for model loading/server readiness.
bash scripts/robodojo/server.sh \
  --robodojo-root "$VPP2_SIM_ROOT" --bundle "$VPP2_BUNDLE" --wan "$VPP2_WAN_ROOT" \
  --gpu 0 --port 23000 --eval-config configs/robodojo/local.eval100k.json
# Terminal 2, in the policy environment:
bash scripts/robodojo/eval.sh --config configs/robodojo/local.eval100k.json
```

Adapter installation refuses to overwrite an existing adapter; reuse a verified
matching installation or choose a separate simulator checkout. Include
`--sim-bin-dir /path/to/ffmpeg-bin` when generating the config if the simulator
cannot find ffmpeg. Keep the supplied asset/layout versions fixed, including swap_T.
The policy reads RGB, robot state and instructions through the standard interface.

The server reads the evaluation's exact shift/step/seed/checkpoint label, checks
the expected bundle step and sizes, and rejects conflicting explicit overrides.
For an ablation, generate a new config with a new run ID and explicit `--shift`
or `--steps`. Use `--expected-step` on the server when testing another checkpoint.
The policy loader verifies paired Video/Action steps and tensor shapes when loading.

Evaluation acquires an exclusive lock for its output directory. To resume after
an interruption, first inspect the existing clients and controller, then rerun
with the identical config. New experiments need new output roots and run IDs.
The scheduler does not own the separately launched policy server; stop that server
after evaluation when it is no longer needed.

## 6. Results

`summary.json`, `audit.json`, and `audit.md` are written below the configured
output root. Only an issue-free complete audit with 2100 native episodes is a
finished benchmark. Official SR and Score average each dimension's task cells,
then average the five dimensions equally; pooled SR is successes / 2100.

The released 100k checkpoint has **680/2100** successful episodes, pooled SR
**32.38%**, official SR **33.53%**, and official Score **40.11**.
