# VPP2 on LIBERO: installation, training and evaluation

This release uses the **four-suite horizontal, large-batch** model from the paper.
All three benchmarks use **Video step 10,000 + Action step 30,000**.
The LIBERO Action expert is approximately **1B** (hidden 512, FFN 2048), while the
RoboDojo recipe uses Action2B. The two recipes have separate entrypoints and configs.

## 1. Reference results

| Benchmark | Protocol | Successful episodes | Success rate |
|---|---|---:|---:|
| Original LIBERO | 40 tasks × 50 packaged init states | 1,976 / 2,000 | 98.80% |
| Modified LIBERO-OOD | 30 tasks × 3 deterministic seed IDs × 50 random resets | 2,877 / 4,500 | 63.93% |
| LIBERO-PRO | 40 tasks × Position/Task × init indices 1–10 | 360 / 800 | 45.00% |

The OOD protocol is the 4,500-episode paper budget. PRO uses the repaired
HarnessVLA public-runtime **direct-policy** 800-episode protocol: no planner,
perception tools, task memory or motion primitives. The upstream paper does not
fully specify every reset/sampler detail; the pinned public runtime below defines
those implementation details. LIBERO113, Plus and the old 300/400-episode
protocols are outside this release.

## 2. Install dependencies

Run commands from the repository root. Python 3.10 is the tested interpreter.
Create a dedicated environment and follow the main [installation guide](robodojo.md#policy-environment)
for the tested CUDA/PyTorch pair. Then install:

```bash
python -m pip install -r requirements-train.txt -c environment-reference.txt
bash scripts/libero/prepare.sh --help
```

Each script loads `src/vpp2` from the checkout; installing the project package is
unnecessary. Use `PYTHON_BIN=/path/to/python` to select an environment. Paths in
arguments resolve from the repository root.

| Script | Operation |
|---|---|
| `scripts/libero/prepare.sh` | Validate the prepared LeRobot dataset |
| `scripts/libero/text_cache.sh` | Cache language embeddings |
| `scripts/libero/init_action.sh` | Initialize the Action expert |
| `scripts/libero/train_video.sh` | Video-10k training |
| `scripts/libero/train_action.sh` | Action-30k training |
| `scripts/libero/prepare_pro.sh` | Prepare the isolated PRO runtime |
| `scripts/libero/eval.sh` | Evaluate standard, OOD or PRO using its config |
| `scripts/libero/audit.sh` | Audit a completed evaluation |

Configure the launcher environment for your runtime. Adjust `batch_size` and
`gradient_accumulation_steps` to retain the global batch size listed for each
stage. Use `--dry-run` to inspect the effective global batch before training.

Training was run with PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130,
Accelerate 1.13.0 and DeepSpeed 0.18.5. The local LeRobot reader uses
`datasets==3.6.0`, `jsonlines==4.0.0`, PyAV, and the preserved LeRobot v2.1
schema. Installing the latest external LeRobot package is unnecessary.

Simulator environments must be isolated from each other. They may have the same
VPP2 checkout, but must not put multiple `libero` packages on
one `PYTHONPATH`. Keep PRO's robosuite 1.5.2 separate from standard/OOD's 1.4.0.
Do not change dependencies in an environment serving a running training job.

## 3. Data and required model files

Standard LIBERO, LIBERO-OOD and LIBERO-PRO evaluation all require the same
**four-suite horizontal Video-10k + Action-30k** pair:

The complete bundle is available on public [Hugging Face](https://huggingface.co/Haodong082399/VPP2)
and [ModelScope](https://modelscope.cn/models/haodong123/VPP2).

| Required artifact | Local path | Release availability |
|---|---|---|
| Video-10k | `weights/libero/video_step010000.pt` | [ModelScope download](https://www.modelscope.cn/api/v1/models/haodong123/VPP2/repo?Revision=master&FilePath=checkpoints/libero/video_step010000.pt) |
| Action-30k | `weights/libero/action_step030000.pt` | [ModelScope download](https://www.modelscope.cn/api/v1/models/haodong123/VPP2/repo?Revision=master&FilePath=checkpoints/libero/action_step030000.pt) |
| Shared VAE, CLIP, UMT5 and tokenizer | `weights/Wan2.1-I2V-14B-480P/` | [ModelScope](https://modelscope.cn/models/haodong123/VPP2), under `checkpoints/Wan2.1-I2V-14B-480P/` |
| Normalization statistics | `data/libero/dataset_stats.json` | Included in [configs/libero/dataset_stats.json](../configs/libero/dataset_stats.json) |

Both checkpoints are released in bf16 with the original tensor values preserved.
The Action checkpoint includes the proprio encoder, portable VPP2 configuration
and a relative reference to the paired Video; optimizer state is excluded.
After downloading, follow the simulator setup in section 6 and the
[evaluation commands](#7-evaluate-the-same-action-30k-checkpoint).

Download Video-10k, Action-30k and the shared encoders with Hugging Face
(`python -m pip install -U huggingface_hub` in a download environment):

```bash
hf download Haodong082399/VPP2 --local-dir weights \
  --include 'config.json' \
  --include 'checkpoints/libero/**' \
  --include 'checkpoints/Wan2.1-I2V-14B-480P/**'
```

Or use ModelScope:

```bash
modelscope download --model haodong123/VPP2 \
  --local_dir weights \
  --include 'checkpoints/libero/**' 'checkpoints/Wan2.1-I2V-14B-480P/**'
```

Then link the downloaded files to the configured paths:

```bash
mkdir -p weights/libero
ln -s ../checkpoints/libero/video_step010000.pt weights/libero/video_step010000.pt
ln -s ../checkpoints/libero/action_step030000.pt weights/libero/action_step030000.pt
ln -s checkpoints/Wan2.1-I2V-14B-480P weights/Wan2.1-I2V-14B-480P
mkdir -p data/libero
cp configs/libero/dataset_stats.json data/libero/dataset_stats.json
```

For training, use Video-10k to start
[Stage 2 Action training](#5-stage-2-large-batch-action-training) directly.
The Stage-1 robot-video initializer and training data remain pending release.
Expected local paths (or override them in copied configs):

```text
weights/
  Wan2.1-I2V-14B-480P/          # shared VAE, CLIP, UMT5 and tokenizer
  libero/
    robot_video_step028000.pth # exact prior robot-video initialization for Stage 1
    video_step010000.pt        # Stage 1 standalone checkpoint, format wan21_video_v1
    action_init.pt            # produced by init-action below
    action_step030000.pt       # compact paper Action checkpoint

data/libero/
  dataset_stats.json
  libero_spatial_no_noops_lerobot/
  libero_object_no_noops_lerobot/
  libero_goal_no_noops_lerobot/
  libero_10_no_noops_lerobot/
  text_cache/
```

Each LeRobot directory contains `meta/info.json`, `meta/tasks.jsonl`,
`meta/episodes.jsonl`, `data/chunk-*/episode_*.parquet`, and camera videos under
`videos/chunk-*/<camera>/episode_*.mp4`. Use the reference conversion rendered
with MuJoCo 3.3.2 and no-op filtering; a generic LIBERO dataset conversion may
produce different actions, camera pixels or episode inventories.

| Suite | Episodes | Frames |
|---|---:|---:|
| Spatial | 434 | 53,229 |
| Object | 457 | 67,309 |
| Goal | 433 | 52,895 |
| LIBERO-10 | 388 | 104,280 |
| Total | 1,712 | 277,713 |

Training samples the natural frame-count mixture. All demonstrations are used
for training; fixed training examples provide diagnostics, not a held-out SR.
Normalization is the supplied **min/max** statistics, including the gripper;
use the exact [reference statistics](../configs/libero/dataset_stats.json).
The first six action coordinates are marked as deltas; the gripper is not.

```bash
mkdir -p data/libero
cp configs/libero/dataset_stats.json data/libero/dataset_stats.json
bash scripts/libero/prepare.sh --config configs/libero/train_action.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/libero/text_cache.sh \
  --config configs/libero/train_action.yaml --device cuda
```

The cache uses the original complete robot-view prompt template and UMT5 context
length 128. Prompt-key addressing is preserved so existing caches remain usable.
Training configs retain the historical model tokenizer default 512, but training
consumes cached 128-token contexts. **Evaluation explicitly sets tokenizer 128.**

### Input contract

- 65 observation timestamps; video selects `t0,t4,...,t64`: 17 RGB frames.
- Head and wrist each resize directly to 224×224 with tensor bilinear antialiasing;
  horizontal concatenation gives `[3,17,224,448]` in `[-1,1]`.
- Action target `[32,7]`; proprio at `t0` is `[1,8]`.
- Episode tails repeat the final value, with separate image/action padding masks.
- VAE produces five video latent frames. Video self-attention is bidirectional
  across all five; action queries directly see the first three. Keep all five.
- Stage 2 encodes only the first RGB frame through VAE/CLIP, then builds all five
  video latent frames as pure noise at timestep 1000 and caches frozen video K/V.

## 4. Stage 1: video-only training

Config: [train_video.yaml](../configs/libero/train_video.yaml).
Only the Wan2.1 video model is instantiated; there is no Action expert in this stage.
Initialization must be the supplied prior robot-video step-28k backbone. Raw Wan
weights are not an equivalent initialization for reproducing the paper.

| Setting | Value |
|---|---|
| Global batch size | 64 |
| Training steps | 10,000 |
| Learning rate / warmup | 5e-6 / 200 |
| Optimizer | AdamW, betas 0.9/0.999, weight decay 0.01 |
| LR schedule / floor | cosine / 0.01 of initial LR |
| Video timestep sampling | Beta(7,1) |
| Precision / clipping | bf16 / 1.0 |
| Checkpoint / video diagnostic | every 2,000 / every 1,000 steps |

First inspect live GPU processes and tmux sessions on the intended machine.
Run a bounded probe in a **new output directory** before a formal job:

```bash
nvidia-smi
# Inspect existing sessions; an absent tmux server is fine.
tmux list-sessions
bash scripts/libero/train_video.sh \
  output_dir=runs/libero/probe_video max_steps=20 warmup_steps=0 \
  preflight.enabled=true preflight.expected_steps=20 eval_every=0
```

Review sample shapes, timestamp/padding alignment, the trainable parameter list,
finite loss/gradients, the 20-step save/reload, and peak GPU memory in `probe.json`.
Test full optimizer/dataloader resume using the saved state. Generate and inspect
fixed-sample one-step/ten-step video comparisons (`eval_at_start=true`, with the
normal `eval_every`), and obtain approval before a formal training launch on a
shared machine.

```bash
bash scripts/libero/train_video.sh output_dir=runs/libero/video
```

The checkpoint is `runs/libero/video/checkpoints/weights/step_010000.pt`.
Copy or symlink it to `weights/libero/video_step010000.pt` for the next stage.

## 5. Stage 2: large-batch Action training

Config: [train_action.yaml](../configs/libero/train_action.yaml).
Freeze the entire Video expert, VAE, CLIP and text encoder; train only the Action
expert and proprio projection. A fresh optimizer and schedule are used.

```bash
bash scripts/libero/init_action.sh \
  --video weights/libero/video_step010000.pt \
  --config configs/libero/train_action.yaml \
  --output weights/libero/action_init.pt
```

Conversion uses sequential linear tensor interpolation with alpha scaling. Action
input/output projections and the proprio projection are newly initialized.
Existing output files are refused rather than overwritten.

| Setting | Value |
|---|---|
| Global batch size | 256 |
| Training steps | 30,000 |
| Action expert | hidden 512, FFN 2048, 40 layers, 40 heads × 128 |
| Learning rate / warmup | 1e-4 / 500 |
| Optimizer | AdamW, betas 0.9/0.95, weight decay 0.01 |
| LR schedule / floor | cosine / 0.01 of initial LR |
| Action noise train/infer shift | 5 / 5 |
| Precision / clipping / seed | bf16 / 1.0 / 42 |
| Checkpoint / action diagnostic | every 5,000 / every 200 steps |

```bash
bash scripts/libero/train_action.sh \
  output_dir=runs/libero/probe_action max_steps=20 warmup_steps=0 \
  preflight.enabled=true preflight.expected_steps=20 eval_every=0
```

Review the Stage-1 video diagnostic, converted tensor shapes and trainable set.
Confirm frozen video parameters are excluded from the optimizer and unchanged
in the probe; compare tensors directly when checking this, without whole-file
hashes. Review finite loss, memory and checkpoint reload before approving formal
training. The release uses ZeRO-2; ZeRO-3 compact export is unsupported.

```bash
bash scripts/libero/train_action.sh output_dir=runs/libero/action
```

Weights: `runs/libero/action/checkpoints/weights/step_030000.pt`.
The compact file needs its Video-10k checkpoint, supplied explicitly in eval
configs; moving an old checkpoint does not require editing historical paths
inside it. Compatibility loading never imports the old project package.

Resume **full training state**, including optimizer, LR schedule and sampler:

```bash
bash scripts/libero/train_action.sh output_dir=runs/libero/action \
  resume=runs/libero/action/checkpoints/state/step_025000
```

`resume_ckpt=...` only initializes parameters; it is not a full training resume.
The CLI checks the global batch size against 64 for Video and 256 for Action.
`--allow-batch-change` is available for diagnostics with a different global batch.

## 6. Prepare three isolated simulator runtimes

The simulator packages and assets are installed separately; they are not bundled.
Use separate Python environments for standard, OOD and PRO. In each environment,
install `requirements.txt` and the same tested PyTorch pair. The evaluation scripts
load the policy from this checkout, including in worker subprocesses.

### Standard LIBERO

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
python -m pip install -e third_party/LIBERO
python -m pip install -r configs/libero/requirements_standard.txt
```

Install the complete upstream assets and packaged `init_files` following that
pinned repository's instructions. Four-suite evaluation requires 50 init states
per task. VPP2 writes an isolated `LIBERO_CONFIG_PATH`, so an unrelated user's
`~/.libero` does not determine the benchmark paths.

### Modified LIBERO-OOD

In its separate environment:

```bash
git clone https://github.com/QuanyiLi/pi0-text-latent.git third_party/pi0-text-latent
git -C third_party/pi0-text-latent checkout 587a6cbf64f16c7b87fa5805dc0ed934192239a4
python -m pip install -e third_party/pi0-text-latent/third_party/modified_libero
python -m pip install -r configs/libero/requirements_standard.txt
```

Use the modified package's assets and BDDLs. Do not install original LIBERO into
this environment. The worker verifies that the imported benchmark lives under
the configured checkout. Deterministic seed derivation is preserved exactly;
it derives seeds from suite/task/episode/seed-ID strings, not file hashes.

### LIBERO-PRO public runtime

In its separate environment, install the versioned simulator dependency overlay:

```bash
python -m pip install -r configs/libero/requirements_pro.txt
python -m pip install --no-deps --target third_party/libero_pro_runtime \
  'rpent-liberopro==0.2.0' 'rpent-libero==0.2.0' 'robosuite==1.5.2'
```

The requirements file installs the simulator dependencies while retaining the
pinned MuJoCo version. Obtain a complete upstream LIBERO-PRO asset tree. Then:

```bash
bash scripts/libero/prepare_pro.sh \
  --runtime third_party/libero_pro_runtime \
  --assets /absolute/path/to/complete/LIBERO-PRO/libero/libero/assets
```

This downloads authoritative init-state/BDDL metadata from
`zhouxueyang/LIBERO-Pro`, revision
`c86fc3b8293185a6f373677018ff3e37f8391602`. It repairs the three malformed Task
BDDL files and incomplete init archives in the 0.2.0 wheel. It also supplies the
asset tree that the wheel omits. Runtime code maps the old Panda `None` base to
robosuite's `NullBase`, rejecting unexpected alternatives.

Before policy inference, each worker constructs/resets all its assigned PRO
cells at seed/index 1. Across the worker grid this checks all 80 cells. It checks
50 packaged states per task and actual BDDL Task language, including Spatial
Task 0's `not between`. Worker failure prevents a complete result.

## 7. Evaluate the same Action-30k checkpoint

| Setting | Standard | OOD | PRO |
|---|---|---|---|
| Policy seed | 42 | 7 | 42 |
| Reset | packaged states 0–49 | random, deterministic task/episode seed | packaged states 1–10 |
| Seed IDs / environment seeds | sequential task RNG | seed IDs 0,1,2 | environment seeds 1–10 |
| Settle no-ops | 30 | 10 | 15 |
| Spatial/Object/Goal/Long horizons | 400/400/400/700 | 330/420/450/— | 220/280/300/520 |
| Tasks / episode budget | 40 / 2,000 | 90 task-seed cells / 4,500 | 80 / 800 |

All use: 256-square rendered cameras rotated 180 degrees; `training_exact`
direct tensor resize to two 224-square views; tokenizer 128; **action horizon 32,
10 Euler denoising steps, shift 5, execute 10 actions**, binary gripper, and
independent Video Gaussian seed `policy_seed + 1`. One policy remains resident
per worker; task-start RNG is restored. PRO constructs a fresh environment for
every episode as required by the paired public protocol.

Inspect configs before assigning GPUs:

```bash
bash scripts/libero/eval.sh --config configs/libero/eval_standard.yaml --gpus 0,1,2,3 --dry-run
bash scripts/libero/eval.sh --config configs/libero/eval_ood.yaml --gpus 0,1,2,3 --dry-run
bash scripts/libero/eval.sh --config configs/libero/eval_pro.yaml --gpus 0,1,2,3 --dry-run
```

Then run each from its matching simulator environment, sequentially if using
the same GPUs:

```bash
bash scripts/libero/eval.sh --config configs/libero/eval_standard.yaml --gpus 0,1,2,3
bash scripts/libero/eval.sh --config configs/libero/eval_ood.yaml --gpus 0,1,2,3
bash scripts/libero/eval.sh --config configs/libero/eval_pro.yaml --gpus 0,1,2,3
```

All paths can be overridden with `key=value`; use absolute paths when launching
from another working directory. GPUs with more than 2 GiB allocated are refused.
The launcher displays current processes/sessions and never terminates other jobs.

Each run creates:

```text
RUN_CONFIG.json               # immutable checkpoint/path/size/mtime/config/GPU contract
manifests/                    # exact task lists and worker configs
logs/                         # worker stdout/stderr
status/                       # per-worker exit codes
results/seed*/<suite>/         # task result JSONs and rollout videos
summary.json                  # strict full-grid audit
full_run.status               # 0 only after issue-free completion
```

Resume with the same config, checkpoint artifacts, GPU list and output directory:

```bash
bash scripts/libero/eval.sh --config configs/libero/eval_ood.yaml --gpus 0,1,2,3 --resume
bash scripts/libero/audit.sh --config configs/libero/eval_ood.yaml
```

The coordinator locks its output directory. Resume refuses changed checkpoint,
statistics, training config or GPU assignments. The audit rejects missing,
duplicate or unexpected cells, non-partitioned trial IDs, wrong init/seed lists,
wrong horizons and nonzero worker status. Intermediate SR is order-dependent;
only `complete=true`, an empty issue/missing list, and zero `full_run.status`
constitute the final protocol result.
