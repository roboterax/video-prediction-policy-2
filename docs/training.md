# Training the released RoboDojo stage

Start with the [complete RoboDojo guide](robodojo.md) for installation,
consistent path variables and complete commands. This document expands the training contract.

## Required artifacts

| Artifact | Purpose |
|---|---|
| Wan2.1-I2V-14B-480P directory | VAE, CLIP, T5, tokenizer, model configuration |
| `robodojo_his10k.pt`, history-conditioned Video-10k | starting Video DiT and source for Action2B initialization |
| converted full-episode metadata + videos + EE16 parquet files | 3500 source demonstrations |
| `configs/robodojo/normalization_ee16.json` | exact statistics from all 3500 episodes |

The release provides RoboDojo history-conditioned Video-10k.
Start joint + Action2B directly from Video-10k;
[download it and follow the integrated scripts](robodojo.md) for one continuous
0–100k run. The robot-video pretrained backbone checkpoint is on the
[TODO list](../README.md#todo). The converter below builds the training data from the public RoboDojo export.

The converter accepts the official, shift-fixed **EE16 LeRobot v3.0** export.
Joint14, LeRobot v2.1 and raw XPolicyLab HDF5 are not inputs to this recipe.
Parquet columns: `action`, `observation.state` (both `[T,16]`), `frame_index`,
`raw_frame_index` (both identity native frame indices). Videos are RGB 480×880
T-shaped composites at 25 Hz. Metadata has `episode_index`, `task_name`, `dimension`,
`video_path`, `parquet_path`, `fps`, `trim_start`, `trim_end`, and `training_prompt`.
`training_prompt` already contains the complete video instruction template.
The EE16 coordinate frame and quaternion order must match the supplied data.

## Convert the public dataset

Install `ffmpeg` with `libx264` support and download the
[official EE16 dataset](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/main/data/RoboDojo_ee_lerobot_v30_video):

```bash
hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
  --include 'data/RoboDojo_ee_lerobot_v30_video/**' --local-dir /data/robodojo_download
bash scripts/robodojo/convert.sh \
  --source /data/robodojo_download/data/RoboDojo_ee_lerobot_v30_video \
  --output /data/robodojo --workers 4
```

This CPU converter uses each episode's Parquet row bounds and each camera's own
file and timestamps. It preserves the absolute EE16 state/action values,
quaternion order and native frame indices without another action shift. The
source must contain 3500 episodes / 1,856,102 frames at 25 Hz. Task names and
prompts come from [`source_tasks.csv`](../configs/robodojo/source_tasks.csv).

The output is one Parquet and one RGB MP4 per episode, plus
`full_episode_metadata.csv`. Composition follows the training export: a 640×480
head view on the left, two 240×240 wrist views stacked on the right. Wrist views
are center-cropped to 4:3 before resizing. Each camera is trimmed independently;
if its final frame is absent, that episode's final valid frame is repeated once.
Training subsequently resizes the 880×480 composite directly to 416×240.

Use a fresh output directory. `--limit 2` runs a small conversion smoke test;
such a subset cannot pass the full training split check. Metadata is written
only after all selected episodes succeed. No normalization statistics are refit.

## Prepare

```bash
bash scripts/robodojo/prepare.sh \
  --metadata /data/robodojo/full_episode_metadata.csv \
  --media-root /data/robodojo \
  --stats configs/robodojo/normalization_ee16.json \
  --output data/robodojo

bash scripts/robodojo/text_cache.sh --data data/robodojo \
  --wan /models/Wan2.1-I2V-14B-480P --device cuda

bash scripts/robodojo/init_action.sh --video /models/robodojo_his10k.pt \
  --output weights/init/action2b_from_his10k.pt
```

`prepare` preserves source row order and the exact held-out episode IDs in
`configs/robodojo/holdout.json`. It validates all training parquet files, detects gripper
transitions, and writes a critical sampling index with portable row identities.
It does not re-encode video or refit normalization statistics.
Set `paths.media` to the media root; metadata and text caches can live separately
under `paths.data`. Relative asset names remain valid when the media root moves.

The sampler retains task/episode weighting and the mass of padded starts. It mixes
uniform full windows with critical-frame scores (mixture 0.5, density limit 3).
The index validates its temporal contract, required arrays, row identities and
lengths. It does not use whole-file checksums.

## Reference training

| Setting | Value |
|---|---:|
| Global batch size | 288 |
| Training steps | 100,000 |

Configure the launcher environment for your runtime as described in the
[RoboDojo guide](robodojo.md). Adjust `batch_size` and
`gradient_accumulation_steps` to retain global batch 288. All training processes
use the same data, weights and output directory.

Launch with the configured environment:

```bash
bash scripts/robodojo/train.sh \
  output_dir=runs/robodojo_joint2b_100k \
  paths.wan=/models/Wan2.1-I2V-14B-480P \
  paths.video_init=/models/robodojo_his10k.pt \
  "paths.action_init=$PWD/weights/init/action2b_from_his10k.pt" \
  "paths.data=$PWD/data/robodojo" \
  paths.media=/data/robodojo
```

Training uses bf16 and DeepSpeed ZeRO-1. Use a fresh output directory.
Explicit resume uses
`resume=/path/to/run/checkpoints/state/step_050000`; weight-only initialization uses
`resume_ckpt=/path/to/full_weights.pt`. These are different operations.

Before a formal run, use a separate output directory for a 20-step probe:

```text
max_steps=20 eval_at_start=false eval_every=0 save_every=20
preflight.enabled=true preflight.expected_steps=20
```

Inspect `probe.json` for finite losses/gradients, correct shapes and global batch,
memory headroom, and checkpoint loading. Start the formal run fresh from the
initializers, not from the probe's weights. Keep global batch 288 for the release
recipe. `--allow-batch-change` is available for diagnostics with a different
global batch.

## Optimization and validation

Video LR is 5e-6; Action/proprio LR is 1e-4. Both use 500-step warmup and the same
80k-step cosine multiplier with floor 0.01, followed automatically by a low-LR
cosine tail to ratio 0.001 at 100k. The complete configuration is
`configs/robodojo/train_100k.yaml`. AdamW uses betas (0.9, 0.95), weight
decay 0.01, and gradient clipping 1.0. Full joint weights are saved every 5000 steps.
The Action backbone is interpolated with alpha scaling; Action input/output and
proprio projections are initialized fresh. Training RNG/data seed is 100008.
The seed is applied before model construction so fresh action/proprio projections
are repeatable with the same software and configuration.

Validation runs at startup and every 1000 steps on 34 episodes × 16 fixed windows.
It reports normalized action MSE/L1 and denormalized L1, excluding padded tokens.
Its default action inference shift is **3**, inherited from the training model
configuration. Closed-loop benchmark inference explicitly overrides it to **1**.
These held-out episodes may have been seen during the earlier Video initialization
stage; this validation measures stage-specific behavior, not clean unseen-task generalization.

## Continuous schedule and interruption recovery

`scripts/robodojo/train.sh` runs the complete 0–100k schedule in one process per
rank. At 80k, the low-LR tail begins automatically: Video 5e-8→5e-9 and
Action/proprio 1e-6→1e-7. There is no extra warmup or optimizer reset.

After an interruption, use the same YAML and add
`resume=runs/robodojo_joint2b_100k/checkpoints/state/step_090000` (or another saved
step). Full state restores model, Adam moments, RNG and sampler progress; the
trainer validates the restored scheduler step and LR. Weight-only `.pt` files
cannot replace full state for continuation. Checkpoint-state restoration is tested;
bitwise identical training trajectories across interruptions are not established.

## Export the selected step

```bash
bash scripts/robodojo/export.sh \
  --checkpoint runs/robodojo_joint2b_100k/checkpoints/weights/step_100000.pt \
  --config runs/robodojo_joint2b_100k/config.yaml \
  --stats data/robodojo/dataset_stats.json \
  --step 100000 --output weights/joint2b_s100000
```

The bundle contains `action.pt`, `video.pt`, `dataset_stats.json`, and
`manifest.json`. It removes training paths from the inference configuration and
references `video.pt` relative to `action.pt`. Always move these files together.
