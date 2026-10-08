# RoboDojo evaluation

For the 100k release, start with the [integrated RoboDojo scripts](robodojo.md#5-configure-and-run-closed-loop-testing).

That guide includes policy-server launch commands and a generated single-machine
evaluation config. The configuration below preserves the distributed resource
template.

## Simulator and policy installation

The reference used RoboDojo revision `08b7ee4` with the SpringButton reset
correction in `patches/robodojo_spring_button_reset.patch`. This correction restores
drive targets in PhysX on repeated resets. The patch is a simulator correction;
the policy consumes only RGB, the provided robot state, and the instruction.
No task implementation or success predicate is imported by the policy.

Install RoboDojo and Isaac Sim using the
[upstream installation guide](https://robodojo-benchmark.com/doc/usage/install-and-download/).
Use that revision and the matching assets/layouts for a reference comparison.
Check the patch before applying it to your separate simulator checkout:

```bash
git -C /path/to/RoboDojo checkout 08b7ee4
git -C /path/to/RoboDojo apply --check "$PWD/patches/robodojo_spring_button_reset.patch"
git -C /path/to/RoboDojo apply "$PWD/patches/robodojo_spring_button_reset.patch"
bash scripts/robodojo/install_adapter.sh --robodojo-root /path/to/RoboDojo
```

The adapter requires the XPolicyLab interface with corrected RGB byte ordering
(at least revision `bb9a0b5`). Install `requirements.txt` in the policy server
environment; `server.sh` loads the policy from this checkout. `install_adapter.sh`
copies only this repository's small adapter and
refuses to overwrite an existing `VPP2` directory.

Seed-1 layouts must be available under `Assets/Eval_Layout/RoboDojo/arx_x5/1`.
The simulator code and assets are external dependencies. The supplied reset patch
was checked against the local pinned upstream tree. Changing simulator revisions,
`swap_T` assets, camera timing, or native layouts creates a different evaluation.

## Start policy servers

Inspect `nvidia-smi` and `tmux ls` on each host before allocating GPUs. Start each
server with a unique free port in the **policy environment**:

```bash
bash scripts/robodojo/server.sh --robodojo-root /path/to/RoboDojo \
  --bundle /models/joint2b_s100000 --wan /models/Wan2.1-I2V-14B-480P \
  --gpu 0 --port 19960 --host 127.0.0.1
```

The server defaults to seed 1, 10 steps, shift 1, horizon 32, replan 24. Its Video
and Action files must have the same step. No initializer Action backbone is
needed during deployment. VAE/CLIP/T5/tokenizer still come from the Wan directory.
Alternate `--steps` / `--shift` values are ablations and must have distinct run IDs.

Each simulator client carries a client ID; its history, episode anchor, and noise
counter are isolated. History receives one observation per executed action,
including intermediate actions within the chunk. Grippers are continuous EE16
channels; no thresholding is applied by this policy.

## Packed evaluation

`configs/robodojo/eval_reference54.json` preserves the reference layout chunking (12),
retry count (5), native task inventory, seed, video settings and 54 client slots.
Hostnames and filesystem paths are placeholders. Configure them for the target
installation, including `sim_python` from the **simulator environment**.
Video saving requires an executable named `ffmpeg`. Install it in that environment
or list its directory in `sim_bin_dirs` (an array of paths on the simulator hosts).
The client launcher checks `ffmpeg -version` before starting Isaac Sim when video
saving is enabled, and reports a nonzero status if the check fails.
Policy ports must be reachable as localhost ports on each simulator host (use
port forwarding when servers are remote). All scheduler, simulator and result
paths must be visible through the same shared filesystem.

The measured reference resource plan had 27 GPUs, 19 policy servers and 54 clients.
Reducing slots is possible for external testing, but is a separately declared
resource configuration, not a new measurement under the original formal topology.

```bash
cp configs/robodojo/eval_reference54.json configs/robodojo/local.eval.json
# Edit absolute paths, SSH aliases/ports and the available GPU/server slots.
bash scripts/robodojo/eval.sh --config configs/robodojo/local.eval.json --dry-run
bash scripts/robodojo/eval.sh --config configs/robodojo/local.eval.json
```

Use a new run ID, checkpoint label, session prefix and output directory for a new
checkpoint. Repeating an identical configuration resumes its saved plan. The
scheduler retries failed chunks and tops up unstable layouts; it selects the
lowest native stable layout IDs, retaining the reference merge rule.
It does not terminate pre-existing tmux sessions.

## Metrics

There are **54 runnable entries**, paired into **42 official cells**:

- 12 Generalization cells have a clean/random pair, **25 episodes per entry**.
- Remaining cells have **50 episodes per entry**.
- Total: **2100 episodes**, seed 1, `arx_x5`, absolute EE16.

Within each dimension, average its cells. Then average the five dimension scores
with equal weight. Apply the same calculation to success rate. The pooled episode
rate weights dimensions by their episode counts, so it is a different metric.
The exact-run audit reports both and refuses to mark incomplete inventories complete.
It reads only the requested run, not the latest result from every task directory.

`bash scripts/robodojo/eval.sh` runs that audit after scheduling completes. Results are written to
the selected output directory as `summary.json`, `audit.json`, and `audit.md`.
Rollout videos remain in the RoboDojo result directory. The reference saves up to
five layouts per shard.

Existing pre-release bundles can be loaded directly; the policy converts their
serialized target names to `vpp2` before model construction. Re-exporting produces
a bundle with VPP2 format identifiers. Install the adapter from `policy/VPP2`.
