# Repository scope: VPP2

Keep RoboDojo configs under `configs/robodojo/` and LIBERO configs under
`configs/libero/`; preserve their separate numerical and data contracts.

Keep this repository limited to the documented VPP2 release recipes:
- RoboDojo history-conditioned Video-10k -> joint Video + fresh Action2B, 0-100k.
  Publish the history-conditioned Video-10k checkpoint as the training entry point.
  The robot-video pretrained backbone checkpoint remains a TODO and must not be
  published until explicitly authorized. Keep intermediate experimental stages
  out of the workflow.
  Public documentation presents this training workflow, released checkpoints and
  their evaluation settings without expanding intermediate experiment histories.
  Keep release verification records in internal artifacts, outside public documentation.
  Describe training scale using global batch size and training steps in public docs.
- LIBERO four-suite horizontal Video-10k -> large-batch Action-30k via `scripts/libero/`.
For LIBERO use the same Action-30k checkpoint for standard2000, OOD paper4500,
and repaired PRO public800. Read `docs/libero.md` before changes. Keep the simulator
packages isolated; never substitute OOD300, old PRO400, LIBERO113, or Plus.
No benchmark sharing of action dimensions, normalization, or inference shift.
Use the fixed task inventory, native episode counts, explicit inference shift, and
exact-run audit in `docs/evaluation.md` when reporting evaluations.

Preserve tested numerical operations during refactors. Validate package imports and
relevant CLI dry runs for release-layout changes. Numerical changes also require
the separately maintained internal regression suite. Record full-size GPU or
simulator checks separately from local checks.
Never present the historical reference result as a rerun of changed code.

Use portable configuration paths. Keep checkpoints, data, host addresses and
credentials out of Git. Check GPU processes and tmux sessions before launching jobs;
do not stop unrelated work. Use path, size, schema, step and loading checks rather
than routine whole-file hashes.

The method is named VPP2. Use `vpp2` for the Python package, CLI and environment
variable prefix, and `VPP2` for the XPolicyLab adapter and model class. Legacy names
belong only in checkpoint conversion, historical provenance and retained licenses.
