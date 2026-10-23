# Kaggle PyTorch Gate Evidence (2026-10-23)

This note records the bounded route that replaced the blocked TensorFlow
session. It is evidence for the pilot only; it is not a paper-reproduction
claim.

## Fixed source

- Repository: `https://github.com/juliusfrost/dreamer-pytorch`
- Dreamer commit: `ccea6ae4a397a94c328891bd78e81d52dd156cb6`
- `rlpyt` submodule: `f04f23db1eb7b5915d88401fca67869968a07a37`
- License: MIT, retained in the upstream checkout.

## Runtime evidence

- Interpreter: `/usr/bin/python3`
- Python: 3.13.15
- PyTorch: `2.11.0+cu128`
- Visible devices: one Tesla T4 after `CUDA_VISIBLE_DEVICES=0`
- dm-control: `1.0.34`
- dm-env: `1.6`
- MuJoCo: `3.3.7`
- glfw: `2.11.0`
- Environment: `cartpole_swingup`, 64x64 RGB, camera 0, action repeat 2.

The control-path check imported `dm_control`, loaded the task, reset it, and
executed ten random actions. The GPU check ran a finite matrix gradient on
`cuda:0` and produced an EGL frame with shape `(64, 64, 3)` and dtype `uint8`.

## Ten-update gate

The bounded gate completed 10 real Dreamer updates. The recorded summary
reported finite update metrics and changed parameters for all three checked
components:

- world model: maximum absolute parameter change `0.011696301400661469`;
- actor: maximum absolute parameter change `0.0007100999355316162`;
- critic: maximum absolute parameter change `0.0007711630314588547`.

The checkpoint was saved and loaded by a fresh Python process. The same
process also exported `eval_episodes.csv` for random and Dreamer policies on
seeds 1000--1009. The debug evaluation is a gate artifact, not a learning
curve and not ten training seeds.

## Compatibility record

The previously successful Kaggle installation used `dm-control --no-deps`.
The current notebook reused that exact version and added only the separately
observed missing `glfw` runtime module. The fixed `rlpyt` checkout also
required `pyprind` at import time; the missing-module traceback was preserved
and the declared `pyprind` package was installed without dependency changes.
The runner removed a duplicate image-shape keyword after the fixed upstream
constructor raised a direct `TypeError`; the Dreamer algorithm was not
rewritten.

## Artifact directory

Kaggle raw outputs are under `/kaggle/working/dreamer_pytorch_pilot/`, including
`debug_summary.json`, `metrics.jsonl`, `eval_episodes.csv`, the checkpoint,
installation logs, source manifest, and compatibility patch record. The
bounded baseline artifacts must be checked separately before being reported.
