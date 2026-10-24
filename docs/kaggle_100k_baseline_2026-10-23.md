# Kaggle 100k-environment-step baseline evidence — 2026-10-23

This record supersedes the earlier 3,000-decision-step pilot as the latest
single-seed baseline evidence. It is still a bounded pilot, not a complete
Dreamer V1 paper reproduction.

## Fixed source and runtime

- Task: `cartpole_swingup`, camera 0, 64 x 64 RGB pixels.
- Upstream: <https://github.com/juliusfrost/dreamer-pytorch>
- Upstream commit: `ccea6ae4a397a94c328891bd78e81d52dd156cb6`.
- `rlpyt` submodule commit: `f04f23db1eb7b5915d88401fca67869968a07a37`.
- Upstream license: MIT; the source license was retained in the Kaggle
  checkout and included in the downloaded archive.
- Kaggle runtime: one visible Tesla T4 (`CUDA_VISIBLE_DEVICES=0`), PyTorch
  `2.11.0+cu128`, headless MuJoCo EGL.
- `dm-control` path: the previously validated no-dependency installation was
  reused. The CPU audit found `dm-control==1.0.34` already visible to
  `/usr/bin/python3`, so no new installation was needed in this run.

## Gates and launch/export fixes

The fresh-process CPU audit recorded `sys.executable`, `sys.path`, pip
version, `pip show -f dm-control`, and `find_spec("dm_control")`. A child
process imported `dm_control`, loaded `cartpole_swingup`, reset the task, and
executed ten random actions. CUDA matrix/gradient and EGL pixel checks then
passed on `cuda:0`.

The ten-update Dreamer gate passed with finite diagnostics and nonzero
parameter changes:

| Module | Maximum absolute parameter change |
|---|---:|
| World model | 0.0116963 |
| Actor | 0.0007101 |
| Critic | 0.0007712 |

The `eval_and_export` path was corrected so `initial_checkpoint` is derived
from the passed path and its existence. The export child completed and wrote
the final summary, evaluation CSV, checkpoint, logs, and archive. The formal
initial checkpoint is a snapshot of the same training model immediately after
prefill and before the first optimizer update; it is not a separately
initialized policy.

## Effective formal configuration

| Field | Value |
|---|---:|
| Training seed | 0 |
| `n_steps` | 50,000 decision steps |
| Action repeat | 2 |
| Underlying environment steps | 100,000 |
| Episode length | 500 decision / 1,000 environment steps |
| Prefill | 1,000 decision steps / 2,000 environment steps |
| First optimizer update | after the prefill boundary |
| `train_every` | 1,000 decision steps |
| `train_steps` | 100 updates per training iteration |
| `batch_size` | 16 |
| `batch_length` | 50 |
| Replay capacity | 60,000 decision transitions / 120,000 environment steps |
| Mixed precision | disabled; precision 32 |
| Anomaly detection | disabled for formal training |
| Nested progress bars | disabled |

The reported iteration is the decision-step index (`iteration == decision
steps` in the saved configuration). With action repeat 2, environment steps are
twice decision steps. `progress.csv` ends at decision step 50,000,
environment step 100,000, iteration 49,999, and cumulative optimizer updates
4,900. The first row is the prefill boundary (`itr=999`, 1,000 decisions,
zero updates); the next row has 100 updates.

The formal training process took 4,438.8 seconds according to `train_subprocess.log`;
the rlpyt progress counter reports 3,772.5 seconds of cumulative training
time. Final fixed-seed evaluation took 636.2 seconds. These are separate
training and evaluation timings; the notebook wrapper also spent time polling
the child process and exporting files.

## Fixed-seed evaluation

Random policy, pre-update formal initial policy, and final Dreamer policy were
evaluated on the same seeds `1000`--`1009`, with the same pixel, action-repeat,
and episode-length contract. The raw rows are in `eval_episodes.csv`.

| Policy | Mean return | Sample standard deviation |
|---|---:|---:|
| Random control | 55.5404 | 42.5973 |
| Formal initial policy | 70.6835 | 3.1504 |
| Final Dreamer policy | 181.7011 | 27.5705 |

These numbers are preliminary evidence from one training seed and a 100,000
environment-step budget. They do not establish paper-level performance,
statistical significance, or a complete reproduction. The random policy is an
auxiliary control, not the mandatory Dreamer baseline.

## Downloaded evidence

The downloaded archive passed `unzip -t` and is stored locally at:

`/Users/zycy/Documents/ChatGPT/IDL/dreamer-pixels/artifacts/kaggle_100k_final/`

It contains the CPU audit/probe, source and compatibility manifests, formal
configuration, raw metrics, debug and training logs, initial and final
checkpoints, `progress.csv`, `eval_episodes.csv`, the export summary, and the
upstream source with its license. The original archive is also retained at
`/Users/zycy/Downloads/dreamer_pilot_artifacts_100k_final.zip`.

Local plots generated from the downloaded raw files are in
`artifacts/kaggle_100k_final/figures/`. They are derived visualizations, not
additional training outputs.

## Limitations and next step

The run did not test full optimizer/replay/environment-state continuation from
the checkpoint. It used one training seed, one task, and a bounded budget. The
next defensible experiment is a separately recorded second seed (and then a
third) using the same effective configuration, followed by a pre-registered
multi-seed comparison. Do not combine this run with the earlier 3,000-step
pilot as if they were one continuous trajectory.
