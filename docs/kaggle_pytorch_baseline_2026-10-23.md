# Kaggle PyTorch baseline evidence — 2026-10-23

This record describes the completed, bounded `cartpole_swingup` pilot. It is
not a full Dreamer V1 paper reproduction.

## Fixed implementation and environment

- Upstream: `https://github.com/juliusfrost/dreamer-pytorch`
- Dreamer commit: `ccea6ae4a397a94c328891bd78e81d52dd156cb6`
- `rlpyt` submodule: `f04f23db1eb7b5915d88401fca67869968a07a37`
- Upstream license: MIT; the Kaggle checkout retains its license and source
  attribution.
- Kaggle runtime: Python 3.13.15, PyTorch `2.11.0+cu128`, one visible Tesla
  T4 (`cuda:0`), `CUDA_VISIBLE_DEVICES=0`.
- Environment route: `dm-env==1.6`, `mujoco==3.3.7`, and
  `dm-control==1.0.34` installed with `--no-deps`, plus `glfw==2.11.0` and
  `pyprind==2.11.3`. Headless rendering used `MUJOCO_GL=egl`.

## Effective configuration and step accounting

The formal run used:

| Field | Effective value | Interpretation |
|---|---:|---|
| task | `cartpole_swingup` | pixel control |
| image | 64x64 RGB, camera 0 | observation stored as uint8, CHW to agent |
| action repeat | 2 | one decision advances two environment steps |
| episode limit | 500 decision / 1,000 environment steps | same contract for train and eval |
| prefill | 1,000 decision steps | first update begins after this boundary |
| replay capacity | 1,060 decision transitions | intentionally small bounded pilot buffer |
| `train_every` | 1,000 decision steps | upstream schedule retained |
| `train_steps` | 100 per training iteration | upstream schedule retained |
| batch | 16 sequences x 50 steps | `batch_size=16` is the recorded memory-conscious adjustment |
| training seed | 0 | one training seed |
| requested horizon | 3,000 decision steps | 6,000 underlying environment steps |
| anomaly detection | off | formal-run setting |
| nested progress bars | off | stdout remained bounded |

The runner records `iteration == decision steps`, so iteration 999 is the
prefill boundary and has zero optimizer updates. The exported
`progress.csv` shows cumulative updates 0 at 1,000 decisions, 100 at 2,000,
and 200 at 3,000. Thus the run has 200 real updates, not 3,000 updates. An
environment-step axis is exactly twice the decision-step axis.

The debug gate intentionally used a separate small configuration
(`batch_size=2`, `batch_length=10`, `prefill=60`, one update per decision)
to test the update path. Its ten updates must not be concatenated with the
formal metrics as if they were one continuous run.

## Gates and training evidence

The CPU `dm-control` import, `suite.load("cartpole", "swingup")`, reset, and
ten random actions passed. CUDA matrix/gradient computation and a 64x64 EGL
pixel render passed on `cuda:0`/Tesla T4. The ten-update Dreamer gate passed
with finite losses and nonzero finite gradients for world model, actor, and
critic; a fresh process loaded the debug checkpoint and evaluated it.

The formal run then completed the 3,000-decision bound. The final training
summary reports finite metrics and these maximum absolute parameter changes:

| Module | Parameter changed | Max absolute change |
|---|---|---:|
| World model | yes | 0.11192846298217773 |
| Actor | yes | 0.019696608185768127 |
| Critic | yes | 0.025965871289372444 |

The training checkpoint was saved at
`train_run/run_0/params.pkl`. The rlpyt log records snapshots at iterations
999, 1999, and 2999; the first is prefill-only, while the latter two follow
100 and 200 cumulative updates respectively. This is a model/checkpoint
load-and-evaluation validation, not a claim of full optimizer/replay
continuation across processes.

## Fixed-seed evaluation

The same environment seeds 1000--1009 were used for random, initial-policy,
and final Dreamer evaluations. Each row is one episode; these ten episodes
are not ten training seeds.

| Policy | Mean episodic return | Sample standard deviation |
|---|---:|---:|
| Random | 50.16178831307825 | 45.60471228907806 |
| Initial policy | 97.63728615798557 | 41.66708649394165 |
| Final Dreamer pilot | 115.46288450648845 | 29.112176152836554 |

These are preliminary single-seed results under a short 6,000-environment-
step budget. They do not establish paper-level performance or a statistically
reliable improvement claim.

## Local evidence and persistence

The Kaggle session was stopped after downloads and verification. The main
downloaded artifacts are in `/Users/zycy/Downloads/`:

- `dreamer_pilot_artifacts.zip` (137 MB; `unzip -t` passed; contains source,
  logs, metrics, config, evaluation CSV, checkpoint, and `progress.csv`)
- `params.pkl` (63.5 MB)
- `metrics.jsonl`, `eval_episodes.csv`, `summary.json`,
  `train_summary (1).json`, and `train_subprocess (1).log`

The zip was checked for corruption and contains the expected checkpoint and
training files. No large training artifact is committed to Git.

Report-ready figures generated only from the downloaded `progress.csv` and
`eval_episodes.csv` are committed under `output/figures/`:

- `kaggle_pytorch_baseline_training_return.png`
- `kaggle_pytorch_baseline_eval_comparison.png`
