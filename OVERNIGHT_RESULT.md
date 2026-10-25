# Overnight Dreamer V1 Result

Status: **Partially complete**. The bounded formal run, evaluation, export, local download, and GPU shutdown completed. The 500,000-environment-step target was not reached because the training wall-clock limit took priority.

## Run

- Run ID: `seed0_overnight_20261024_01`
- Task: `cartpole_swingup`, camera 0, 64x64 RGB, action repeat 2
- Training seed: 0 (one training seed; this is not two seeds despite the separate 100k and overnight runs)
- Dreamer PyTorch commit: `ccea6ae4a397a94c328891bd78e81d52dd156cb6`
- `rlpyt` commit: `f04f23db1eb7b5915d88401fca67869968a07a37`
- Effective schedule: prefill 1,000 decisions; `train_every=1000`; `train_steps=100`; batch `16 x 50`; imagination horizon 15; replay capacity 60,000 decision transitions; `use_pcont=False`; anomaly detection and nested tqdm disabled.

## Status checks

| Check | Result |
|---|---|
| Persistence probe before training | PASS; saved-version bundle downloaded and fresh-process load evidence recorded |
| CPU dm-control import/reset/10 random steps | PASS |
| CUDA/EGL pixel preflight | PASS |
| Three-module debug update/checkpoint | PASS; finite nonzero diagnostics and fresh-process evaluation |
| Long training process | PASS; return code 0, stopped at wall-clock limit |
| Final evaluation process | PASS; return code 0 |
| Minimal export process | PASS; return code 0 |
| Final bundle local ZIP test and SHA256 | PASS; 53 entries, `unzip -t` clean |
| GPU session | CLOSED |

## Counts and evaluation

- Actual training counters: 165,366 decision steps, 330,732 environment steps, 16,500 optimizer updates.
- Training wall time: 12,601.9 seconds (about 210.0 minutes).
- Final evaluation wall time: 671.4 seconds.
- Evaluation seeds: 1000--1009, 10 episodes per policy; reported standard deviations are sample standard deviations across episodes.

| Policy | Mean return | Sample std. |
|---|---:|---:|
| Random | 45.6311 | 37.4796 |
| Training-initial | 73.7413 | 2.3887 |
| Final Dreamer | 322.0840 | 9.6868 |

The initial policy is the formal training model after prefill and before the first optimizer update (`itr=999`, `updates=0`), not a separately initialized comparison model. The final policy checkpoint changed the world model, actor, and critic parameters and was evaluated by a fresh Kaggle child process. The local Mac environment does not have PyTorch, so a second local `torch.load` was not performed; the local verification is based on the downloaded files, ZIP integrity, hashes, and the fresh Kaggle evaluation subprocess evidence.

## Local artifacts

The verified run directory is:

`/Users/zycy/Downloads/dreamer_experiments/seed0_overnight_20261024_01/`

It contains the extracted key artifacts, initial and final checkpoints, progress CSV, raw logs, evaluation CSV, three videos for seeds 1000--1002, the original final bundle, and `download_manifest.csv`. The bundle SHA256 is:

`a49517de599fc82568be2067de3d34d08431d00e8f9375837ffb1a6ca857e41b`

The report draft was rebuilt as a five-page PDF at `output/pdf/dreamer_pixels_midterm_report.pdf`; the pre-overnight PDF is preserved as `output/pdf/dreamer_pixels_midterm_report_100k_baseline.pdf`.

## Limitations and next step

This is a bounded single-training-seed extension, not a 500k run or a full Dreamer paper reproduction. It should not be used to claim statistical significance or stable task mastery. The next priority is to inspect the three fixed-seed videos and, if the protocol remains sound, add seeds 1 and 2 with the same configuration. Horizon sensitivity and 4-pixel random-shift augmentation remain proposal-aligned future experiments and were not run.
