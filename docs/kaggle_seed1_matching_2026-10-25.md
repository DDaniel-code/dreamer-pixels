# Seed-1 matching run status — 2026-10-25

This note records the bounded audit attempt. It is not a training result.

## Inputs and fixed route

- Seed-0 input: `seed0_overnight_20261024_01`, supplied to Kaggle as the
  stage-evaluation input dataset.
- Dreamer PyTorch: `juliusfrost/dreamer-pytorch`, commit
  `ccea6ae4a397a94c328891bd78e81d52dd156cb6`.
- `rlpyt`: `f04f23db1eb7b5915d88401fca67869968a07a37`.
- Environment: `cartpole_swingup`, camera 0, 64x64 RGB, action repeat 2.
- Fixed dm-control route: `dm-control==1.0.34 --no-deps`, with
  `dm-env==1.6`, `mujoco==3.3.7`, `glfw==2.11.0`, and `pyprind==2.11.3`.

## Checkpoint identity

The old overnight final evaluation was performed with:

`long_train_run/run_0/params.pkl`

Its recorded payload position is `itr=164999`, `cum_steps=164999`; the
reported 322.084 score therefore belongs to that file. The run also contains
separate 100k, 200k, and 300k stage checkpoints and a shutdown snapshot. File
presence alone does not make a checkpoint a reported evaluation result.

## Audit status

The Kaggle notebook installed the fixed dm-control route successfully. The
stage runner produced a downloaded aggregate CSV with ten Dreamer rows at
each of 100k, 200k, and 300k environment steps. The independently recomputed
300k result is `258.4441 +/- 37.9997`. A checkpoint-specific `summary.json`
and `eval_episodes.csv` were not downloaded separately; the actual aggregate
CSV and a transparent 300k subset are preserved in
`results/seed0_300k/`. No seed-1 training was started, and no seed-1 score
exists.

The runner and notebook were updated locally so that future evaluation
summaries include the evaluated checkpoint path, SHA256, payload counters,
optimizer-update provenance, and a non-log-boundary indicator. The notebook
was regenerated from the current runner. These local changes were not
committed or pushed in this stage.

## Next split (prepared, not started)

The next work is deliberately split so that each checkpoint evaluation can be
downloaded before another task begins:

### A — seed-0 300k evaluation

```bash
python scripts/kaggle_pytorch_pilot.py --phase eval \
  --checkpoint /path/to/decision_150000_env_300000.pkl \
  --initial-checkpoint /path/to/initial_policy.pkl \
  --output-dir /path/to/seed0_300k_eval \
  --progress-path /path/to/progress.csv
```

This uses the new per-episode CSV/JSONL/status files and skips completed
`policy:seed` pairs on restart. The already recovered A evidence is the
300k subset in `results/seed0_300k/`; no new GPU session was started for it.

### B — matching seed-1 training

```bash
python scripts/kaggle_pytorch_pilot.py --phase train_long --seed 1 \
  --n-steps 150000 --max-train-minutes <bounded-budget>
```

This is 150,000 decisions = 300,000 environment steps, with the overnight
seed-0 configuration held fixed except for the training seed. The command is
prepared only; it was not run. On the seed-0 measured throughput, training
alone is about 191 minutes before evaluation/export overhead, so it should
only start after a fresh time-budget check.

### C — seed-1 evaluation

```bash
python scripts/kaggle_pytorch_pilot.py --phase eval --seed 1 \
  --checkpoint /path/to/seed1_final.pkl \
  --initial-checkpoint /path/to/seed1_initial.pkl \
  --output-dir /path/to/seed1_eval \
  --progress-path /path/to/seed1_progress.csv
```

Plan approximately 10--15 minutes for the three-policy, ten-seed evaluation
depending on the Kaggle session. This is an estimate, not a result.
