# Midterm Report — Dreamer from Pixels

## Executive summary

This project is in the preparation and reproducibility-audit phase. The target
is the original Dreamer V1 paper, starting with DeepMind Control Suite
`cartpole_swingup` from pixel observations. No GPU training has been completed,
and this report does not claim a paper reproduction result.

The practical conclusion for the midterm is that substantial progress can be
made without the university allocation: the paper-to-code audit, environment
contract, run protocol, configuration record, CPU-side checks, and a bounded
Kaggle smoke test can all be completed first. A full Dreamer V1 reproduction
still requires a Linux GPU environment because the authors' implementation is
an old TensorFlow 1 stack and the experiment budget is much larger than a
local CPU check.

## Research target and implementation choice

We selected the paper authors' original TensorFlow 1 implementation at commit
`d517542f9777c53864ff6f2683cea88be9ea98ac` in
`google-research/dreamer`. This keeps the reproduction target tied to Dreamer
V1 rather than silently changing to Dreamer V2 or V3. The repository also
records the paper, upstream configuration differences, code map, and cluster
questions.

The first environment is `cartpole_swingup`. `walker_walk` and
`ball_in_cup_catch` remain follow-up tasks. The initial comparison target is
the paper's continuous-control evaluation in Appendix G; the action-repeat
sensitivity in Figure 12 is a secondary comparison after the base pipeline is
working. The primary project metrics are episodic return and learning curves.

## What can be completed before GPU access

The following items are CPU-feasible or require only a very small GPU budget:

1. Verify configuration loading, path handling, seed recording, checkpoint and
   log-directory conventions, and CLI behavior.
2. Verify the environment contract with a random policy: pixel shape, action
   bounds, action repeat, episode termination, reward accumulation, and the
   distinction between environment steps and decision steps.
3. Generate a short local frame montage or GIF if the local MuJoCo graphics
   backend works.
4. Run a bounded model-forward/backward smoke test after the Linux dependency
   environment is available.
5. Use Kaggle for a bounded 1,000–10,000-step import and training smoke test,
   not as evidence of final learning or paper reproduction.

These checks are valuable for the report because they reduce integration risk
without pretending that a short run establishes a learning curve.

## Local verification record

| Check | Result | Evidence or limitation |
|---|---|---|
| Dependency-free project tests | Passed | `2 passed` with Conda Python 3.14.6 |
| Git whitespace check | Passed | `git diff --check` |
| Project-local isolated environment | Passed | `.venv` created; installation did not modify system Python |
| `dm-control` import and pixel rollout | Failed on this Mac | MuJoCo needs an available OpenGL backend; `MUJOCO_GL=osmesa` reported that OSMesa is not installed, while the default graphical initialization did not complete within the bounded check |
| Dreamer V1 training | Not run | TensorFlow 1.13.1 and the authors' full Linux stack are not installed |
| Kaggle 1,000-step smoke test | Failed | Logged in successfully, but the new notebook session returned `Firebase: Error (auth/internal-error)` after retry and refresh; notebook URL: `https://www.kaggle.com/code/daniel999929/notebook8d89511e5d/edit` |
| University GPU run | Not run | GPU allocation and cluster interface are still pending |

The random rollout script is real code at
`scripts/random_rollout.py`; it records both environment steps and decision
steps and writes local evidence under `artifacts/random_rollout/`. Generated
media and experiment outputs are ignored by Git.

## Kaggle assessment

Kaggle is suitable for a smoke test if the notebook session can import the
required packages and obtain a GPU. A short run can test imports, one batch of
image preprocessing, a tiny forward/backward pass, checkpoint writing, and
resume behavior. It cannot by itself establish that Dreamer V1 matches the
paper.

The risks are session limits, queue or quota availability, package
compatibility, and the age of the original TensorFlow 1 code. Kaggle should
therefore be treated as a temporary preflight environment. The formal run
should move to the university Linux environment after its CUDA, driver,
storage, and scheduling details are confirmed.

## What remains blocked or unverified

- A Linux-compatible installation of the pinned Dreamer V1 stack.
- The exact university GPU host, allocation, storage, and scheduler workflow.
- A short GPU debug run and a 1M-environment-step reproduction.
- Multiple seeds, evaluation episode count, learning curves, and comparison
  against the paper's Appendix G values.
- Whether Kaggle can start a usable execution session today; authentication itself succeeded, but the editor session failed.

## Next submission-ready step

Run the bounded Kaggle smoke test if a logged-in Kaggle session is available.
Record the notebook URL, package versions, GPU name, wall-clock time, step
budget, checkpoint/resume result, and any failure logs. Report it explicitly as
a smoke test. In parallel, ask the university for the minimum cluster
information listed in `docs/cluster_setup.md`.

## Sources

- Dreamer V1 paper: <https://arxiv.org/abs/1912.01603>
- Authors' implementation: <https://github.com/google-research/dreamer>
- TensorFlow 2 reference implementation: <https://github.com/danijar/dreamer>
- DeepMind Control Suite: <https://github.com/google-deepmind/dm_control>
- Kaggle Notebook documentation: <https://www.kaggle.com/docs/notebooks>
- Kaggle efficient GPU usage: <https://www.kaggle.com/docs/efficient-gpu-usage>
