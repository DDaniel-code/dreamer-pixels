# Reproduction protocol

## Goal

Use `cartpole_swingup` to establish the pixel-input, training, evaluation, and logging pipeline first; then extend to `walker_walk` and `ball_in_cup_catch`. The main paper comparisons are the continuous-control scores in Appendix G and the action-repeat sensitivity in Figure 12. No local result from Phase 1 should be described as a paper reproduction result.

The paper reports a Dreamer reference score of 833.66 for `Cartpole Swingup` and 823.39 as the continuous-control average. These are source-material references, not measurements from this project.

## Fixed settings

| Item | Convention | Status |
|---|---|---|
| Observation | 64×64×3 RGB pixels | Confirmed by paper; runtime pending |
| Action repeat | R=2 | Confirmed by paper; wrapper semantics pending |
| Episode | 1000 environment steps | Confirmed by paper |
| Decision steps | Agent decisions after action repeat | Must be checked against the actual wrapper |
| Training budget | Target: 1,000,000 environment steps | Paper/figure convention; not run |
| Task | Start with `cartpole_swingup` | Fixed |
| Random seeds | Record every seed; exact seed set TBD | Not fixed |
| Evaluation | Episodic return and learning curves | Fixed |
| Other metrics | Do not use classification accuracy to describe control performance | Fixed |

## Training and evaluation protocol

Before formal reproduction, run dependency checks, a random-policy pixel check, and a very short debug run on Linux GPU, followed by a small-budget preflight. Formal experiments must record the source commit, configuration, environment versions, GPU, seed, environment steps, and log path. Evaluation should use no exploration noise or the upstream test planner with a fixed number of evaluation episodes. Fill in the exact episode count only after reading and running the upstream interface; do not invent it.

## Not yet verified

There is currently no passing record for training, pixel rendering, or paper-result reproduction.

## Not yet confirmed

- The item-by-item difference between all appendix defaults and the current upstream configuration.
- The dm_control version, MuJoCo renderer, and Linux CUDA combination.
- The exact locations where environment steps and decision steps are counted in training logs.
- The approved university GPU-hour budget and job scheduling method.
