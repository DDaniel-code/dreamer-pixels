# Midterm Report - Reproducing Dreamer V1 from Pixels

## Abstract

This project investigates whether a Dreamer V1 pipeline can be reproduced from pixel observations in the DeepMind Control Suite. The immediate target is `cartpole_swingup`; `walker_walk` and `ball_in_cup_catch` are reserved for later experiments. The completed pilot uses the fixed PyTorch implementation at `juliusfrost/dreamer-pytorch@ccea6ae4a397a94c328891bd78e81d52dd156cb6`, with its pinned `rlpyt` submodule. The TensorFlow repositories remain historical references. The project has completed a bounded GPU-backed debug gate and an independent 100,000-environment-step single-seed Dreamer run, while explicitly not claiming a paper reproduction.

The main empirical result is a bounded CMU-verified Kaggle pilot. A Tesla T4 session ran PyTorch CUDA, reused the previously validated `dm-control --no-deps` route, rendered `cartpole_swingup` pixels through EGL, passed a ten-update Dreamer gate, and completed 50,000 decision steps (100,000 underlying environment steps). The run produced 4,900 real optimizer updates and changed the world model, actor, and critic parameters. On the same ten evaluation seeds, random, initial-policy, and final Dreamer mean returns were 55.540, 70.684, and 181.701 respectively. These are preliminary single-seed results, not paper-level performance.

## 1. Problem, motivation, and research target

Dreamer learns a latent world model from image observations and improves behavior by imagining future latent trajectories. This makes it a useful test case for a project centered on reinforcement learning from pixels: the input is high-dimensional, the control objective is continuous, and the agent must learn both representation and dynamics. The project asks a practical reproduction question: can the central Dreamer V1 pipeline be reconstructed and evaluated under a documented environment contract, before adding a research extension?

The first target is `cartpole_swingup` because it is small enough for debugging while still requiring visually grounded control. The two follow-up tasks are `walker_walk` and `ball_in_cup_catch`. The project will report episodic return and learning curves. Classification accuracy is not an appropriate metric for this control problem.

The target paper is Hafner et al., *Dream to Control: Learning Behaviors by Latent Imagination* (2019). The executable code reference for this pilot is `juliusfrost/dreamer-pytorch` at commit `ccea6ae4a397a94c328891bd78e81d52dd156cb6`, with `rlpyt` at `f04f23db1eb7b5915d88401fca67869968a07a37`. It is used as a maintained PyTorch implementation of the Dreamer V1 target, not as Dreamer V2 or V3. The original `google-research/dreamer` and `danijar/dreamer` repositories remain pinned audit references.

## 2. Related work and implementation audit

The paper describes a recurrent state-space model (RSSM) that separates a deterministic recurrent state from a stochastic latent state. An encoder maps pixels to embeddings. The world model predicts rewards, continuation, and future latent states. The actor is optimized through imagined trajectories, and the critic estimates imagined returns. The replay buffer stores sequences of real interaction, rather than isolated images, so that the recurrent model can learn temporal structure.

The repository audit distinguishes three layers. First, the paper is the source for the algorithmic target and reported evaluation protocol. Second, the pinned author repository is the source for the concrete V1 code path and default experiment configuration. Third, the local project adds documentation, configuration records, tests, and bounded smoke-test tooling. No new learning algorithm has been inserted into the upstream implementation.

The paper and repository are not identical in every operational detail. The repository contains defaults for task selection, sequence length, image size, action repeat, optimization schedules, logging, and checkpointing. Some of those values are represented in code or configuration rather than stated in one concise table in the paper. Therefore, the local protocol marks any value that still needs confirmation as `TBD` instead of inventing a paper number. The local project also separates environment steps from decision steps: with action repeat 2, 1,000 environment steps correspond to 500 policy decisions.

The original repository's license and attribution remain upstream-owned. The local repository does not relicense copied upstream source. The project files identify the upstream URL and commit so that the exact code can be retrieved again. The newer `danijar/dreamer` repository has different maintenance and dependency assumptions and is therefore treated as a reference, not as the selected implementation.

## 3. Methodology and reproduction protocol

### Environment and observation contract

The first environment is DeepMind Control Suite `cartpole_swingup`. The initial pixel wrapper uses 64 x 64 RGB observations. The action space is one-dimensional and bounded by the environment action specification. The current smoke protocol uses action repeat 2. The environment step count is the number of calls to `env.step`; the decision count is the number of sampled actions. Episodic return is the sum of environment rewards within an episode, with no normalization applied in the smoke test.

The formal run will preserve the observation key, pixel shape, action bounds, action repeat, seed, environment-step budget, checkpoint directory, and evaluation episode count in an experiment record. A run is not considered comparable if those fields are changed without being recorded.

### Training and evaluation plan

The primary comparison is the paper's continuous-control result for the selected task, with learning curves as the central artifact. The project will first reproduce a single-task baseline, then extend to additional tasks only after the baseline pipeline is stable. The intended progression is:

1. Very short Dreamer debug run proving three optimizer paths update and a checkpoint restores in a fresh process.
2. One bounded `cartpole_swingup` Dreamer V1 training run with raw logs and a return curve.
3. Final Dreamer and random-policy evaluation on the same ten environment seeds.
4. Deterministic exports and only then optional world-model prediction diagnostics.

The completed midterm pilot uses one training seed, 50,000 decision steps (100,000 underlying environment steps), and evaluation seeds `1000`--`1009`. The formal schedule uses `prefill=1000`, `train_every=1000`, `train_steps=100`, `batch_size=16`, `batch_length=50`, and replay capacity 60,000 decision transitions; the exact record is in `docs/kaggle_100k_baseline_2026-10-23.md`. The project does not label this single-seed pilot as a reproduction. It reports episodic return, optimizer-update counts, and the underlying environment-step axis.

### Baselines and planned iteration

The mandatory baseline is the selected Dreamer V1 implementation with the pinned configuration. A random policy is an auxiliary control evaluated under the same ten final environment seeds; it is not the project baseline. Stale-frame/dropout and world-model-prediction experiments are postponed until the actor-critic baseline has a working checkpoint, learning curve, and final evaluation path.

The midterm rubric asks for evidence of iteration. The two random-policy seeds reported below are the first controlled configuration check. The next iteration is a change from CPU/macOS inspection to a Linux EGL-backed Kaggle environment, because the local Mac did not provide a usable MuJoCo OpenGL backend. This is an infrastructure adjustment, not an algorithmic improvement.

## 4. Compute feasibility and bounded experiments

The local Mac is useful for documentation, configuration, tests, and static interface checks. It is not currently a reliable pixel-training environment: the project-local `dm-control` installation could be prepared, but MuJoCo rendering could not find a usable OSMesa or EGL/OpenGL backend under the bounded local check. TensorFlow 1.13.1 and TensorFlow Probability 0.6.0 were not installed locally. No long CPU or MPS training was started.

The CMU Kaggle account provided a useful, bounded alternative. The notebook used a Tesla T4 and PyTorch `2.11.0+cu128`. A synthetic CUDA forward/backward check completed, and the standard dependency-resolving `pip install dm-control` route was not reused because it attempted to build `labmaze`. Instead, the notebook used the previously validated `dm-control --no-deps` path with `dm-env==1.6`, `mujoco==3.3.7`, and `dm-control==1.0.34`, plus `glfw==2.11.0` and `pyprind==2.11.3`. Setting `MUJOCO_GL=egl` allowed headless rendering.

The resulting pilot used `cartpole_swingup`, a 64 x 64 x 3 observation, camera 0, action repeat 2, and a 500-decision/1,000-environment-step episode limit. The debug gate completed ten real updates with finite diagnostics and changed all three module parameter groups. The independent formal run completed 50,000 decision steps and 100,000 environment steps, with cumulative updates 0 at the 1,000-decision prefill boundary and 4,900 at the final checkpoint. The GPU session was stopped after the checkpoint and exported archive were downloaded and checked.

| Run | Policy | Seed(s) | Environment steps | Decision steps | Pixel shape | Action repeat | Episodic return | Status |
|---|---|---:|---:|---:|---|---:|---:|---|
| 1 | Random control | 1000--1009 | 1,000/episode | 500/episode | 64 x 64 x 3 | 2 | 55.5404 mean, 42.5973 sample std | Passed |
| 2 | Initial policy | 1000--1009 | 1,000/episode | 500/episode | 64 x 64 x 3 | 2 | 70.6835 mean, 3.1504 sample std | Passed |
| 3 | Final Dreamer pilot | 1000--1009 | 1,000/episode | 500/episode | 64 x 64 x 3 | 2 | 181.7011 mean, 27.5705 sample std | Passed |

These results provide evidence that the actor-critic path ran and that the final model differs from the initial policy. They do not establish a statistically reliable improvement, because there is one training seed and a bounded 100,000-environment-step budget. They also reveal a reproducibility issue that must be recorded: Kaggle's current image requires a headless EGL setup and a dependency workaround for `labmaze`. The university Linux environment may resolve that issue differently.

Kaggle is therefore sufficient for small smoke tests and a bounded pilot. It is not yet the preferred place for a formal reproduction because sessions, package versions, quota limits, and persistent storage are less controlled than a university allocation. The requested university GPU access was not available for this run; no remote workflow was executed and no scheduler has been assumed.

## 5. Results, failure analysis, and limitations

### Results available at midterm

Passed checks include project metadata tests, Git whitespace validation, isolated-environment setup, the synthetic CUDA forward/backward check, headless MuJoCo pixel rendering on Kaggle, two bounded random rollouts, and short GIF generation. These results are reproducible at the notebook level through the CMU-verified Kaggle account and the saved draft notebook URL recorded in the project handoff.

The local macOS pixel probe failed because the available renderer backend was insufficient. The normal `dm-control` installation also encountered a `labmaze` build problem in Kaggle. These are meaningful failure cases rather than missing results: they identify environment assumptions that must be handled in the final Linux setup. The first non-CMU Kaggle notebook session also returned an authentication/internal error, so that session is not used as evidence. The correct CMU account subsequently produced a working T4 session.

### What remains unverified

The following claims are intentionally not made: this is not a full Dreamer V1 paper reproduction, not a multi-seed result, and not evidence of reaching the paper's numerical score. The checkpoint was loaded for evaluation in a fresh process, but full optimizer/replay continuation was not tested. No claim is made about university GPU-hours. The exact university login host, storage policy, and scheduler interface remain to be confirmed.

The most important scientific limitation is replication: the pilot has one training seed and one task. Random-policy return varies substantially across seeds and is not a quality benchmark. The next scientific evidence should be two additional independently recorded training seeds with a pre-registered evaluation protocol.

## 6. Next steps

The immediate report task is to include the completed 100k pilot record, its raw-artifact path, and its limitations. A future experiment should add multiple training seeds; it should not silently combine the 100k run with the earlier 3k pilot or call either one a paper reproduction. The university should confirm the SSH host and username, VPN or jump-host requirements, whether the login node is separate from compute nodes, whether a scheduler is used, account and partition names if Slurm is used, CUDA and driver versions, Python/Conda/module or container policy, project and scratch paths, and storage quotas.

The report's immediate deliverable is a transparent midterm state: the repository is organized, the source choice is fixed, the local and Kaggle failures are documented, the environment contract is tested, a real 100k-step Dreamer update path is measured, and the remaining uncertainty is narrowed to replication and full paper-level comparison. The reported values are exported measurements, not fabricated example performance numbers.

## References

1. Hafner et al. *Dream to Control: Learning Behaviors by Latent Imagination*. arXiv:1912.01603. <https://arxiv.org/abs/1912.01603>
2. Google Research. Original Dreamer V1 implementation. <https://github.com/google-research/dreamer>
3. Danijar. TensorFlow 2 Dreamer reference implementation. <https://github.com/danijar/dreamer>
4. Google DeepMind. DeepMind Control Suite. <https://github.com/google-deepmind/dm_control>
5. Kaggle. Notebooks documentation. <https://www.kaggle.com/docs/notebooks>
6. Kaggle. Efficient GPU usage. <https://www.kaggle.com/docs/efficient-gpu-usage>
