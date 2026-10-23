# Implementation differences

## Paper and selected upstream

The paper describes Dreamer V1: an RSSM world model with image reconstruction and reward prediction, plus actor and critic models trained through latent imagination. The active pilot uses the PyTorch implementation at `juliusfrost/dreamer-pytorch@ccea6ae4a397a94c328891bd78e81d52dd156cb6`, with `rlpyt@f04f23db1eb7b5915d88401fca67869968a07a37`.

The TensorFlow 2 route at `danijar/dreamer@56d4d444dfd0582b0e79dab80aebbea74c0ce40d` and the original TensorFlow 1 route are now historical evidence only. The algorithm is not rewritten.

## Differences in this project

1. This project adds documentation, configuration records, a Kaggle pilot runner, and validation checks; it does not copy or relicense upstream source.
2. The first task is fixed to `cartpole_swingup`, while the author README examples include `walker_run`.
3. The PyTorch Kaggle working copy uses the upstream source unchanged and records one runtime compatibility shim when required: newer PyTorch distributions require `TanhBijector` to expose `domain` and `codomain`. The shim adds only those constraints and is recorded in `compatibility_patches.json`; it does not change the loss or model equations.
4. The Kaggle notebook installs `dm-env==1.6` and the pinned `mujoco==3.3.7` wheel explicitly alongside the `dm-control==1.0.34` wheel because the no-dependency install used to avoid the unavailable `labmaze` build path otherwise leaves required runtime modules absent. This changes packaging only, not the algorithm or environment contract.
5. Scheduler scripts do not assume Slurm; if needed, an explicitly marked optional template may be added later.
6. Paper results, upstream defaults, and local measurements are recorded separately.

## Correctness risks

The selected PyTorch repository is in maintenance mode, so modern Kaggle package APIs must be checked without blindly reinstalling PyTorch. The pilot therefore has a bounded import, pixel/reset, real-update, parameter-change, checkpoint, and fresh-process evaluation gate. It must not silently become Dreamer V2/V3 or a from-scratch reimplementation.
