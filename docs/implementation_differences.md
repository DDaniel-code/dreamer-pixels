# Implementation differences

## Paper and selected upstream

The paper describes Dreamer V1: an RSSM world model with image reconstruction and reward prediction, plus actor and critic models trained through latent imagination. The active pilot uses the PyTorch implementation at `juliusfrost/dreamer-pytorch@ccea6ae4a397a94c328891bd78e81d52dd156cb6`, with `rlpyt@f04f23db1eb7b5915d88401fca67869968a07a37`.

The TensorFlow 2 route at `danijar/dreamer@56d4d444dfd0582b0e79dab80aebbea74c0ce40d` and the original TensorFlow 1 route are now historical evidence only. The algorithm is not rewritten.

## Differences in this project

1. This project adds documentation, configuration records, a Kaggle pilot runner, and validation checks; it does not copy or relicense upstream source.
2. The first task is fixed to `cartpole_swingup`, while the author README examples include `walker_run`.
3. The PyTorch Kaggle working copy uses the upstream source unchanged and records one runtime compatibility shim when required: newer PyTorch distributions require `TanhBijector` to expose `domain` and `codomain`. The shim adds only those constraints and is recorded in `compatibility_patches.json`; it does not change the loss or model equations.
4. The Kaggle notebook installs `dm-env==1.6`, `mujoco==3.3.7`, and `dm-control==1.0.34` with `--no-deps`, reusing the previously successful installation record. It then installs the explicitly declared `glfw==2.11.0` runtime wheel because the first direct import reported that module as missing. This changes packaging only, not the algorithm or environment contract.
5. The fixed PyTorch agent derives its image shape from the environment. The local runner therefore removes a duplicate `model_kwargs.image_shape` argument that caused a constructor `TypeError`; no model equation or loss was changed.
6. The fixed `rlpyt` runtime declares `pyprind` in its test/runtime path but does not list it in `INSTALL_REQUIRES`. The pilot installs `pyprind==2.11.3` without dependencies after the actual import traceback identified the missing module. This is recorded as a runtime packaging compatibility step.
7. Scheduler scripts do not assume Slurm; if needed, an explicitly marked optional template may be added later.
8. Paper results, upstream defaults, and local measurements are recorded separately.

## Correctness risks

The selected PyTorch repository is in maintenance mode, so modern Kaggle package APIs must be checked without blindly reinstalling PyTorch. The pilot therefore has a bounded import, pixel/reset, real-update, parameter-change, checkpoint, and fresh-process evaluation gate. It must not silently become Dreamer V2/V3 or a from-scratch reimplementation.
