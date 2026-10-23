# Implementation differences

## Paper and selected upstream

The paper describes Dreamer V1: an RSSM world model with image reconstruction and reward prediction, plus actor and critic models trained through latent imagination. For today's pilot, the selected implementation is the authors' simpler TensorFlow 2 repository, which implements the same Dreamer V1 paper target. Its README specifies TensorFlow 2.2.0-era dependencies and the `dreamer.py` entry point.

Selected upstream version: `danijar/dreamer@56d4d444dfd0582b0e79dab80aebbea74c0ce40d`. The original TensorFlow 1 implementation remains pinned as historical reference: `google-research/dreamer@d517542f9777c53864ff6f2683cea88be9ea98ac`. The algorithm is not rewritten.

## Differences in this project

1. This project adds documentation, configuration records, a Kaggle pilot runner, and validation checks; it does not copy or relicense upstream source.
2. The first task is fixed to `cartpole_swingup`, while the author README examples include `walker_run`.
3. The Kaggle working copy applies only compatibility patches required by current TensorFlow/Keras APIs and an evaluation guard so `training=False` cannot trigger optimizer updates. These patches are recorded in the Kaggle output manifest.
4. The Kaggle notebook installs `dm-env==1.6` and the pinned `mujoco==3.3.7` wheel explicitly alongside the `dm-control==1.0.34` wheel because the no-dependency install used to avoid the unavailable `labmaze` build path otherwise leaves required runtime modules absent. This changes packaging only, not the algorithm or environment contract.
5. Scheduler scripts do not assume Slurm; if needed, an explicitly marked optional template may be added later.
6. Paper results, upstream defaults, and local measurements are recorded separately.

## Correctness risks

The selected TensorFlow 2 code uses old APIs such as experimental mixed precision and `experimental_run_v2`, while current Kaggle images use newer TensorFlow/Keras. The pilot therefore has a bounded compatibility gate and explicit source-level replacements. It must not silently become Dreamer V2/V3 or a third-party reimplementation. The original TensorFlow 1 stack remains historical reference only for today's run.
