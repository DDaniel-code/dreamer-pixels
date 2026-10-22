# Implementation differences

## Paper and upstream

The paper describes Dreamer V1: an RSSM world model with image reconstruction and reward prediction, plus actor and critic models trained through latent imagination. The original repository is the authors' TensorFlow 1 implementation. Its README specifies Ubuntu 18, TensorFlow GPU 1.13.1, TensorFlow Probability 0.6.0, dm_control, and related dependencies.

Pinned upstream version: `google-research/dreamer@d517542f9777c53864ff6f2683cea88be9ea98ac`. This version is the reproduction baseline; the algorithm is not rewritten.

## Differences in this project

1. This project adds documentation, configuration records, and a dependency-free environment check; it does not copy or relicense upstream source.
2. The first task is fixed to `cartpole_swingup`, while the author README examples include `walker_run`.
3. This project separates macOS development checks from Linux GPU training; it does not modify the algorithm to make TensorFlow 1 train fully on macOS.
4. Scheduler scripts do not assume Slurm; if needed, an explicitly marked optional template may be added later.
5. Paper results, upstream defaults, and local measurements are recorded separately.

## Correctness risks

The original dependencies are old and are likely incompatible with the current Python 3.14/macOS ARM environment. Directly changing dependencies or replacing the implementation with V2/V3 would compromise the reproduction target. Establish a Linux GPU environment and lock an installable legacy dependency set before formal runs.
