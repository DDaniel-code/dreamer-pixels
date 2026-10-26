# Seed-0 overnight 300k checkpoint evidence

This directory records the evaluation of the `300,000 environment-step` periodic checkpoint from overnight run `seed0_overnight_20261024_01`. It is not the old 100k experiment and it is not the overnight shutdown evaluation at 330,732 sampled environment steps.

The downloaded Kaggle artifact was `stage_evaluation_returns.csv`. It contains ten rows for each of the 100k, 200k, and 300k stage checkpoints. `eval_episodes.csv` is the exact ten-row 300k subset, and `source_stage_evaluation_returns.csv` preserves the complete downloaded source. The 300k rows were independently recomputed as:

`258.4441 ± 37.9997` (mean and sample standard deviation over ten evaluation episodes, seeds 1000--1009).

The checkpoint file is identified by the local stage-audit path and SHA256 in `checkpoint_manifest.json`. The periodic boundary is 150,000 decisions / 300,000 environment steps; the run later sampled 165,366 decisions / 330,732 environment steps in total. The `itr=149999` value is recorded from the periodic-save boundary convention and log; local payload inspection was not possible because the Mac environment does not have PyTorch. The Kaggle child process did load the checkpoint for evaluation.

The evaluation code was built into the Kaggle notebook from the working tree recorded in the manifest. The runner had an uncommitted local diff; its hash is recorded rather than silently treating the working tree as a clean commit.

Limitations:

- The checkpoint-specific Kaggle `summary.json` and `eval_episodes.csv` were not downloaded separately; the checked-in files are a transparent derived subset of the downloaded aggregate CSV.
- Random and training-initial controls were skipped in this stage notebook, so this directory does not claim a three-policy comparison.
- This is one training seed and ten evaluation episodes, not a multi-seed stability estimate.
