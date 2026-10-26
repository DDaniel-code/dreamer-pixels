# Seed-2 100k Kaggle execution evidence

This directory records the downloadable evidence recovered from Kaggle
notebook `danielyang3/notebook9c3a976209`, run ID
`seed2_matching_20261025_01`, Version 1.

The notebook output page reported **0 files**, and the editor showed the
session as stopped. The notebook itself was downloadable and is preserved as
`kaggle_notebook_executed.ipynb`. Its recorded outputs show successful return
from the preflight, debug, training, evaluation, and minimal-export wrapper
phases; the training wrapper reported about 69 minutes, evaluation about 0.5
minutes, and minimal export about 0.5 minutes.

The notebook records these ephemeral paths:

```text
/kaggle/working/dreamer_seed2_matching_20261025_01/train_run/run_0/params.pkl
/kaggle/working/dreamer_seed2_matching_20261025_01/initial_policy.pkl
```

Those paths are not local files and were not present in Kaggle's downloadable
Output Data. No raw seed-2 `eval_episodes.csv`, `summary.json`, checkpoint,
progress file, complete log, or ZIP was recovered. Consequently this is an
execution/persistence audit, not a verified seed-2 numerical result. The
aggregate values embedded in the notebook's seed-1 provenance gate are not
seed-2 results and are intentionally not copied here as seed-2 metrics.

The downloaded notebook SHA256 is recorded in `manifest.json`.

