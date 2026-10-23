# Kaggle Overnight Experiment Design

## Status

Design only. This document has not started a long-running training job.

The school GPU allocation is unavailable, so Kaggle is being used as a
temporary feasibility platform. The experiment must still preserve the
Dreamer V1 target: the authors' original TensorFlow 1 implementation remains
the primary run, and a newer Dreamer version must not be substituted silently.

## Objective

Obtain the first reproducible learned-policy evidence for `cartpole_swingup`
from pixel observations, while producing recoverable checkpoints and enough
logs to diagnose failure. The result will be classified as one of:

1. a short Dreamer V1 learning run;
2. a compatibility failure with actionable diagnostics; or
3. a bounded infrastructure result only.

It will not be called a paper reproduction unless the selected implementation,
configuration, evaluation protocol, and learning curve are all verified.

## Execution gates

### Gate 0: fresh-session environment check

Run in a clean Kaggle GPU session and record:

- Python, TensorFlow, TensorFlow Probability, CUDA, driver, MuJoCo, and GPU versions;
- the pinned upstream commit `d517542f9777c53864ff6f2683cea88be9ea98ac`;
- `dm_control` import, EGL rendering, and the `cartpole_swingup` pixel shape;
- a small TensorFlow GPU operation and device placement;
- the exact installation log.

If the legacy TensorFlow 1 dependency set cannot be installed or imported in a
short bounded check, stop the long run. Do not spend several hours compiling
or repeatedly retrying an incompatible stack.

### Gate 1: short faithful debug run

Use the author's real training entry point and the smallest available debug
configuration. Verify, in one short run:

- replay sequence creation;
- world-model and actor/critic forward passes;
- one optimizer update;
- log creation;
- checkpoint creation; and
- checkpoint restore followed by evaluation.

The command must be derived from the upstream repository rather than invented
for this project. Any parameter not confirmed from the pinned code or config
must remain marked `TBD` in the run record.

### Gate 2: overnight single-seed run

Only after Gates 0 and 1 pass, run one `cartpole_swingup` seed for a bounded
wall-clock budget, initially 4--6 hours and never beyond the Kaggle session
limit. Use one visible GPU if Kaggle offers that option; the original code is
not assumed to benefit from two GPUs.

The run must:

- write checkpoints at regular environment-step intervals;
- write a machine-readable metrics file containing environment steps,
  decision steps, losses, returns, and wall-clock time;
- evaluate periodically with a fixed evaluation protocol;
- save the final checkpoint, logs, and plots under `/kaggle/working/`; and
- finish with a compact summary that can be reviewed without rerunning the
  notebook.

The first long run is intentionally one seed. It is a feasibility and
debugging run, not a statistically sufficient reproduction. If it learns, the
next runs should use at least two additional seeds and the same configuration.

## What to measure

Primary metrics:

- episodic return versus environment steps;
- evaluation return versus environment steps;
- seed, action repeat, pixel shape, and environment-step/decision-step counts;
- checkpoint existence and successful restore.

Diagnostic metrics, when exposed by the upstream logger:

- image reconstruction loss;
- reward and continuation prediction losses;
- RSSM/KL losses;
- actor and critic losses; and
- episode length and wall-clock throughput.

The random-policy returns already measured on Kaggle remain infrastructure
baselines. They must not be plotted as learned performance or compared to the
paper's learned curve as if they were the same method.

## Recovery and persistence

The notebook should be executed as a saved Kaggle version rather than relying
only on an interactive cell. Interactive sessions have an idle-time limit;
saved versions execute from a clean state and preserve their generated output.
The run should use `/kaggle/working/` for checkpoints and logs, remain below
the available output limit, and save a final notebook version after completion.

Before starting the long run, record the current quota and accelerator type.
After completion or failure, record the remaining quota and the exact session
status. Never assume that a running session will survive a browser disconnect;
checkpointing and saved outputs are the recovery mechanism.

## Stop conditions

Stop immediately if any of the following occurs:

- the code is not the pinned Dreamer V1 implementation;
- the environment renders blank, malformed, or non-64 x 64 x 3 pixels;
- losses become NaN or Inf;
- checkpoints cannot be written or restored;
- the process repeatedly restarts without progress;
- the session approaches its runtime limit without a recoverable checkpoint;
- GPU memory errors recur after one controlled reduction of batch size or
  sequence length; or
- the run is only consuming quota without producing new diagnostics.

## Expected outcomes

| Outcome | Interpretation | Next action |
|---|---|---|
| V1 imports and learns for one seed | Feasible Kaggle preflight | Repeat with two more seeds and fixed evaluation |
| V1 imports but does not learn | Useful debugging evidence | Inspect losses, throughput, config, and checkpoint restore |
| V1 cannot import | Legacy-stack compatibility blocker | Preserve logs; do not replace V1 silently |
| Session ends early | Infrastructure limitation | Resume from the latest saved checkpoint if supported |
| Only random/environment checks pass | Integration evidence | Keep report honest; wait for a compatible Linux runtime |

## References

- Kaggle Notebooks documentation: https://www.kaggle.com/docs/notebooks
- Kaggle efficient GPU usage: https://www.kaggle.com/docs/efficient-gpu-usage
- Original Dreamer V1 implementation: https://github.com/google-research/dreamer
