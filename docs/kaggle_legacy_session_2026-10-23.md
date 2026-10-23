# Kaggle legacy session evidence — 2026-10-23

This file preserves the state of the stopped Kaggle draft session before the
Python 3.10/TensorFlow 2.15 recovery attempt. It is evidence, not a training
result.

## Session state

- Kaggle notebook: `notebookb3d0ba28e9`
- Accelerator: `GPU T4 x2`
- Session state when stopped: `Session stopped`
- Last observed quota display: approximately `02:13 / 30 hrs`
- Main Dreamer training: not successfully started
- Active failure phase: `preflight`

## Last failure

```text
FAILED in preflight: RuntimeError: expected one occurrence in tools.py, found 0:
'class Module(tf.Module):\n  def save(self, filename):'
```

The failure came from the incremental compatibility patcher finding a stale
or partially patched generated checkout. It does not establish that the
Dreamer algorithm or the GPU is unusable.

## Earlier failures retained for the report

- Missing `dm_env` and `mujoco` dependencies.
- Old GRU `get_initial_state()` API.
- `float16`/`float32` concatenation in the RSSM path.
- Scalar decoder dimension passed as a non-integer.
- TensorFlow Probability distribution dtype mismatch.
- Keras 3 variable discovery found zero model parameters.
- Keras 3 optimizer variable and loss-scaling API mismatch.
- Notebook bootstrap ordering error where `repo` was undefined.

## Interpretation

The previous route is closed. No scientific claim should be based on this
session. The replacement route uses a separate generated directory, Python
3.10, TensorFlow 2.15, Keras 2.15, TensorFlow Probability 0.23, float32
precision, and a bounded ten-update gate before any longer training.
