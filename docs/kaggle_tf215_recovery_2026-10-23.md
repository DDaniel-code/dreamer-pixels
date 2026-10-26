# Kaggle TF 2.15 Recovery Evidence — 2026-10-23

## Scope

This was a bounded recovery attempt. The previous Python 3.13/Keras 3
session was stopped first. The new Kaggle draft notebook used an isolated
Python 3.10 environment and the pinned author TensorFlow 2 Dreamer source at
commit `56d4d444dfd0582b0e79dab80aebbea74c0ce40d`.

Notebook: <https://www.kaggle.com/code/danielyang3/notebookf935ec7be0/edit>

The notebook session used Kaggle `GPU T4 x2`, Internet on, and
`CUDA_VISIBLE_DEVICES=0`. It was stopped after the GPU preflight failure;
the notebook draft and its outputs remain available in Kaggle.

## Environment evidence

- Isolated interpreter: Python `3.10.22`.
- TensorFlow imported as `2.15.1`.
- The setup cell installed the requested pinned stack: Keras `2.15.0`,
  TensorFlow Probability `0.23.0`, NumPy `1.26.4`, plus dm-env, dm-control,
  MuJoCo, Gym, and plotting/export dependencies.
- `nvidia-smi -L` reported two Tesla T4 devices.
- In the isolated Python subprocess,
  `tf.config.list_physical_devices('GPU')` returned `[]`.
- The recovery runner therefore stopped with:
  `RuntimeError: TensorFlow sees no GPU; do not start a long pilot`.

The evidence shows that Kaggle exposed the devices at the system level but
the isolated TensorFlow runtime did not enumerate them. The exact dynamic
library/runtime cause was not pursued further because the recovery budget
explicitly forbids another compatibility loop.

## Source and compatibility evidence

The pinned checkout completed successfully:

```text
HEAD is now at 56d4d44 Add scores in JSON format.
56d4d444dfd0582b0e79dab80aebbea74c0ce40d
```

One necessary TF 2.15 change was applied for the removed
`tensorflow.keras.mixed_precision.experimental` API. The source patch path
also recorded absent optional symbols instead of failing before import. These
changes did not alter the Dreamer objective, but they were not sufficient to
reach a real update because GPU preflight failed first.

## Gate result

| Check | Result |
|---|---|
| Isolated Python 3.10 environment | Passed |
| Pinned author source checkout | Passed |
| TF 2.15 import | Passed |
| System GPU visibility | Passed (`nvidia-smi`, 2× T4) |
| TensorFlow GPU visibility | Failed (`[]`) |
| Pixel environment / real model update | Not run |
| Ten real updates | Not run; count 0 |
| Checkpoint save and fresh-process evaluation | Not run |
| Baseline training | Not started |
| Random vs Dreamer final evaluation | Not run |

No learned-performance claim or GPU-training claim is supported by this run.
