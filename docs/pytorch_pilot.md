# PyTorch Kaggle pilot

## Fixed source

- Repository: <https://github.com/juliusfrost/dreamer-pytorch>
- Commit: `ccea6ae4a397a94c328891bd78e81d52dd156cb6`
- Required `rlpyt` submodule: `f04f23db1eb7b5915d88401fca67869968a07a37`
- License: upstream MIT license, retained in the Kaggle checkout.
- The commit is the source revision associated with the repository's README
  comparison table. The table is not treated as our result.

## Route

`scripts/kaggle_pytorch_pilot.py` clones the fixed source with its submodule,
checks the existing Kaggle PyTorch/CUDA installation, and runs the upstream
Dreamer algorithm on `cartpole_swingup`. It does not install or replace
PyTorch. The environment adapter supplies 64x64 RGB pixels, camera 0,
action repeat 2, and a 500-decision-step time limit corresponding to 1000
underlying environment steps.

The first gate uses a deliberately small batch (`batch_size=2`,
`batch_length=10`) and exactly 10 real optimizer updates. It records loss and
gradient diagnostics, checks finite values and nonzero parameter changes for
world model, actor, and critic, saves the upstream rlpyt checkpoint, and
loads it in a fresh Python process for fixed-seed evaluation.

The optional bounded baseline uses the same upstream components with a
single training seed. Its `progress.csv` is the learning-curve source and
`eval_episodes.csv` contains random and final Dreamer episodes on seeds
1000--1009. These are pilot measurements, not a paper-level reproduction.

## Compatibility record

The pinned source predates the PyTorch `Transform` domain/codomain contract.
The pilot adds those constraints to `TanhBijector` at runtime when missing.
This is recorded in `compatibility_patches.json`; it does not change the
Dreamer loss or model equations.

## Kaggle entry

Run the script from a notebook cell after the Kaggle GPU session starts:

```python
import subprocess, sys
subprocess.run([
    sys.executable, "scripts/kaggle_pytorch_pilot.py", "--phase", "debug"
], check=True)
```

The code is expected to be invoked in a Kaggle working checkout or from an
equivalent cell that writes the script to `/kaggle/working`. All outputs go to
`/kaggle/working/dreamer_pytorch_pilot/`.
