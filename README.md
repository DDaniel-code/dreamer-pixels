# Dreamer from Pixels — CMU 11-785

Phase 1 project: local preparation, Dreamer V1 reproduction review, and GPU cluster workflow preparation.

## Current status

- The active Kaggle route is the maintained PyTorch Dreamer V1 implementation at `juliusfrost/dreamer-pytorch@ccea6ae4a397a94c328891bd78e81d52dd156cb6`, with its pinned `rlpyt` submodule at `f04f23db1eb7b5915d88401fca67869968a07a37`. The upstream repository is in maintenance mode and is MIT-licensed; its `LICENSE` file is retained in the Kaggle checkout.
- The earlier TensorFlow 2 route at `danijar/dreamer@56d4d444dfd0582b0e79dab80aebbea74c0ce40d` is retained as historical investigation only. Its Kaggle recovery stopped because the isolated TensorFlow process could not enumerate the system-visible T4 GPUs; it is no longer the execution route.
- First validation task: DeepMind Control Suite `cartpole_swingup` with pixel observations.
- Later tasks: `walker_walk` and `ball_in_cup catch`.
- This phase does not connect to the university server, submit remote jobs, or run long training.
- The active PyTorch route has a bounded cartpole_swingup gate before any baseline training: pixel/reset/action-repeat checks, ten real updates, parameter-change checks, checkpoint save, and fresh-process evaluation. GPU training and paper reproduction are not claimed until those checks produce actual outputs.
- The four-page midterm report records the bounded CMU Kaggle validation and its limitations.

## Upstream sources and licensing

If upstream code is reused, this project will preserve the original authorship and Apache-2.0 license. Upstream source should be obtained at the commit recorded in `docs/paper_to_code.md`. New documentation, checks, and configuration in this repository are maintained by this project.

Upstream:

- Paper: https://arxiv.org/abs/1912.01603
- Original implementation: https://github.com/google-research/dreamer
- TensorFlow 2 reference implementation: https://github.com/danijar/dreamer
- Environment: https://github.com/google-deepmind/dm_control

## Local preparation

The local machine is Apple Silicon macOS with Python 3.14.6 available through Conda. The historical TensorFlow 1 implementation depends on TensorFlow 1.13.1 and is not the prerequisite for today's pilot. The Kaggle run uses the author's TensorFlow 2 implementation with a small, recorded compatibility patch for current Keras/TensorFlow APIs. Formal runs should use a Linux GPU environment; the CUDA, driver, and Python versions must be recorded.

Run the dependency-free environment check first:

```bash
python3 scripts/env_check.py
python3 -m pytest -q
```

The project-local probe environment can be created with:

```bash
/opt/anaconda3/bin/python3 -m venv .venv
.venv/bin/python -m pip install dm-control imageio pillow numpy pyyaml
```

The probe entry point is intentionally short and uses a random policy:

```bash
.venv/bin/python scripts/random_rollout.py --max-env-steps 200
```

This command is an observation/rendering check only. On macOS, MuJoCo pixel
rendering may require a working graphical backend; the current machine has not
passed that check. The original Dreamer V1 training environment remains a
Linux/GPU target and is not replaced by this modern environment probe.

The active Kaggle pilot retrieves the pinned PyTorch source and its `rlpyt` submodule, then uses the upstream `Dreamer` algorithm, agent, replay buffer, RSSM, and DMC wrappers through `scripts/kaggle_pytorch_pilot.py`. It does not reinstall PyTorch or rewrite the algorithm. A small runtime shim records the `TanhBijector` domain/codomain compatibility needed by newer PyTorch distributions. The TensorFlow routes remain only as recorded historical evidence.

For a future Linux GPU preflight, use the author's training entry point:

```bash
python3 -m dreamer.scripts.train --logdir ./logdir/cartpole_swingup \
  --params '{defaults: [dreamer], tasks: [cartpole_swingup]}'
```

This command has not yet been verified locally or on the cluster and should not be used to submit a long-running job directly.

## Documentation

- [Reproduction protocol](docs/reproduction_protocol.md)
- [Implementation differences](docs/implementation_differences.md)
- [Paper-to-code map](docs/paper_to_code.md)
- [Cluster setup checklist](docs/cluster_setup.md)
- [Midterm report](docs/midterm_report.md)
- [Midterm report PDF](output/pdf/dreamer_pixels_midterm_report.pdf)
- [Handoff record](docs/HANDOFF.md)
