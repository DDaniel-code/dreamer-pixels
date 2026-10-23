# Dreamer from Pixels — CMU 11-785

Phase 1 project: local preparation, Dreamer V1 reproduction review, and GPU cluster workflow preparation.

## Current status

- We selected the paper authors' original TensorFlow 1 implementation: `google-research/dreamer`, pinned to commit `d517542f9777c53864ff6f2683cea88be9ea98ac`.
- First validation task: DeepMind Control Suite `cartpole_swingup` with pixel observations.
- Later tasks: `walker_walk` and `ball_in_cup catch`.
- This phase does not connect to the university server, submit remote jobs, or run long training.
- GPU training has not been completed, so this project does not claim to reproduce the paper yet.
- The four-page midterm report records the bounded CMU Kaggle validation and its limitations.

## Upstream sources and licensing

If upstream code is reused, this project will preserve the original authorship and Apache-2.0 license. Upstream source should be obtained at the commit recorded in `docs/paper_to_code.md`. New documentation, checks, and configuration in this repository are maintained by this project.

Upstream:

- Paper: https://arxiv.org/abs/1912.01603
- Original implementation: https://github.com/google-research/dreamer
- TensorFlow 2 reference implementation: https://github.com/danijar/dreamer
- Environment: https://github.com/google-deepmind/dm_control

## Local preparation

The local machine is Apple Silicon macOS with Python 3.14.6 available through Conda. The original implementation depends on TensorFlow 1.13.1 and has not been installed or trained locally. Formal runs should use a Linux GPU environment; the CUDA, driver, and Python versions must be confirmed with the university.

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

After the upstream source is available on a Linux GPU node, use the author's training entry point:

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
