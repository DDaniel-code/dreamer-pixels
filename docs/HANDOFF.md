# Handoff

## Completed

- Preserved the earlier Python 3.13/Keras 3 failure evidence in
  `docs/kaggle_legacy_session_2026-10-23.md` and stopped that session.
- Built the bounded TF 2.15 recovery notebook and fixed the isolated
  Python 3.10 environment bootstrap to use seeded `uv` when Kaggle's
  `/usr/local/bin/mamba` is not a compatible CLI.
- Installed the requested isolated stack and reached the pinned author source
  checkout at `56d4d444dfd0582b0e79dab80aebbea74c0ce40d`.
- Applied the one directly evidenced TF 2.15 mixed-precision API mapping and
  recorded the recovery evidence in
  `docs/kaggle_tf215_recovery_2026-10-23.md`.
- Stopped the new Kaggle session after the decisive GPU preflight failure.

- Switched the active route to the fixed PyTorch Dreamer V1 source at
  `juliusfrost/dreamer-pytorch`, commit
  `ccea6ae4a397a94c328891bd78e81d52dd156cb6`, with `rlpyt` submodule
  `f04f23db1eb7b5915d88401fca67869968a07a37`.
- Imported and started the self-contained Kaggle notebook
  `kaggle_dreamer_v1_pytorch_cartpole_pilot.ipynb` on the existing T4 GPU
  environment. The upstream repository and submodule cloned successfully.
- Preserved the failed PyTorch gate output and the complete `dm-control`
  installation log in the Kaggle working directory.
- Reused the previously successful `dm-control` installation path on Kaggle:
  `dm-control==1.0.34 --no-deps`, with the separately evidenced runtime wheels
  `dm-env==1.6`, `mujoco==3.3.7`, and `glfw==2.11.0`.
- Passed the CPU control-path check: `dm_control` imported, `suite.load
  ("cartpole", "swingup")` succeeded, reset succeeded, and ten random actions
  executed with the recorded interpreter and package versions.
- Passed the PyTorch CUDA/EGL check on one visible Tesla T4: matrix gradients
  ran on `cuda:0`, and a 64x64 RGB MuJoCo EGL frame was produced.
- Passed the Dreamer V1 ten-update gate with real finite updates, changed
  world-model/actor/critic parameters, a saved checkpoint, and fresh-process
  evaluation. The debug checkpoint is at
  `/kaggle/working/dreamer_pytorch_pilot/debug_run/run_0/params.pkl`.

- Created the local project directory and Git ignore rules.
- Reviewed the paper, the original author implementation, the TensorFlow 2 reference implementation, and dm_control.
- Pinned the original Dreamer V1 upstream commit: `d517542f9777c53864ff6f2683cea88be9ea98ac`.
- Fixed `cartpole_swingup` as the first task and recorded the pixel, action-repeat, and step-count conventions.
- Added a dependency-free environment check and project metadata tests.
- Added a bounded random-policy pixel rollout entry point at
  `scripts/random_rollout.py` and a midterm report draft.
- Completed two bounded Kaggle `cartpole_swingup` pixel rollouts on the
  CMU-verified account using a Tesla T4 and headless EGL rendering.
- Saved Kaggle draft notebook: https://www.kaggle.com/code/danielyang3/notebook7962593bae/edit
- Expanded the midterm report with the measured results, failure analysis, and
  the distinction between infrastructure validation and learned performance.

## Verified

- Host: Apple Silicon macOS, with Python 3.14.6 available through Conda.
- Project-document and no-fabricated-results tests passed (`/opt/anaconda3/bin/python3 -m pytest -q`).
- The environment inventory ran with `/opt/anaconda3/bin/python3`; the default `/usr/bin/python3` is Python 3.9.6.
- TensorFlow, TensorFlow Probability, dm_control, and dm_env are not installed; pixel environments and training entry points were therefore not run.
- A project-local `.venv` was created and `dm-control` plus its non-maze
  dependencies were installed. The macOS pixel probe did not pass: OSMesa is
  unavailable, and the default graphical initialization did not complete within
  the bounded check. This is a local renderer limitation, not a training result.

## Not verified

- Compatibility of TensorFlow 1.13.1 and TensorFlow Probability 0.6.0 with the current macOS/Python setup.
- The distinction between default `python3` and Conda Python has not yet been unified with a project environment file.
- dm_control/MuJoCo pixel rendering, a short random-policy video, and all training results.
- A local random-policy video/frame artifact; the script is ready, but the
  renderer must be verified on a supported backend.
- The earlier bounded random-rollout evidence is retained separately; it is
  an environment/pixel smoke test, not a Dreamer baseline.
- In the new recovery notebook, `nvidia-smi` reported two Tesla T4 devices,
  but TensorFlow 2.15.1 in the isolated Python 3.10 subprocess returned an
  empty GPU list. The recovery stopped before any model update.
- Dreamer learning, checkpoint restore, and paper-level reproduction remain
  unverified. The current recovery produced zero real updates and no
  checkpoint.
- The earlier PyTorch route failure remains historical evidence: it used a
  Python 3.13 Kaggle image without `dm-control`, and the ordinary dependency
  install failed while building `labmaze`. The current route avoids that
  unadjusted install and has passed the CPU, CUDA/EGL, and ten-update gates.
- The bounded single-seed baseline was started after the gate passed. Its final
  `train_summary.json`, learning curve, and final evaluation must be checked
  before reporting the baseline as complete.
- Linux GPU dependencies, CUDA, EGL, checkpoint recovery, and formal evaluation.
- The repository contains the local recovery commits, but shell DNS/network
  access prevented verifying a push from this environment. Do not claim the
  latest local commits are on GitHub until the remote branch is checked.

## Blockers

- The local machine does not have the `gh` CLI; shell DNS access to GitHub is unavailable, so browser-based GitHub operations were used.
- University GPU access, SSH host details, and the scheduling system are not yet confirmed.

## Next steps

1. Verify the bounded baseline final artifacts and distinguish them from the
   ten-update debug evidence.
2. Save the completed Kaggle notebook version and preserve the raw output under
   `/kaggle/working/dreamer_pytorch_pilot/`; do not put checkpoints or videos in
   Git.
3. Update the midterm report with measured curves and returns only after the
   final artifacts are confirmed. The single seed and short budget remain an
   initial pilot, not a full paper reproduction.
