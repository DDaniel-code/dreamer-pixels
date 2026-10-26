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
- Completed the bounded single-seed PyTorch baseline on 2026-10-23: 3,000
  decision steps / 6,000 environment steps, 200 real optimizer updates, and
  a persisted final checkpoint. The exact configuration, step accounting,
  parameter changes, evaluation returns, and local download paths are in
  `docs/kaggle_pytorch_baseline_2026-10-23.md`.
- Completed the independent 100,000-environment-step Kaggle baseline on
  2026-10-23 using 50,000 decision steps, one training seed, replay capacity
  60,000 decision transitions, and 4,900 real optimizer updates. The CPU
  package audit, fresh-process dm_control probe, CUDA/EGL preflight, ten-update
  gate, training checkpoint, evaluation CSV, raw logs, and archive were
  downloaded and checked. The detailed record is in
  `docs/kaggle_100k_baseline_2026-10-23.md`; the ignored local artifacts are
  under `artifacts/kaggle_100k_final/`.

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
- The short Dreamer learning path, the 100k single-seed training path, and
  fresh-process checkpoint evaluation are verified; full optimizer/replay
  continuation, paper-level reproduction, and multiple training seeds remain
  unverified.
- The earlier PyTorch route failure remains historical evidence: it used a
  Python 3.13 Kaggle image without `dm-control`, and the ordinary dependency
  install failed while building `labmaze`. The current route avoids that
  unadjusted install and has passed the CPU, CUDA/EGL, and ten-update gates.
- The earlier interrupted 10,000-step attempt remains historical evidence only.
  It is superseded for reporting by the completed 100k-environment-step run,
  whose output archive was downloaded and checked before stopping the Kaggle
  session. The earlier 3,000-decision pilot remains separately documented and
  must not be concatenated with the new run.
- The final Kaggle draft session was stopped after the exported archive was
  downloaded. The archive passed `unzip -t` and contains the final checkpoint,
  raw logs, metrics, evaluation CSV, source checkout, and license.
- Linux GPU dependencies, CUDA, EGL, checkpoint recovery, and formal evaluation.
- The recovery evidence and interrupted-baseline update are pushed to the
  private `origin/codex/bootstrap` branch through commit `b0651d1`.

## Blockers

- The local machine does not have the `gh` CLI; shell DNS access to GitHub is unavailable, so browser-based GitHub operations were used.
- University GPU access, SSH host details, and the scheduling system are not yet confirmed.

## Next steps

1. Replace the midterm report's short-pilot numbers with the 100k run numbers,
   explicitly labeling the result as one training seed and preliminary evidence.
2. Do not combine this independent run with the earlier pilot. If another run
   is approved, use a new seed and the same recorded configuration.
3. Keep the random and formal initial-policy evaluations as controls; do not
   present them as the Dreamer baseline or as a paper-level result.

## 2026-10-24 long-baseline attempt

- Audited the existing `artifacts/kaggle_100k_final/` locally without
  retraining. The saved evaluation CSV independently reproduces the recorded
  random, training-initial, and final means/sample standard deviations. The
  saved formal initial snapshot is documented as the same training model at
  prefill, before the first optimizer update.
- The existing 100k checkpoint contains model and optimizer state, but does
  not contain replay contents, replay sampling state, complete RNG state, or
  environment/recurrent state. It is therefore not a verified full resume;
  the planned 500k run is an independent seed-0 run and must not be
  concatenated with the 100k curve.
- Prepared the bounded long-run runner and notebook:
  `notebooks/kaggle_dreamer_v1_seed0_long_500k_20261024.ipynb`. It uses the
  fixed Julius Frost commit `ccea6ae4a397a94c328891bd78e81d52dd156cb6`, the
  validated PyTorch/dm-control route, `train_every=1000`, `train_steps=100`,
  batch size 16, sequence length 50, replay capacity 60,000 decision
  transitions, action repeat 2, and a 165-minute training cap within the
  four-hour window. It saves separate logs/checkpoints and final evaluation
  videos when a GPU session is available.
- Imported and named the notebook in Kaggle as
  `dreamer-v1-seed0-long-500k`. A private 100k-artifact dataset upload was
  attempted, but Kaggle returned `Failed to execute 'json' on 'Response':
  Unexpected end of JSON input`; no prior artifact was attached to the
  notebook run.
- No new GPU training was started. Kaggle account settings show `Kaggle GPU
  00:00 / 30 hrs`, but the notebook accelerator menu marks `GPU T4 x2`
  disabled and the previous notebook explicitly shows that phone verification
  is required for GPU/notebook features. The temporary CPU session was
  stopped. This is an account access blocker, not evidence about Dreamer
  performance.
- Added `docs/long_baseline_config_2026-10-24.md`, recording the effective
  long-run configuration, paper/upstream differences, `use_pcont=False`, and
  the model-and-optimizer-only resume classification. Local pytest, Python
  compilation, and notebook JSON validation pass; no new training metrics were
  generated.

## 2026-10-24 retry status

- The Kaggle account was verified and a T4 session was started; the earlier
  phone-verification blocker is no longer current.
- The fixed `dm-control --no-deps` route passed again in a fresh subprocess:
  `dm-env==1.6`, `mujoco==3.3.7`, `dm-control==1.0.34`, `glfw==2.11.0`, and
  `pyprind==2.11.3`. The prior 100k zip was not attached, so no old
  checkpoint was re-evaluated in this run.
- The first retry stopped after a real debug update/checkpoint succeeded but
  its evaluation CSV export raised a field-union error when video metadata was
  present only on selected Dreamer rows. The complete log and summary were
  downloaded; this was an export bug, not a failed optimizer update.
- The local exporter now writes the union of row keys. The replacement notebook
  `notebooks/kaggle_dreamer_v1_seed0_long_500k_20261024_retry1.ipynb` passed
  notebook AST validation and pytest.
- The clean retry passed CUDA/EGL preflight and the ten-update gate. The
  session has since stopped; only partial long-run progress and logs were
  recovered. Do not report the run as complete because its checkpoint,
  metrics, evaluation CSV, and export bundle were not recovered.

## 2026-10-24 artifact rescue and midterm update

- GPU remains off and no new training was started.
- Dreamer-related Downloads were organized under
  `/Users/zycy/Downloads/dreamer_experiments/`. The SHA256 manifest is
  `/Users/zycy/Downloads/dreamer_experiments/manifest.csv`.
- The verified 100k ZIP and its gate evidence are under `pilot_100k/`.
  Earlier 3k/6k pilot artifacts are under `pilot_100k/legacy_6k/`. The
  long-run partial CSVs and logs are under `long_run_partial/`. The historical
  TensorFlow notebook is retained under `unresolved/legacy_notebooks/`.
  Exact hash duplicates are under `duplicates/`; no files were deleted or
  overwritten.
- The latest independently verified long-run progress remains 72,000
  decision steps, 144,000 environment steps, and 7,100 updates. This is not
  a final long-run count. No long-run final checkpoint, final progress file,
  evaluation CSV, summary, or export ZIP was recovered.
- The midterm LaTeX report was updated to keep the completed 100k run as the
  main baseline and describe the long run only as partial progress. The old
  100k performance numbers were not changed. Contribution details remain
  pending confirmation of Chevy Yang's exact artifact and role breakdown.
- Before the next long run, validate persistence with a short stop-and-
  download test and an external/session-independent checkpoint destination;
  do not rely only on a final export from `/kaggle/working`.

## 2026-10-25 overnight bounded run

- GPU is closed. The Kaggle notebook was
  `dreamer-v1-seed0-overnight-20261024` under the Daniel Yang
  account, with a separate run ID `seed0_overnight_20261024_01`.
- The persistence probe passed before training: its saved-version bundle was
  downloaded and loaded by a fresh child process. The final 53-file bundle was
  downloaded before stopping the session, passed `unzip -t`, and has SHA256
  `a49517de599fc82568be2067de3d34d08431d00e8f9375837ffb1a6ca857e41b`.
- The fixed PyTorch route used Dreamer commit
  `ccea6ae4a397a94c328891bd78e81d52dd156cb6` and `rlpyt` commit
  `f04f23db1eb7b5915d88401fca67869968a07a37`. CPU dm-control, CUDA/EGL,
  three-module debug, checkpoint, and fresh-process evaluation gates passed.
- The independent formal run returned code 0 at the wall-clock limit with
  165,366 decision steps, 330,732 environment steps, and 16,500 optimizer
  updates. It did not reach the 500,000-environment-step target and must not
  be concatenated with the earlier 100k run or described as another training
  seed.
- The final evaluation returned code 0 on environment seeds 1000--1009:
  random `45.6311 +/- 37.4796`, training-initial `73.7413 +/- 2.3887`, and
  final Dreamer `322.0840 +/- 9.6868`. The initial checkpoint metadata records
  the formal pre-update snapshot (`itr=999`, `updates=0`).
- Local artifacts are under
  `/Users/zycy/Downloads/dreamer_experiments/seed0_overnight_20261024_01/`.
  The extracted key files include both checkpoints, `progress.csv`, raw logs,
  `eval_episodes.csv`, `summary.json`, three fixed-seed videos, and
  `download_manifest.csv`.
- The midterm draft was rebuilt as a five-page PDF with the overnight run
  reported separately from the 100k baseline. The old PDF is preserved as
  `output/pdf/dreamer_pixels_midterm_report_100k_baseline.pdf`. No course
  submission, GitHub push, or new training was performed.
- Remaining verification: local PyTorch is unavailable, so checkpoint loading
  was not repeated with local `torch.load`; it was verified in the fresh Kaggle
  evaluation child process. Full optimizer/replay/RNG/environment continuation
  remains unverified. Seeds 1 and 2 and the two proposal extensions remain
  future work.

## 2026-10-25 seed-0 stage audit / seed-1 gate

- No seed-0 training was restarted. The fixed stage-input files confirm that
  the overnight run contains the 100k, 200k, and 300k decision-boundary
  checkpoints plus `latest_before_shutdown.pkl`.
- The prior final score `322.0840 +/- 9.6868` came from
  `long_train_run/run_0/params.pkl`, with `itr=164999` and
  `cum_steps=164999` (165,000 decision steps in the checkpoint payload), not
  from the later shutdown snapshot. The shutdown snapshot is a separate file
  and must not be silently substituted into the old result.
- A fresh Kaggle stage-audit notebook was started with the same pinned
  PyTorch route and the previously validated dm-control installation. The
  fixed 100k stage evaluation completed in about four minutes on `cuda:0`.
  No verified 200k or 300k stage evaluation output was recovered from this
  session; therefore no stage learning curve and no seed-1 training result is
  reported.
- Seed 1 was not started. The gate requires all stage checks and a measured
  budget check before training; those conditions were not satisfied in the
  bounded session. The session should remain closed after preserving the
  available output/log evidence.
- The runner now records evaluated checkpoint path, SHA256, `itr`, decision
  and environment steps, optimizer-update source, and whether the progress
  file ended at a non-log boundary. A regression test covers the latter case.

## 2026-10-25 stage-evaluation recovery and seed-1 preparation

- The fresh stage-audit Kaggle session was visibly stopped. The UI changed
  from `Running Draft Session` to `Draft Session off (run a cell to start)`
  and showed `Session stopped`; no new training or replacement GPU session
  was started.
- The session produced and the browser downloaded
  `/Users/zycy/Downloads/stage_evaluation_returns.csv`. Its SHA256 is
  `e7e064e8254a2ce45a54cc6d2270ba4ccf447a17da62e7e2629ff117785601af`.
  It has 30 rows: ten Dreamer episodes at each of the 100k, 200k, and 300k
  overnight checkpoints. The independently recomputed 300k result is
  `258.4441 +/- 37.9997` over environment seeds 1000--1009.
- A separate checkpoint-specific `eval_episodes.csv` and `summary.json` for
  300k were not downloaded. The stage aggregate CSV is the actual evidence;
  the ten 300k rows, derived summary, protocol, checkpoint manifest, and
  plots are preserved in `results/seed0_300k/`. No random or training-initial
  control is included in this stage directory.
- The old overnight evaluation value `322.0840 +/- 9.6868` is tied to
  `long_train_run/run_0/params.pkl`, `itr=164999`. Its checkpoint payload
  represents 165,000 decisions / 330,000 environment steps. The run's total
  interaction count is separately 165,366 decisions / 330,732 environment
  steps and 16,500 optimizer updates. These counters must not be conflated.
- The 300k checkpoint is
  `decision_150000_env_300000.pkl`, SHA256
  `5aa0dfd42c2a218d91a16d828b9e9d455abc9f942a88c471a37f1e738f0e2367`.
  Its 150,000-decision / 300,000-environment-step position is confirmed by
  the filename and periodic-save log. The manifest records `itr=149999` as
  the save-boundary convention and explicitly notes that local PyTorch was
  unavailable for payload inspection; Kaggle did load the checkpoint for the
  stage evaluation.
- Targeted local checks passed for Python compilation and the non-log-boundary
  metadata helper. `pytest` was not available in either tested local Python
  environment, so no pytest suite is claimed as passed. Seed 1 remains
  unstarted.

## 2026-10-25 revised three-seed 100k plan

- The requested comparison is now seed 0 / seed 1 / seed 2 at 100,000
  environment steps each. The 300k/330k seed-0 overnight run remains a
  separate long-budget result.
- No new GPU session or training was started in this revision. The current
  Kaggle notebook was the older seed-1 matching notebook and its session was
  stopped. Local and remote repository checks found no teammate seed-1
  summary, actual configuration, evaluation CSV, or checkpoint metadata.
- A gated seed-2 notebook was prepared at
  `notebooks/kaggle_dreamer_v1_seed2_100k_20261025.ipynb`. It stops before
  training unless real seed-1 evidence passes checks against the completed
  seed-0 100k configuration. Details are in
  `docs/seed2_100k_gate_2026-10-25.md`.

## 2026-10-25 seed-1 provenance update

- The teammate reports that seed 1 used the repository's completed 100k
  notebook with only the training seed changed to `1`.
- The supplied aggregate summary is consistent with `cartpole_swingup` and
  ten evaluation seeds, but raw seed-1 config, per-episode CSV, and
  checkpoint files were not supplied. Treat seed 1 as a provisional match,
  not independently verified configuration evidence.
- The seed-2 notebook was regenerated to record this caveat and to use the
  repository seed-0 notebook as the fixed configuration source. It also fixes
  an undefined `ROOT` variable that would have stopped the notebook before
  training.

## 2026-10-25 seed-2 100k run

- The independent Kaggle run `seed2_matching_20261025_01` completed on
  `cartpole_swingup` with seed 2, 50,000 decision steps, and 100,000
  environment steps. The training subprocess completed in about 69 minutes;
  evaluation and minimal export also returned successfully.
- The notebook's final output identifies
  `train_run/run_0/params.pkl` as the final checkpoint and
  `initial_policy.pkl` as the pre-update checkpoint. The Kaggle output was
  saved as Quick Save `Version #1`, and the GPU session was explicitly
  stopped after completion.
- The Version 1 Output page was checked after the run and reported `0 files`;
  the editor showed `Draft Session off`. The executed notebook was downloaded
  and preserved at `results/seed2_100k/kaggle_notebook_executed.ipynb` with
  SHA256 `2461cadc2c4b836ae9045435818ff8069f676d1f09467c9eda24036d782e25c4`.
  Its recorded wrapper output reports training/evaluation/minimal-export
  completion, but no raw seed-2 CSV, summary, checkpoint, progress, complete
  log, or ZIP was available through Kaggle Output Data. Seed 2 therefore
  remains execution-record-only and must not be included in a verified
  cross-seed statistic. Do not treat the seed-1 aggregate values embedded in
  the notebook's provenance gate as seed-2 results.
