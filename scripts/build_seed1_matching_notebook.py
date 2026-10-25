#!/usr/bin/env python3
"""Build the bounded stage-evaluation and matching seed-1 Kaggle notebook."""

from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_B64 = base64.b64encode(
    (ROOT / "scripts" / "kaggle_pytorch_pilot.py").read_bytes()
).decode("ascii")


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(True),
    }


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


cells = [
    markdown("""# Dreamer V1 — seed-0 checkpoint audit and matching seed-1 run

This notebook first evaluates only the seed-0 overnight checkpoints that are
actually supplied as an input dataset. It then runs one independent seed-1
experiment with the same effective configuration. The old 100k run and the
seed-0 curve are never concatenated with this run.

The seed-1 target is 150,000 decision steps (300,000 environment steps). A
four-hour wall-clock guard and a final-evaluation reserve take priority. If
the measured throughput cannot fit the target and the required evaluation and
download reserve, the notebook records a clean skip instead of starting an
under-budget run.
"""),
    code("""import base64
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

SESSION_STARTED = time.time()
SEED1_RUN_ID = 'seed1_matching_20261025_01'
SEED1_ROOT = Path('/kaggle/working') / f'dreamer_{SEED1_RUN_ID}'
SEED1_ROOT.mkdir(parents=True, exist_ok=True)
RUNNER = SEED1_ROOT / 'kaggle_pytorch_pilot.py'
RUNNER.write_bytes(base64.b64decode('""" + RUNNER_B64 + """'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['DREAMER_RUN_ID'] = SEED1_RUN_ID
os.environ['DREAMER_PILOT_ROOT'] = str(SEED1_ROOT)
print('run_id:', SEED1_RUN_ID)
print('root:', SEED1_ROOT)
print('python:', sys.executable)
"""),
    markdown("""## Input and checkpoint audit

Attach the seed-0 stage input as a Kaggle dataset/input before running this
notebook. Kaggle may expose an uploaded ZIP as an already-unpacked directory;
this cell supports both forms and requires the 100k, 200k, and 300k checkpoints.
"""),
code("""input_root = Path('/kaggle/input')
input_files = list(input_root.rglob('seed0_stage_eval_input_with_progress.zip'))
if not input_files:
    input_files = list(input_root.rglob('*.zip'))
if input_files:
    stage_input = SEED1_ROOT / 'seed0_stage_input'
    stage_input.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_files[0]) as archive:
        archive.extractall(stage_input)
    input_description = str(input_files[0])
else:
    # Kaggle commonly unpacks an uploaded ZIP into the mounted dataset tree.
    stage_input = input_root
    input_description = str(input_root)
stage_checkpoints = {}
for item in stage_input.rglob('decision_*_env_*.pkl'):
    match = re.search(r'env_(\\d+)\\.pkl$', item.name)
    if match:
        stage_checkpoints[int(match.group(1))] = item
required_env_steps = [100000, 200000, 300000]
missing = [step for step in required_env_steps if step not in stage_checkpoints]
if missing:
    raise FileNotFoundError(f'Missing stage checkpoints: {missing}')
progress_candidates = list(stage_input.rglob('progress.csv'))
seed0_progress = progress_candidates[0] if progress_candidates else None
print('input:', input_description)
print('stage checkpoints:', {k: str(v) for k, v in sorted(stage_checkpoints.items())})
print('seed0 progress:', seed0_progress)
"""),
    code("""def run_child(phase, *args, child_root=SEED1_ROOT, timeout=None):
    child_root.mkdir(parents=True, exist_ok=True)
    log_path = child_root / f'{phase}_subprocess.log'
    env = os.environ.copy()
    env.update({
        'DREAMER_RUN_ID': SEED1_RUN_ID,
        'DREAMER_PILOT_ROOT': str(child_root),
        'CUDA_VISIBLE_DEVICES': '0',
        'MUJOCO_GL': 'egl',
    })
    command = [sys.executable, str(RUNNER), '--phase', phase, *map(str, args)]
    started = time.time()
    with log_path.open('w', encoding='utf-8') as stream:
        proc = subprocess.Popen(command, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, text=True)
        while proc.poll() is None:
            elapsed = time.time() - started
            print(f'[{phase}] elapsed={elapsed/60:.1f} min log_bytes={log_path.stat().st_size if log_path.exists() else 0}', flush=True)
            if timeout is not None and elapsed > timeout:
                proc.terminate()
                raise TimeoutError(f'{phase} exceeded {timeout}s')
            time.sleep(30)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, command)
    print(f'[{phase}] complete in {(time.time()-started)/60:.1f} min')
"""),
    markdown("""## Existing seed-0 intermediate evaluation

Each checkpoint is evaluated with the same ten environment seeds and episode
contract. Random and initial controls are skipped here so the time is spent
on the three actual seed-0 stage policies. The output is one directory per
checkpoint and includes the checkpoint path, hash, payload counters, and the
exact progress-row update count when available.
"""),
    code("""stage_records = []
for env_steps in required_env_steps:
    checkpoint = stage_checkpoints[env_steps]
    out_dir = SEED1_ROOT / 'stage_evaluations' / f'env_{env_steps:06d}'
    run_child(
        'eval', '--checkpoint', checkpoint, '--output-dir', out_dir,
        '--progress-path', seed0_progress, '--skip-random', '--skip-initial',
        '--no-video', child_root=SEED1_ROOT / f'stage_eval_{env_steps:06d}',
        timeout=15*60,
    )
    summary = json.loads((out_dir / 'summary.json').read_text())
    for row in csv.DictReader((out_dir / 'eval_episodes.csv').open()):
        row['checkpoint_env_steps'] = env_steps
        stage_records.append(row)
    print(json.dumps(summary, indent=2, sort_keys=True))
with (SEED1_ROOT / 'stage_evaluation_returns.csv').open('w', newline='') as handle:
    fields = ['checkpoint_env_steps', 'policy', 'seed', 'return', 'decision_steps', 'env_steps']
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(stage_records)
print('stage evaluation rows:', len(stage_records))
"""),
    markdown("""## Persistence, CPU audit, and GPU/debug gates

The persistence probe is repeated in this independent run before training.
The fixed `dm-control --no-deps` route is reused only if the package is not
visible. The runner then checks CUDA/EGL and ten real Dreamer updates.
"""),
    code("""run_child('persistence_probe', timeout=5*60)
run_child('preflight', timeout=15*60)
run_child('debug', timeout=15*60)
"""),
    markdown("""## Time guard before seed-1 training

The previous completed run measured its formal training throughput from its
saved counters and wall time. This guard includes the elapsed stage work and
reserves 45 minutes for final evaluation, export, and download. It writes a
skip record if 150,000 decision steps cannot fit; it never starts a knowingly
under-budget run.
"""),
    code("""seed0_decisions = 165366
seed0_training_seconds = 12601.907940864563
decisions_per_second = seed0_decisions / seed0_training_seconds
target_decisions = 150000
estimated_training_seconds = target_decisions / decisions_per_second
reserve_seconds = 45 * 60
elapsed_seconds = time.time() - SESSION_STARTED
remaining_seconds = 4 * 60 * 60 - elapsed_seconds
guard = {
    'seed0_measured_decisions_per_second': decisions_per_second,
    'target_decision_steps': target_decisions,
    'estimated_training_seconds': estimated_training_seconds,
    'elapsed_before_training_seconds': elapsed_seconds,
    'remaining_session_seconds': remaining_seconds,
    'required_reserve_seconds': reserve_seconds,
    'estimated_total_with_reserve_seconds': estimated_training_seconds + reserve_seconds,
}
can_start = estimated_training_seconds + reserve_seconds <= remaining_seconds
guard['can_start'] = can_start
(SEED1_ROOT / 'seed1_time_guard.json').write_text(json.dumps(guard, indent=2) + '\n')
print(json.dumps(guard, indent=2, sort_keys=True))
if not can_start:
    (SEED1_ROOT / 'seed1_skipped_due_budget.txt').write_text(
        'The measured seed-0 throughput could not fit the 300k-env-step seed-1 target plus the required 45-minute reserve.\n'
    )
    print('SEED-1 TRAINING SKIPPED: insufficient bounded window.')
else:
    # Allow a small scheduling margin above the measured estimate while keeping
    # the four-hour session bound and the 45-minute finalization reserve.
    training_minutes = min(192, max(1, int(estimated_training_seconds / 60) + 2))
    guard['training_max_minutes'] = training_minutes
    (SEED1_ROOT / 'seed1_time_guard.json').write_text(json.dumps(guard, indent=2) + '\n')
    run_child('train_long', '--seed', '1', '--n-steps', str(target_decisions),
              '--max-train-minutes', str(training_minutes),
              timeout=(training_minutes + 5) * 60)
    checkpoint = SEED1_ROOT / 'long_train_run' / 'run_0' / 'latest_before_shutdown.pkl'
    if not checkpoint.exists():
        checkpoint = SEED1_ROOT / 'long_train_run' / 'run_0' / 'params.pkl'
    initial = SEED1_ROOT / 'initial_policy.pkl'
    run_child('eval', '--seed', '1', '--checkpoint', checkpoint,
              '--initial-checkpoint', initial, timeout=25*60)
    run_child('export_minimal', timeout=10*60)
"""),
    markdown("""## Output and shutdown contract

The seed-1 output root is `/kaggle/working/dreamer_seed1_matching_20261025_01/`.
The stage evaluation CSV and summaries remain separate from seed-1 training.
Download and hash the final bundle, checkpoints, logs, and stage-evaluation
files before stopping the session. Report a skipped or partial seed-1 run
with its actual counters; never call it a matched 300k comparison unless it
reaches 300,000 environment steps.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "notebooks" / "kaggle_dreamer_v1_seed1_matching_20261025.ipynb"
out.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(out)
