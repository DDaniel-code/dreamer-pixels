#!/usr/bin/env python3
"""Build the clean Kaggle notebook for the bounded long seed-0 run."""

from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
runner_b64 = base64.b64encode((ROOT / "scripts" / "kaggle_pytorch_pilot.py").read_bytes()).decode("ascii")


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


cells = [
    markdown("""# Dreamer V1 — bounded long seed-0 baseline

This notebook uses Julius Frost's pinned Dreamer V1 PyTorch source and the
validated Kaggle PyTorch/dm-control route. It first audits the existing 100k
artifacts when they are attached as an input, then performs a separate
single-seed run. It never concatenates the old 100k curve with the new run.

The target is 500,000 environment steps (250,000 decision steps), but the
training child has a 165-minute wall-clock cap so that final evaluation,
checkpoint export, and download remain possible inside the four-hour window.
The actual step count is reported from the saved progress/checkpoint state.
"""),
    code("""import base64
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

root = Path('/kaggle/working/dreamer_pytorch_long_seed0_20261024_retry1')
root.mkdir(parents=True, exist_ok=True)
runner = root / 'kaggle_pytorch_pilot.py'
runner.write_bytes(base64.b64decode('""" + runner_b64 + """'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MUJOCO_GL'] = 'egl'
print('runner:', runner)
print('root:', root)
"""),
    markdown("""## CPU/environment audit

This child process is run before the GPU preflight. It records the actual
interpreter and package visibility, then imports dm-control, loads
cartpole_swingup, resets, and executes ten random actions.
"""),
    code("""import importlib.util

install_log = root / 'dm_control_no_deps_install.log'
if importlib.util.find_spec('dm_control') is None:
    install_command = [
        sys.executable, '-m', 'pip', 'install', '--no-deps',
        'dm-env==1.6',
        'mujoco==3.3.7',
        'dm-control==1.0.34',
        'glfw==2.11.0',
        'pyprind==2.11.3',
    ]
    print('Installing the previously validated dm-control --no-deps route:', install_command)
    install = subprocess.run(install_command, capture_output=True, text=True)
    install_log.write_text(
        'COMMAND: ' + ' '.join(install_command) + '\\n\\n'
        + 'RETURN: ' + str(install.returncode) + '\\n\\n'
        + 'STDOUT:\\n' + install.stdout + '\\n\\n'
        + 'STDERR:\\n' + install.stderr,
        encoding='utf-8',
    )
    print(install_log.read_text())
    install.check_returncode()
else:
    install_log.write_text(
        'dm_control was already visible to ' + sys.executable
        + '; the fixed no-deps installation was not repeated.\\n',
        encoding='utf-8',
    )
    print(install_log.read_text())

probe = root / 'cpu_dm_control_probe.py'
probe.write_text('''import importlib.util
import sys
import numpy as np
print('sys.executable:', sys.executable)
print('sys.path:', sys.path)
print('dm_control spec:', importlib.util.find_spec('dm_control'))
from dm_control import suite
env = suite.load('cartpole', 'swingup', task_kwargs={'random': 0})
try:
    ts = env.reset()
    spec = env.action_spec()
    print('reset:', ts.reward, ts.last(), 'action_shape:', spec.shape)
    for i in range(10):
        action = np.random.uniform(spec.minimum, spec.maximum).astype(spec.dtype)
        ts = env.step(action)
        print('step', i + 1, 'reward:', ts.reward, 'last:', ts.last())
finally:
    env.close()
print('CPU dm_control reset/step: PASS')
''', encoding='utf-8')
audit = root / 'cpu_package_audit.txt'
audit_cmds = []
for label, command in [('python', [sys.executable, '-c', 'import sys; print(sys.executable); print(sys.version)']),
                       ('pip', [sys.executable, '-m', 'pip', '--version']),
                       ('dm-control', [sys.executable, '-m', 'pip', 'show', '-f', 'dm-control'])]:
    res = subprocess.run(command, capture_output=True, text=True)
    audit_cmds.append(label + '\\n' + res.stdout + res.stderr)
audit.write_text('\\n\\n'.join(audit_cmds), encoding='utf-8')
res = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True)
(root / 'cpu_dm_control_probe.log').write_text('RETURN: ' + str(res.returncode) + '\\n\\nSTDOUT:\\n' + res.stdout + '\\nSTDERR:\\n' + res.stderr, encoding='utf-8')
print((root / 'cpu_dm_control_probe.log').read_text())
res.check_returncode()
"""),
    markdown("""## Existing 100k artifact audit and corrected re-evaluation

Attach the previously downloaded `dreamer_pilot_artifacts_100k_final.zip` as
a Kaggle input before running this cell if it is available. If it is not
attached, the new run continues independently and records that the prior
checkpoint re-evaluation was skipped for lack of a transferable checkpoint.
"""),
    code("""def find_prior_zip():
    candidates = list(Path('/kaggle/input').rglob('dreamer_pilot_artifacts_100k_final.zip'))
    candidates += list(Path('/kaggle/input').rglob('*.zip'))
    return candidates[0] if candidates else None

prior_zip = find_prior_zip()
prior_root = root / 'prior_100k'
prior_root.mkdir(parents=True, exist_ok=True)
prior_final = prior_root / 'train_run' / 'run_0' / 'params.pkl'
prior_initial = prior_root / 'initial_policy.pkl'
if prior_zip:
    with zipfile.ZipFile(prior_zip) as archive:
        archive.extractall(prior_root)
    nested = list(prior_root.rglob('train_run/run_0/params.pkl'))
    if nested:
        source_root = nested[0].parents[2]
        prior_root = source_root
        prior_final = source_root / 'train_run' / 'run_0' / 'params.pkl'
        prior_initial = source_root / 'initial_policy.pkl'
print('prior_zip:', prior_zip)
print('prior_final:', prior_final, prior_final.exists())
print('prior_initial:', prior_initial, prior_initial.exists())
"""),
    code("""def run_child(phase, *args, timeout=None, child_root=root):
    log_path = child_root / f'{phase}_subprocess.log'
    env = os.environ.copy()
    env.update({'DREAMER_PILOT_ROOT': str(child_root), 'CUDA_VISIBLE_DEVICES': '0', 'MUJOCO_GL': 'egl'})
    # The runner is written once at the shared root; each phase isolates outputs via DREAMER_PILOT_ROOT.
    command = [sys.executable, str(runner), '--phase', phase, *map(str, args)]
    started = time.time()
    with log_path.open('w', encoding='utf-8') as stream:
        proc = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True)
        while proc.poll() is None:
            elapsed = time.time() - started
            print(f'[{phase}] elapsed={elapsed/60:.1f} min log_bytes={log_path.stat().st_size if log_path.exists() else 0}', flush=True)
            if timeout is not None and elapsed > timeout:
                proc.terminate()
                raise TimeoutError(f'{phase} exceeded {timeout}s')
            time.sleep(60)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, command)
    print(f'[{phase}] complete in {(time.time()-started)/60:.1f} min')

if prior_final.exists() and prior_initial.exists():
    run_child('resume_audit', '--checkpoint', prior_final, '--initial-checkpoint', prior_initial,
              child_root=root / 'prior_audit')
    run_child('eval', '--checkpoint', prior_final, '--initial-checkpoint', prior_initial,
              child_root=root / 'prior_reeval')
else:
    (root / 'prior_reeval_skipped.txt').write_text('Prior 100k checkpoint zip was not attached to this Kaggle run.\\n', encoding='utf-8')
    print((root / 'prior_reeval_skipped.txt').read_text())
"""),
    markdown("""## GPU preflight and ten-update gate"""),
    code("""run_child('preflight', timeout=20*60)
run_child('debug', timeout=20*60)
print('GPU/EGL preflight and Dreamer update gate passed.')
"""),
    markdown("""## Independent long-budget seed-0 training

Configuration remains the validated 100k baseline: prefill 1,000 decision
steps, replay capacity 60,000 decision transitions, `train_every=1000`,
`train_steps=100`, batch size 16, sequence length 50, action repeat 2, and
imagination horizon 15. The runner saves checkpoints at 100k, 200k, 300k,
and 500k environment-step targets when those decision steps are reached.
"""),
    code("""run_child('train_long', '--n-steps', '250000', '--max-train-minutes', '165', timeout=190*60)
run_child('export', timeout=10*60)
print('Long run/export complete; inspect train_summary.json and summary.json for actual counts.')
"""),
    markdown("""## Output

All new outputs are under `/kaggle/working/dreamer_pytorch_long_seed0_20261024_retry1/`.
The notebook must be saved as a Kaggle version, and the final zip, checkpoint,
logs, CSV files, videos, and summaries must be downloaded before the session
is stopped. A missing 500k target is reported as the actual achieved count.
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

out = ROOT / "notebooks" / "kaggle_dreamer_v1_seed0_long_500k_20261024_retry1.ipynb"
out.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(out)
