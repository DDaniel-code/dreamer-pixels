#!/usr/bin/env python3
"""Build the self-contained Kaggle notebook for the PyTorch pilot."""

from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_B64 = base64.b64encode((ROOT / "scripts" / "kaggle_pytorch_pilot.py").read_bytes()).decode("ascii")


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


cells = [
    markdown("""# Dreamer V1 PyTorch Cartpole Swingup — 100k environment-step run

This notebook reuses Kaggle's existing PyTorch environment and the fixed
`juliusfrost/dreamer-pytorch` commit `ccea6ae4a397a94c328891bd78e81d52dd156cb6`
with `rlpyt` commit `f04f23db1eb7b5915d88401fca67869968a07a37`. It does not
install or replace PyTorch. The run is an independent single-seed pilot:
`--n-steps 50000` means 50,000 decision steps and 100,000 underlying environment
steps because action repeat is 2.

Run the CPU audit first. It checks the actual child interpreter, package
visibility, and a fresh-process `dm_control` reset/step. Only after that passes
should the session be switched to one GPU for the CUDA/EGL preflight and the
ten-update Dreamer gate. The formal run creates its initial checkpoint from the
same training process immediately before the first optimizer update after
prefill; there is no separate randomly initialized training-before run.
"""),
    code("""import base64
import os
from pathlib import Path

root = Path('/kaggle/working/dreamer_pytorch_100k_env_run')
root.mkdir(parents=True, exist_ok=True)
runner = root / 'kaggle_pytorch_pilot.py'
runner.write_bytes(base64.b64decode('""" + RUNNER_B64 + """'))
print('runner:', runner)
print('runner bytes:', runner.stat().st_size)
"""),
    code("""import importlib.util
import platform
import subprocess
import sys

audit_lines = []
audit_lines.append('sys.executable: ' + sys.executable)
audit_lines.append('sys.version: ' + sys.version)
audit_lines.append('platform: ' + platform.platform())
audit_lines.append('sys.path:')
audit_lines.extend('  ' + item for item in sys.path)
for label, command in [
    ('pip --version', [sys.executable, '-m', 'pip', '--version']),
    ('pip show -f dm-control', [sys.executable, '-m', 'pip', 'show', '-f', 'dm-control']),
]:
    result = subprocess.run(command, capture_output=True, text=True)
    audit_lines.append('\\n$ ' + ' '.join(command) + '\\n' + result.stdout + result.stderr)
spec = importlib.util.find_spec('dm_control')
audit_lines.append('find_spec(dm_control): ' + repr(spec))
audit_path = root / 'cpu_package_audit.txt'
audit_path.write_text('\\n'.join(audit_lines) + '\\n', encoding='utf-8')
print(audit_path.read_text(encoding='utf-8'))

if spec is None:
    command = [sys.executable, '-m', 'pip', 'install', '--no-deps',
               'dm-env==1.6', 'mujoco==3.3.7', 'dm-control==1.0.34',
               'glfw==2.11.0', 'pyprind==2.11.3']
    result = subprocess.run(command, capture_output=True, text=True)
    install_log = 'COMMAND: ' + ' '.join(command) + '\\n\\nSTDOUT:\\n' + result.stdout + '\\nSTDERR:\\n' + result.stderr
    (root / 'dm_control_no_deps_install.log').write_text(install_log, encoding='utf-8')
    print(install_log)
    result.check_returncode()
else:
    print('dm_control is already visible; fixed no-deps install not repeated.')

probe = root / 'cpu_dm_control_probe.py'
probe.write_text('''import importlib.util
import sys
import numpy as np
print('child sys.executable:', sys.executable)
print('child sys.path:', sys.path)
print('child dm_control spec:', importlib.util.find_spec('dm_control'))
from dm_control import suite
env = suite.load('cartpole', 'swingup', task_kwargs={'random': 0})
try:
    timestep = env.reset()
    print('reset reward:', timestep.reward, 'last:', timestep.last())
    spec = env.action_spec()
    for index in range(10):
        action = np.random.uniform(spec.minimum, spec.maximum).astype(spec.dtype)
        timestep = env.step(action)
        print('step', index + 1, 'reward:', timestep.reward, 'last:', timestep.last())
finally:
    env.close()
print('CPU dm_control reset/step: PASS')
''', encoding='utf-8')
probe_result = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True)
probe_log = 'RETURN: ' + str(probe_result.returncode) + '\\n\\nSTDOUT:\\n' + probe_result.stdout + '\\nSTDERR:\\n' + probe_result.stderr
(root / 'cpu_dm_control_probe.log').write_text(probe_log, encoding='utf-8')
print(probe_log)
probe_result.check_returncode()
"""),
    markdown("""## GPU preflight and ten-update gate

After the CPU cell passes, switch this Kaggle notebook to one GPU. This cell
sets `CUDA_VISIBLE_DEVICES=0` before importing torch, verifies CUDA and EGL
pixels, and runs the real ten-update gate. Do not start training if it fails.
"""),
    code("""import os
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MUJOCO_GL'] = 'egl'

def run_pilot(phase, *args, timeout=None):
    log_path = root / f'{phase}_subprocess.log'
    command = [sys.executable, str(runner), '--phase', phase, *args]
    started = time.time()
    child_env = os.environ.copy()
    child_env['DREAMER_PILOT_ROOT'] = str(root)
    child_env['CUDA_VISIBLE_DEVICES'] = '0'
    child_env['MUJOCO_GL'] = 'egl'
    with log_path.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, env=child_env, stdout=stream,
                                   stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            elapsed = time.time() - started
            size = log_path.stat().st_size if log_path.exists() else 0
            print(f'[{phase}] child running: {elapsed:.0f}s, log_bytes={size}', flush=True)
            if timeout is not None and elapsed > timeout:
                process.terminate()
                raise TimeoutError(f'{phase} exceeded {timeout}s')
            time.sleep(30)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    print(f'[{phase}] completed in {time.time() - started:.1f}s; log={log_path}')

"""),
    code("""run_pilot('preflight')
print('CUDA/EGL preflight passed.')
run_pilot('debug')
print('PyTorch Dreamer ten-update gate passed.')
"""),
    markdown("""## Independent single-seed 100k-environment-step baseline

This is an independent run, not a seamless continuation of the earlier short
pilot. The formal process saves the pre-update initial checkpoint from the
same model immediately after prefill, then runs one seed with
`train_every=1000`, `train_steps=100`, `batch_size=16`, `batch_length=50`,
and replay capacity 60,000 decision transitions.
"""),
    code("""# 50,000 decision steps x action_repeat 2 = 100,000 environment steps.
run_pilot('train', '--n-steps', '50000')
run_pilot('export')
"""),
    markdown("""Outputs are written to `/kaggle/working/dreamer_pytorch_100k_env_run/`.
At minimum, download the CPU audit/probe logs, `source_manifest.json`,
`compatibility_patches.json`, `config.json`, `metrics.jsonl`,
`debug_summary.json`, `initial_policy.pkl`, `initial_policy_metadata.json`,
`train_summary.json`, `eval_episodes.csv`, `summary.json`, the training
checkpoint, and `dreamer_pilot_artifacts.zip` before stopping the session.
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

out = ROOT / "notebooks" / "kaggle_dreamer_v1_pytorch_cartpole_pilot.ipynb"
out.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(out)
