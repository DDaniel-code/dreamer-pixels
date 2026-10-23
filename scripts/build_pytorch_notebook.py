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
    markdown("""# Dreamer V1 PyTorch Cartpole Swingup Pilot

This notebook stops the TensorFlow route and reuses Kaggle's existing PyTorch GPU environment. It retrieves `juliusfrost/dreamer-pytorch` at commit `ccea6ae4a397a94c328891bd78e81d52dd156cb6` with the pinned `rlpyt` submodule, without reinstalling PyTorch. Formal training restores the upstream `train_every=1000` and `train_steps=100`; the retained `batch_size=16` is recorded as a memory-conscious pilot adjustment.

The first cell runs a bounded gate: 64x64 RGB pixels, camera 0, action repeat 2, episode reset contract, ten real Dreamer updates, finite diagnostics, parameter changes for world model/actor/critic, checkpoint save, and fresh-process same-seed evaluation. Only after that gate passes should the bounded single-seed baseline cell be run. README scores are not our results.
"""),
    code("""import base64
from pathlib import Path

root = Path('/kaggle/working/dreamer_pytorch_pilot')
root.mkdir(parents=True, exist_ok=True)
runner = root / 'kaggle_pytorch_pilot.py'
runner.write_bytes(base64.b64decode('""" + RUNNER_B64 + """'))
print(runner)
"""),
    code("""import os, platform, subprocess, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MUJOCO_GL'] = 'egl'
out = Path('/kaggle/working/dreamer_pytorch_pilot')
out.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps',
       'dm-env==1.6', 'mujoco==3.3.7', 'dm-control==1.0.34']
result = subprocess.run(cmd, capture_output=True, text=True)
(out / 'dm_control_no_deps_install_gpu.log').write_text(
    'COMMAND: ' + ' '.join(cmd) + '\\n\\nSTDOUT:\\n' + result.stdout +
    '\\nSTDERR:\\n' + result.stderr, encoding='utf-8')
print('sys.executable:', sys.executable)
print('python:', sys.version)
print('platform:', platform.platform())
print('dm-control install returncode:', result.returncode)
print(result.stdout)
print(result.stderr)
result.check_returncode()
glfw_cmd = [sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps', 'glfw==2.11.0']
glfw = subprocess.run(glfw_cmd, capture_output=True, text=True)
(out / 'glfw_no_deps_install_gpu.log').write_text(
    'COMMAND: ' + ' '.join(glfw_cmd) + '\\n\\nSTDOUT:\\n' + glfw.stdout +
    '\\nSTDERR:\\n' + glfw.stderr, encoding='utf-8')
print('glfw install returncode:', glfw.returncode)
print(glfw.stdout)
print(glfw.stderr)
glfw.check_returncode()
pyprind_cmd = [sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps', 'pyprind==2.11.3']
pyprind = subprocess.run(pyprind_cmd, capture_output=True, text=True)
(out / 'pyprind_no_deps_install_gpu.log').write_text(
    'COMMAND: ' + ' '.join(pyprind_cmd) + '\\n\\nSTDOUT:\\n' + pyprind.stdout +
    '\\nSTDERR:\\n' + pyprind.stderr, encoding='utf-8')
print('pyprind install returncode:', pyprind.returncode)
print(pyprind.stdout)
print(pyprind.stderr)
pyprind.check_returncode()
print('GPU environment setup packages passed')
"""),
    markdown("""## Preflight and ten-update gate

The default Kaggle interpreter is used only to launch the subprocess. The actual Dreamer code runs in that same existing PyTorch environment, with `CUDA_VISIBLE_DEVICES=0` set before framework import.
"""),
    code("""import subprocess, sys, time

def run_pilot(phase, *args):
    log_path = root / f'{phase}_subprocess.log'
    command = [sys.executable, str(runner), '--phase', phase, *args]
    started = time.time()
    with log_path.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   text=True)
        while process.poll() is None:
            elapsed = time.time() - started
            size = log_path.stat().st_size if log_path.exists() else 0
            print(f'[{phase}] child running: {elapsed:.0f}s, log_bytes={size}', flush=True)
            time.sleep(60)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    print(f'[{phase}] completed in {time.time() - started:.1f}s; log={log_path}')

run_pilot('debug')
print('PyTorch Dreamer gate passed; the baseline cell is now authorized.')
"""),
    markdown("""## Initial policy evaluation

This cell saves the untrained policy before any optimizer update and evaluates it on seeds 1000--1009. It is the training-before reference, not a learned result.
"""),
    code("""run_pilot('initial_eval')
print('Initial policy checkpoint and same-seed evaluation exported.')
"""),
    markdown("""## Bounded single-seed cartpole_swingup baseline

This is a pilot budget, not a complete 1M/5M-step paper reproduction. Adjust `--n-steps` only within the available Kaggle session budget; do not run this cell if the gate failed.
"""),
    code("""# 3,000 decision steps = 6,000 underlying environment steps.
# This is intentionally bounded; it is not a paper-scale reproduction.
run_pilot('train', '--n-steps', '3000')
run_pilot('export')
"""),
    markdown("""Outputs are written to `/kaggle/working/dreamer_pytorch_pilot/`, including `source_manifest.json`, `compatibility_patches.json`, `config.json`, `metrics.jsonl`, `debug_summary.json`, `initial_policy.pkl`, `initial_eval_episodes.csv`, `train_summary.json`, `eval_episodes.csv`, `summary.json`, the rlpyt checkpoint, and `dreamer_pilot_artifacts.zip`. The final zip must be downloaded before stopping the session.
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
