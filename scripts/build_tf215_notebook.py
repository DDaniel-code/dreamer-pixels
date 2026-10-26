#!/usr/bin/env python3
"""Build the self-contained Kaggle recovery notebook from the tracked runner."""

from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / "scripts" / "kaggle_pilot.py").read_bytes()
RUNNER_B64 = base64.b64encode(RUNNER).decode("ascii")


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


cells = [
    markdown(
        """# Dreamer V1 TF 2.15 Recovery Pilot

This notebook deliberately stops using the Kaggle default Python 3.13/Keras 3 interpreter. It creates an isolated Python 3.10 environment, installs TensorFlow 2.15/Keras 2.15/TFP 0.23, and runs every Dreamer phase through that environment's Python subprocess. The route is bounded: the recovery gate has a 60-minute deadline, and no long training starts unless the ten-update gate and fresh-process checkpoint evaluation pass.

The implementation is the authors' TensorFlow 2 Dreamer V1 code at commit `56d4d444dfd0582b0e79dab80aebbea74c0ce40d`. This is a pilot, not a complete paper reproduction.
"""
    ),
    code(
        """from pathlib import Path
import os, shutil, subprocess, sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
root = Path('/kaggle/working/dreamer_tf215_pilot')
root.mkdir(parents=True, exist_ok=True)
envdir = root / 'python310'
venv_python = envdir / 'bin' / 'python'
venv_pip = envdir / 'bin' / 'pip'

def run(args, **kwargs):
    print('$', ' '.join(map(str, args)), flush=True)
    return subprocess.run(args, check=True, **kwargs)

if venv_python.exists() and venv_pip.exists():
    py = venv_python
else:
    if envdir.exists():
        shutil.rmtree(envdir)
    candidate = shutil.which('python3.10') or shutil.which('python3.10m')
    if candidate:
        run([candidate, '-m', 'venv', str(envdir)])
    else:
        # Kaggle images may expose an unrelated Python package named ``mamba``
        # on PATH.  Do not treat that executable as a conda-compatible solver:
        # it accepts neither ``create`` nor the conda flags and would fail
        # before the isolated environment is created.  Prefer real conda/
        # micromamba binaries, then uv, and only use a discoverable Python 3.10.
        manager = shutil.which('conda') or shutil.which('micromamba')
        if manager:
            run([manager, 'create', '-y', '-p', str(envdir), 'python=3.10'], timeout=1200)
        else:
            uv = shutil.which('uv')
            if not uv:
                raise RuntimeError('No Python 3.10, conda/mamba, or uv is available; the bounded recovery stops here.')
            run([uv, 'venv', '--seed', '--python', '3.10', str(envdir)], timeout=1200)
    py = venv_python

packages = [
    'numpy==1.26.4', 'tensorflow==2.15.1', 'keras==2.15.0',
    'tensorflow-probability==0.23.0', 'dm-env==1.6',
    'dm-control==1.0.14', 'mujoco==3.1.6', 'gym==0.23.1',
    'pandas', 'matplotlib', 'pillow', 'imageio',
]
run([str(py), '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'], timeout=600)
run([str(py), '-m', 'pip', 'install', '--no-cache-dir', *packages], timeout=1200)
print('isolated interpreter:', subprocess.check_output([str(py), '-c', 'import sys; print(sys.executable)'], text=True).strip())
print('default notebook interpreter:', sys.executable)
"""
    ),
    code(
        """import base64
runner = root / 'kaggle_pilot.py'
runner.write_bytes(base64.b64decode('""" + RUNNER_B64 + """'))
print('runner:', runner)
print('all framework phases will use:', py)
"""
    ),
    markdown("""## Recovery gate: environment, GPU, pixels, and ten real updates"""),
    code(
        """import subprocess

run([str(py), str(runner), '--phase', 'preflight', '--legacy-tf215', '--output-dir', str(root / 'run')], timeout=1200)
run([str(py), str(runner), '--phase', 'debug', '--legacy-tf215', '--output-dir', str(root / 'run')], timeout=1800)
run([str(py), str(runner), '--phase', 'resume_check', '--legacy-tf215', '--output-dir', str(root / 'run')], timeout=1200)
print('Recovery gate passed; only now is the baseline training cell authorized.')
"""
    ),
    markdown("""## Single-seed Dreamer V1 baseline

This cell is intentionally separate from the recovery gate. If any previous cell fails, this cell must not be run. The runner preserves a separate training timer and saves the upstream checkpoint after each collection block.
"""),
    code(
        """run([str(py), str(runner), '--phase', 'train', '--legacy-tf215', '--output-dir', str(root / 'run')], timeout=21600)
"""
    ),
    markdown("""## Same-seed random/Dreamer evaluation and export

The final evaluation uses environment seeds 1000–1009. These are evaluation episodes, not training seeds.
"""),
    code(
        """run([str(py), str(runner), '--phase', 'final_eval', '--legacy-tf215', '--output-dir', str(root / 'run')], timeout=1800)
run([str(py), str(runner), '--phase', 'export', '--legacy-tf215', '--output-dir', str(root / 'run')], timeout=600)
print('Outputs:', root / 'run')
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "notebooks" / "kaggle_dreamer_v1_tf215_recovery_pilot.ipynb"
out.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(out)
