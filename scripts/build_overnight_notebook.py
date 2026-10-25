#!/usr/bin/env python3
"""Build the persistence-first, bounded Kaggle notebook for the seed-0 run."""

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
    markdown("""# Dreamer V1 — persistence-first overnight seed-0 run

This notebook uses Julius Frost's pinned PyTorch Dreamer V1 implementation at
`ccea6ae4a397a94c328891bd78e81d52dd156cb6` and the pinned `rlpyt` submodule.
The run is independent of the earlier 100k and partial long runs. It first
proves that a small checkpoint survives the Kaggle version/output path, then
passes the already validated environment and update gates, and only then
starts one bounded seed-0 run. The target is 250,000 decision steps (500,000
environment steps), but the wall-clock deadline takes priority and no result
is extended beyond the current session window.
"""),
    code("""import base64
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

run_id = 'seed0_overnight_20261024_01'
root = Path('/kaggle/working') / f'dreamer_{run_id}'
root.mkdir(parents=True, exist_ok=True)
runner = root / 'kaggle_pytorch_pilot.py'
runner.write_bytes(base64.b64decode('""" + RUNNER_B64 + """'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['DREAMER_RUN_ID'] = run_id
os.environ['DREAMER_PILOT_ROOT'] = str(root)
print('run_id:', run_id)
print('runner:', runner)
print('root:', root)
"""),
    markdown("""## Gate 0 — persistence probe (no Dreamer training)

This creates a tiny checkpoint and CSV, loads the checkpoint in a fresh child
process, and writes a minimal bundle. Save a Kaggle version after this cell,
download the bundle, and verify it locally before starting the GPU gates.
"""),
    code("""def run_child(phase, *args, timeout=None, child_root=root):
    child_root.mkdir(parents=True, exist_ok=True)
    log_path = child_root / f'{phase}_subprocess.log'
    env = os.environ.copy()
    env.update({
        'DREAMER_PILOT_ROOT': str(child_root),
        'DREAMER_RUN_ID': run_id,
        'CUDA_VISIBLE_DEVICES': '0',
        'MUJOCO_GL': 'egl',
    })
    command = [sys.executable, str(runner), '--phase', phase, *map(str, args)]
    started = time.time()
    with log_path.open('w', encoding='utf-8') as stream:
        proc = subprocess.Popen(command, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, text=True)
        while proc.poll() is None:
            elapsed = time.time() - started
            print(f'[{phase}] elapsed={elapsed/60:.1f} min bytes={log_path.stat().st_size}', flush=True)
            if timeout is not None and elapsed > timeout:
                proc.terminate()
                raise TimeoutError(f'{phase} exceeded {timeout}s')
            time.sleep(30)
    status = 'completed' if proc.returncode == 0 else 'failed'
    print(f'[{phase}] {status} return_code={proc.returncode} elapsed={(time.time()-started)/60:.1f} min')
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, command)

run_child('persistence_probe', timeout=5*60)
print((root / 'persistence_probe_bundle.zip').resolve())
"""),
    markdown("""## CPU/environment audit

The fixed `dm-control --no-deps` route is reused only if `dm_control` is not
already visible. This child records the interpreter, package locations, and a
fresh cartpole reset plus ten random actions before GPU work.
"""),
    code("""install_log = root / 'dm_control_no_deps_install.log'
if importlib.util.find_spec('dm_control') is None:
    install_command = [
        sys.executable, '-m', 'pip', 'install', '--no-deps',
        'dm-env==1.6', 'mujoco==3.3.7', 'dm-control==1.0.34',
        'glfw==2.11.0', 'pyprind==2.11.3',
    ]
    install = subprocess.run(install_command, capture_output=True, text=True)
    install_log.write_text(
        'COMMAND: ' + ' '.join(install_command) + '\\n\\nRETURN: ' + str(install.returncode)
        + '\\n\\nSTDOUT:\\n' + install.stdout + '\\n\\nSTDERR:\\n' + install.stderr,
        encoding='utf-8')
    print(install_log.read_text())
    install.check_returncode()
else:
    install_log.write_text('dm_control already visible; no installation repeated.\\n', encoding='utf-8')

probe = root / 'cpu_dm_control_probe.py'
probe.write_text('''import importlib.util\nimport sys\nimport numpy as np\nprint("sys.executable:", sys.executable)\nprint("sys.path:", sys.path)\nprint("dm_control spec:", importlib.util.find_spec("dm_control"))\nfrom dm_control import suite\nenv = suite.load("cartpole", "swingup", task_kwargs={"random": 0})\ntry:\n    ts = env.reset()\n    spec = env.action_spec()\n    print("reset:", ts.reward, ts.last(), "action_shape:", spec.shape)\n    for i in range(10):\n        ts = env.step(np.random.uniform(spec.minimum, spec.maximum).astype(spec.dtype))\n        print("step", i + 1, "reward:", ts.reward, "last:", ts.last())\nfinally:\n    env.close()\nprint("CPU dm_control reset/step: PASS")\n''', encoding='utf-8')

audit_lines = []
for label, command in [
    ('python', [sys.executable, '-c', 'import sys; print(sys.executable); print(sys.version); print(sys.path)']),
    ('pip', [sys.executable, '-m', 'pip', '--version']),
    ('dm-control', [sys.executable, '-m', 'pip', 'show', '-f', 'dm-control']),
]:
    result = subprocess.run(command, capture_output=True, text=True)
    audit_lines.append(label + '\\n' + result.stdout + result.stderr)
(root / 'cpu_package_audit.txt').write_text('\\n\\n'.join(audit_lines), encoding='utf-8')
result = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True)
(root / 'cpu_dm_control_probe.log').write_text(
    f'RETURN: {result.returncode}\\n\\nSTDOUT:\\n{result.stdout}\\nSTDERR:\\n{result.stderr}', encoding='utf-8')
print((root / 'cpu_dm_control_probe.log').read_text())
result.check_returncode()
"""),
    markdown("""## Gate 1 — CUDA/EGL and ten real Dreamer updates"""),
    code("""run_child('preflight', timeout=20*60)
run_child('debug', timeout=20*60)
print('CUDA/EGL and three-module update gate passed.')
"""),
    markdown("""## Gate 2 — independent seed-0 long training

The formal run keeps the validated configuration: prefill 1,000 decision
steps, replay capacity 60,000 decision transitions, `train_every=1000`,
`train_steps=100`, batch size 16, sequence length 50, action repeat 2, and
imagination horizon 15. The 210-minute training cap leaves time for final
evaluation, minimal export, and persistence checks. The runner writes the
formal pre-update initial checkpoint and stage/wall-clock backups.
"""),
    code("""run_child('train_long', '--n-steps', '250000', '--max-train-minutes', '210', timeout=225*60)
print('Long training child completed; inspect train_summary.json and status_train_long.json.')
"""),
    markdown("""## Gate 3 — final evaluation, export, and manifest

This cell runs the final evaluation only after a successful training child,
then makes a minimal bundle that excludes the upstream checkout. The four
status fields distinguish computation from persistence; Kaggle output is not
called locally verified until the downloaded files are checked outside the
session.
"""),
    code("""checkpoint = root / 'long_train_run' / 'run_0' / 'params.pkl'
initial = root / 'initial_policy.pkl'
if not checkpoint.exists() or not initial.exists():
    raise FileNotFoundError(f'missing formal checkpoints: {checkpoint} / {initial}')
run_child('eval', '--checkpoint', checkpoint, '--initial-checkpoint', initial, timeout=30*60)
run_child('export_minimal', timeout=10*60)

manifest_rows = []
for item in sorted(root.rglob('*')):
    if item.is_file() and 'upstream_dreamer_pytorch' not in item.parts:
        digest = hashlib.sha256(item.read_bytes()).hexdigest()
        manifest_rows.append({'path': str(item.relative_to(root)), 'size': item.stat().st_size, 'sha256': digest})
with (root / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=['path', 'size', 'sha256'])
    writer.writeheader()
    writer.writerows(manifest_rows)
statuses = {}
for phase in ['persistence_probe', 'preflight', 'debug', 'train_long', 'eval', 'export_minimal']:
    path = root / f'status_{phase}.json'
    statuses[phase] = json.loads(path.read_text()) if path.exists() else {'status': 'missing'}
statuses['local_download_verified'] = False
(root / 'run_status.json').write_text(json.dumps(statuses, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
print('manifest files:', len(manifest_rows))
print('run_status:', json.dumps(statuses, indent=2, sort_keys=True))
"""),
    markdown("""## Output contract

The unique output root is `/kaggle/working/dreamer_seed0_overnight_20261024_01/`.
Download and hash `persistence_probe_bundle.zip` before proceeding to the
long run. After the long run, download the final minimal bundle, checkpoint,
progress, CSV, logs, videos, `run_status.json`, and `manifest.csv`. Do not
report a Kaggle output path as locally verified until the files are actually
downloaded and loaded in a new local process.
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

out = ROOT / "notebooks" / "kaggle_dreamer_v1_seed0_overnight_20261024.ipynb"
out.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(out)
