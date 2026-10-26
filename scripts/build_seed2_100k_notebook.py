#!/usr/bin/env python3
"""Derive a gated seed-2/100k Kaggle notebook from the existing runner notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "kaggle_dreamer_v1_seed1_matching_20261025.ipynb"
OUTPUT = ROOT / "notebooks" / "kaggle_dreamer_v1_seed2_100k_20261025.ipynb"


def cell(kind: str, source: str) -> dict:
    if kind == "markdown":
        return {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    return {"cell_type": kind, "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(True)}


gate_markdown = """## Seed-1 provenance gate

The teammate reports that seed 1 was run from the repository's seed-0 100k
notebook with only the training seed changed to 1. The attached aggregate
summary confirms the task, ten evaluation seeds, and seed-1 summary values,
but does not independently prove every training configuration field. This
cell records that limitation and lets the independent seed-2 run proceed
with the repository notebook as the configuration source of truth.
"""

gate_code = r'''import csv
import hashlib
import json
from pathlib import Path

seed0 = {
    "task": "cartpole_swingup", "n_steps": 50000,
    "action_repeat": 2, "batch_size": 16, "batch_length": 50,
    "prefill_decision_steps": 1000, "train_every": 1000,
    "train_steps_per_itr": 100, "replay_capacity_decision_transitions": 60000,
}
summary = {
    "task": "cartpole_swingup",
    "seeds": list(range(1000, 1010)),
    "dreamer_mean_return": 227.0930596746478,
    "dreamer_sample_std_return": 29.473746946933314,
    "initial_mean_return": 71.937023100428,
    "initial_sample_std_return": 4.822296391089559,
    "random_mean_return": 45.15436797873804,
    "random_sample_std_return": 27.32074936914384,
    "paper_reproduction_claim": False,
    "source_summary_sha256": "8a9d51bcc040bf3c2368eda9f8f97e94cfc8f11f291496273c9b02821a82a7ef",
}
cfg = {
    "seed": 1,
    "task": "cartpole_swingup",
    "n_steps": 50000,
    "environment_steps": 100000,
    "action_repeat": 2,
    "batch_size": 16,
    "batch_length": 50,
    "prefill_decision_steps": 1000,
    "train_every": 1000,
    "train_steps_per_itr": 100,
    "replay_capacity_decision_transitions": 60000,
    "provenance": "teammate_reported_same_repository_notebook_only_seed_changed",
}
errors = []
if int(cfg.get('seed', -1)) != 1: errors.append(f"seed={cfg.get('seed')}, expected 1")
if cfg.get('task') != seed0['task']: errors.append('task mismatch')
for key in ('action_repeat', 'batch_size', 'batch_length', 'prefill_decision_steps', 'train_every', 'train_steps_per_itr', 'replay_capacity_decision_transitions'):
    if key in cfg and int(cfg[key]) != int(seed0[key]): errors.append(f'{key} mismatch')
actual_env = summary.get('actual_environment_steps') or summary.get('environment_steps') or cfg.get('environment_steps')
if actual_env is not None and int(actual_env) != 100000: errors.append(f'actual environment steps={actual_env}, expected 100000')
if summary.get('task') != seed0['task']: errors.append('summary task mismatch')
if summary.get('seeds') != list(range(1000, 1010)): errors.append('summary evaluation seeds mismatch')
gate = {
    'status': 'provisional_match' if not errors else 'blocked_mismatch',
    'errors': errors,
    'training_seed': cfg.get('seed'),
    'actual_environment_steps': actual_env,
    'evaluation_rows': 10,
    'raw_config_present': False,
    'raw_eval_csv_present': False,
    'checkpoint_present': False,
    'evidence_level': 'user_attested_repository_notebook_plus_aggregate_summary',
    'repository_notebook_source': 'notebooks/kaggle_dreamer_v1_pytorch_cartpole_pilot_e8b9486.ipynb',
    'upstream_commit': 'ccea6ae4a397a94c328891bd78e81d52dd156cb6',
    'rlpyt_commit': 'f04f23db1eb7b5915d88401fca67869968a07a37',
    'summary_sha256': summary['source_summary_sha256'],
}
(ROOT / 'seed1_match_gate.json').write_text(json.dumps(gate, indent=2) + '\n')
print(json.dumps(gate, indent=2, sort_keys=True))
if errors: raise RuntimeError('Seed-1 evidence gate failed: ' + '; '.join(errors))
'''

install_code = r'''import importlib.util
import subprocess
import sys

install_log = SEED2_ROOT / 'dm_control_no_deps_install.log'
if importlib.util.find_spec('dm_control') is None:
    command = [
        sys.executable, '-m', 'pip', 'install', '--no-deps',
        'dm-env==1.6', 'mujoco==3.3.7', 'dm-control==1.0.34',
        'glfw==2.11.0', 'pyprind==2.11.3',
    ]
    with install_log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, stdout=stream,
                                stderr=subprocess.STDOUT, text=True)
    print('dm-control install returncode:', result.returncode)
    print('install log:', install_log)
    print(install_log.read_text(encoding='utf-8'))
    if result.returncode:
        raise RuntimeError('fixed dm-control --no-deps install failed')
else:
    install_log.write_text('dm_control already visible; install skipped.\n', encoding='utf-8')
    print('dm_control already visible; fixed install skipped.')
print('dm_control spec:', importlib.util.find_spec('dm_control'))
'''

train_markdown = """## Independent seed-2 100k training

This is a fresh seed-2 run: 50,000 decisions and 100,000 environment steps.
The formal seed-0 100k settings remain fixed: batch size 16, sequence length
50, prefill 1,000 decisions, train_every 1,000, train_steps 100, and replay
capacity 60,000 decision transitions. No other seed's weights are loaded.
"""

train_code = """run_child('train', '--seed', '2', '--n-steps', '50000', timeout=90*60)
final_checkpoint = ROOT / 'train_run' / 'run_0' / 'params.pkl'
initial_checkpoint = ROOT / 'initial_policy.pkl'
run_child('eval', '--seed', '2', '--checkpoint', final_checkpoint,
          '--initial-checkpoint', initial_checkpoint, timeout=25*60)
run_child('export_minimal', timeout=10*60)
print('final checkpoint:', final_checkpoint)
print('initial checkpoint:', initial_checkpoint)
"""

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
new_cells = []
for item in notebook["cells"]:
    source = "".join(item.get("source", []))
    if source.startswith("# Dreamer V1") and item["cell_type"] == "markdown":
        new_cells.append(cell("markdown", """# Dreamer V1 — gated seed-2 100k-environment-step pilot

This notebook targets one independent seed-2 baseline on `cartpole_swingup`:
50,000 decision steps and 100,000 environment steps. The 300k/330k seed-0
overnight run remains separate. Seed 1 is treated as a provisional match
because its raw config/checkpoint files were not supplied; it is not used to
claim a statistically verified three-seed result.
"""))
    elif source.startswith("import base64") and "SEED1_RUN_ID" in source:
        new_cells.append(cell("code", source.replace("SEED1", "SEED2").replace("seed1", "seed2") + "\nROOT = SEED2_ROOT\n"))
        new_cells.append(cell("code", install_code))
    elif source.startswith("def run_child"):
        new_cells.append(cell("code", source.replace("SEED1", "SEED2").replace("seed1", "seed2")))
    elif "## Input and checkpoint audit" in source:
        new_cells.append(cell("markdown", gate_markdown))
    elif "input_root = Path('/kaggle/input')" in source:
        new_cells.append(cell("code", gate_code))
    elif "## Existing seed-0 intermediate evaluation" in source or "stage_records = []" in source:
        continue
    elif "## Time guard before seed-1 training" in source:
        new_cells.append(cell("markdown", train_markdown))
    elif "dm_control_no_deps_install.log" in source:
        # The seed-1 source notebook may already contain an old install cell;
        # replace it with the single fixed cell inserted after setup above.
        continue
    elif source.startswith("run_child('persistence_probe'"):
        new_cells.append(cell("code", """run_child('preflight', '--seed', '2', timeout=15*60)
run_child('debug', '--seed', '2', timeout=15*60)
"""))
    elif source.startswith("seed0_decisions = 165366"):
        new_cells.append(cell("code", train_code))
    elif "## Output and shutdown contract" in source:
        new_cells.append(cell("markdown", """## Output and shutdown contract

Before stopping the session, download and hash the independent seed-2 run:
config, train summary, metrics, progress, initial/final checkpoints,
eval_episodes.csv, summary, logs, and the minimal bundle. Do not compute a
three-seed summary until the seed-1 gate and seed-2 CSV are independently
verified.
"""))
    else:
        if item["cell_type"] in ("code", "markdown"):
            source = source.replace("SEED1", "SEED2").replace("seed1", "seed2")
            new_cells.append(cell(item["cell_type"], source))
        else:
            new_cells.append(item)

notebook["cells"] = new_cells
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
