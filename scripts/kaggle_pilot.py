#!/usr/bin/env python3
"""Run the bounded Kaggle Dreamer V1 TensorFlow 2 pilot.

The script downloads one pinned upstream commit, applies only explicit
compatibility/evaluation patches, performs short gates, and then runs a
single-seed cartpole_swingup pilot with recoverable outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


UPSTREAM_URL = "https://github.com/danijar/dreamer.git"
UPSTREAM_COMMIT = "56d4d444dfd0582b0e79dab80aebbea74c0ce40d"
HISTORICAL_URL = "https://github.com/google-research/dreamer.git"
HISTORICAL_COMMIT = "d517542f9777c53864ff6f2683cea88be9ea98ac"
DEFAULT_ROOT = Path("/kaggle/working/dreamer_pilot")
ENV_NAME = "dmc_cartpole_swingup"
ACTION_REPEAT = 2
EPISODE_ENV_STEPS = 1000
TARGET_ENV_STEPS = 200_000
SEED = 0
DEBUG_END_MINUTES = 60
TRAIN_END_MINUTES = 285
EVAL_END_MINUTES = 330
FINAL_END_MINUTES = 360


def now() -> float:
    return time.time()


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def log_line(root: Path, message: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pilot.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(message, flush=True)


def run_cmd(root: Path, args: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    log_line(root, "$ " + " ".join(args))
    result = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (root / "commands.log").open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(args)}\n{result.stdout}\n")
    print(result.stdout, end="", flush=True)
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(args)}")
    return result


def source_dir(root: Path) -> Path:
    return root / "upstream_dreamer"


def ensure_source(root: Path) -> Path:
    src = source_dir(root)
    if not (src / ".git").exists():
        run_cmd(root, ["git", "clone", UPSTREAM_URL, str(src)])
    run_cmd(root, ["git", "fetch", "--all", "--tags"], cwd=src)
    run_cmd(root, ["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=src)
    actual = run_cmd(root, ["git", "rev-parse", "HEAD"], cwd=src).stdout.strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit mismatch: {actual} != {UPSTREAM_COMMIT}")
    return src


def replace_once(path: Path, old: str, new: str, patches: list[dict[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence in {path.name}, found {count}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    patches.append({"file": str(path.name), "old": old, "new": new})


def patch_source(root: Path, src: Path) -> list[dict[str, str]]:
    marker = src / ".codex_patches.json"
    if marker.exists():
        return read_json(marker, [])
    patches: list[dict[str, str]] = []
    for name in ("dreamer.py", "models.py", "tools.py"):
        replace_once(
            src / name,
            "from tensorflow.keras.mixed_precision import experimental as prec",
            "from tensorflow.keras import mixed_precision as prec",
            patches,
        )
    replace_once(src / "dreamer.py", "self._strategy.experimental_run_v2(", "self._strategy.run(", patches)
    replace_once(src / "dreamer.py", "prec.set_policy(", "prec.set_global_policy(", patches)
    replace_once(src / "dreamer.py", "m.reset_states()", "m.reset_state()", patches)
    replace_once(src / "models.py", "self._cell.get_initial_state(None, batch_size, dtype)", "self._cell.get_initial_state(batch_size=batch_size, dtype=dtype)", patches)
    replace_once(src / "dreamer.py", "if self._should_train(step):", "if training and self._should_train(step):", patches)
    replace_once(src / "dreamer.py", "def make_env(config, writer, prefix, datadir, store):", "def make_env(config, writer, prefix, datadir, store, seed=None):", patches)
    replace_once(src / "dreamer.py", "env = wrappers.DeepMindControl(task)", "env = wrappers.DeepMindControl(task, seed=seed)", patches)
    replace_once(src / "wrappers.py", "def __init__(self, name, size=(64, 64), camera=None):", "def __init__(self, name, size=(64, 64), camera=None, seed=None):", patches)
    replace_once(src / "wrappers.py", "self._env = suite.load(domain, task)", "self._env = suite.load(domain, task, task_kwargs={'random': seed} if seed is not None else None)", patches)
    write_json(marker, patches)
    write_json(root / "patches.json", {"upstream_commit": UPSTREAM_COMMIT, "patches": patches})
    return patches


def set_seed(seed: int) -> None:
    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_rng(root: Path) -> None:
    import numpy as np
    import tensorflow as tf

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        state["tensorflow_global_generator"] = tf.random.get_global_generator().state.numpy()
    except Exception as exc:  # The manifest records when TF does not expose it.
        state["tensorflow_global_generator_error"] = repr(exc)
    with (root / "rng_state.pkl").open("wb") as handle:
        pickle.dump(state, handle)


def restore_rng(root: Path) -> None:
    import tensorflow as tf

    with (root / "rng_state.pkl").open("rb") as handle:
        state = pickle.load(handle)
    random.setstate(state["python"])
    import numpy as np
    np.random.set_state(state["numpy"])
    if "tensorflow_global_generator" in state:
        tf.random.get_global_generator().state.assign(state["tensorflow_global_generator"])


def elapsed(root: Path) -> float:
    return max(0.0, now() - float((root / "started_at.txt").read_text(encoding="utf-8")))


def stage_open(root: Path, end_minutes: int, reserve_seconds: int = 0) -> bool:
    return elapsed(root) < end_minutes * 60 - reserve_seconds


def manifest(root: Path, src: Path, phase: str, patches: list[dict[str, str]]) -> dict[str, Any]:
    import importlib.metadata as metadata
    import platform

    packages = {}
    for name in ("tensorflow", "tensorflow-probability", "dm-control", "dm-env", "dm-tree", "mujoco", "gym", "numpy", "pandas", "matplotlib"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    import tensorflow as tf
    gpus = [device.name for device in tf.config.list_physical_devices("GPU")]
    data = {
        "phase": phase,
        "upstream": {"repository": UPSTREAM_URL, "commit": UPSTREAM_COMMIT, "historical_repository": HISTORICAL_URL, "historical_commit": HISTORICAL_COMMIT},
        "contract": {"task": "cartpole_swingup", "camera": 0, "pixels": [64, 64, 3], "action_repeat": ACTION_REPEAT, "episode_environment_steps": EPISODE_ENV_STEPS},
        "training": {"seed": SEED, "target_environment_steps": TARGET_ENV_STEPS, "deter_size": 200, "stoch_size": 30, "batch_size": 50, "batch_length": 50, "imagination_horizon": 15},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "tensorflow": tf.__version__, "gpus": gpus, "packages": packages},
        "patches": patches,
        "elapsed_seconds": elapsed(root),
    }
    write_json(root / "environment_manifest.json", data)
    return data


def import_upstream(src: Path):
    sys.path.insert(0, str(src))
    import dreamer  # type: ignore
    import models  # type: ignore
    import tools  # type: ignore
    import wrappers  # type: ignore
    return dreamer, models, tools, wrappers


def configure(dreamer, root: Path, *, debug: bool = False):
    config = dreamer.define_config()
    config.logdir = root / "logdir"
    config.seed = SEED
    config.task = ENV_NAME
    config.envs = 1
    config.parallel = "none"
    config.action_repeat = ACTION_REPEAT
    config.time_limit = EPISODE_ENV_STEPS
    config.prefill = 5000
    config.eval_every = 10000
    config.log_every = 1000 if not debug else 1
    config.steps = 10000 if debug else TARGET_ENV_STEPS
    config.log_scalars = True
    config.log_images = False if debug else True
    config.precision = 16
    config.deter_size = 200
    config.stoch_size = 30
    config.batch_size = 50
    config.batch_length = 50
    config.horizon = 15
    config.pretrain = 1 if debug else 100
    config.train_steps = 1 if debug else 100
    config.train_every = 1 if debug else 1000
    return config


def make_train_env(dreamer, wrappers, config, root: Path, writer, seed: int):
    datadir = config.logdir / "episodes"
    return wrappers.Async(lambda: dreamer.make_env(config, writer, "train", datadir, store=True, seed=seed), config.parallel)


def make_eval_env(wrappers, seed: int):
    env = wrappers.DeepMindControl("cartpole_swingup", seed=seed)
    env = wrappers.ActionRepeat(env, ACTION_REPEAT)
    env = wrappers.NormalizeActions(env)
    env = wrappers.TimeLimit(env, EPISODE_ENV_STEPS / ACTION_REPEAT)
    return env


def evaluate(agent, wrappers, seeds: list[int], root: Path, *, tag: str) -> list[dict[str, Any]]:
    import numpy as np

    rows = []
    for seed in seeds:
        env = make_eval_env(wrappers, seed)
        obs = env.reset()
        state = None
        done = np.array([True])
        ret = 0.0
        env_steps = 0
        decision_steps = 0
        frames = []
        while True:
            batch = {key: np.asarray([value]) for key, value in obs.items()}
            action, state = agent(batch, done, state, training=False)
            action = np.asarray(action)[0]
            obs, reward, finished, _ = env.step(action)
            ret += float(reward)
            env_steps += ACTION_REPEAT
            decision_steps += 1
            done = np.array([finished])
            if len(frames) < 100:
                frames.append(np.asarray(obs["image"]).copy())
            if bool(done[0]):
                break
        rows.append({"tag": tag, "seed": seed, "return": ret, "environment_steps": env_steps, "decision_steps": decision_steps, "wall_time": elapsed(root)})
        if hasattr(env, "close"):
            env.close()
    path = root / "eval_episodes.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    return rows


def append_metric(root: Path, value: dict[str, Any]) -> None:
    with (root / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def gradient_audit(root: Path) -> dict[str, Any]:
    path = root / "logdir" / "metrics.jsonl"
    values = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "actor_grad_norm" in item:
                values.append(item)
    latest = values[-1] if values else {}
    names = ["model_grad_norm", "actor_grad_norm", "value_grad_norm"]
    result = {name: latest.get(name) for name in names}
    result["finite_nonzero"] = bool(values) and all(isinstance(result[name], (int, float)) and result[name] == result[name] and abs(result[name]) > 0 for name in names)
    write_json(root / "gradient_audit.json", result)
    return result


def preflight(root: Path) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    src = ensure_source(root)
    patches = patch_source(root, src)
    dreamer, _, _, wrappers = import_upstream(src)
    set_seed(SEED)
    data = manifest(root, src, "preflight", patches)
    import numpy as np
    import tensorflow as tf

    if not data["runtime"]["gpus"]:
        raise RuntimeError("TensorFlow sees no GPU; do not start a long pilot")
    if tf.config.list_physical_devices("GPU"):
        with tf.device("/GPU:0"):
            x = tf.ones((256, 256), dtype=tf.float32)
            y = tf.linalg.matmul(x, x)
            _ = float(tf.reduce_sum(y).numpy())
    env = wrappers.DeepMindControl("cartpole_swingup", seed=SEED)
    obs = env.reset()
    if list(np.asarray(obs["image"]).shape) != [64, 64, 3]:
        raise RuntimeError(f"unexpected pixel shape: {np.asarray(obs['image']).shape}")
    if hasattr(env, "close"):
        env.close()
    write_json(root / "preflight.json", {"status": "passed", "manifest": data, "image_shape": list(np.asarray(obs["image"]).shape)})
    log_line(root, "Gate 0 passed: TensorFlow GPU, EGL, pinned source, and pixels verified.")


def debug(root: Path) -> None:
    if not stage_open(root, DEBUG_END_MINUTES, reserve_seconds=60):
        raise RuntimeError("debug stage deadline already passed")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    src = ensure_source(root)
    patches = patch_source(root, src)
    dreamer, _, tools, wrappers = import_upstream(src)
    set_seed(SEED)
    config = configure(dreamer, root, debug=True)
    config.logdir.mkdir(parents=True, exist_ok=True)
    writer = __import__("tensorflow").summary.create_file_writer(str(config.logdir))
    datadir = config.logdir / "episodes"
    env = make_train_env(dreamer, wrappers, config, root, writer, SEED)
    actspace = env.action_space
    step = tools.count_steps(datadir, config) if datadir.exists() else 0
    if step < config.prefill:
        tools.simulate(lambda o, d, s: ([actspace.sample() for _ in d], None), [env], (config.prefill - step) / ACTION_REPEAT)
    agent = dreamer.Dreamer(config, datadir, actspace, writer)
    before = [value.numpy().copy() for value in agent.variables]
    tools.simulate(agent, [env], steps=1)
    after = [value.numpy().copy() for value in agent.variables]
    changed = any((a != b).any() for a, b in zip(before, after))
    agent.save(config.logdir / "variables.pkl")
    save_rng(root)
    audit = gradient_audit(root)
    append_metric(root, {"phase": "debug", "environment_steps": tools.count_steps(datadir, config), "decision_steps": 1, "gradient_updates": 1, "wall_time": elapsed(root), "variables_changed": changed})
    if not changed or not audit.get("finite_nonzero"):
        raise RuntimeError(f"debug gradient/update audit failed: changed={changed}, audit={audit}")
    if hasattr(env, "close"):
        env.close()
    log_line(root, "Gate 1 passed: one update changed variables and gradient norms are finite/nonzero.")
    evaluate(agent, wrappers, [2000], root, tag="debug")


def resume_check(root: Path) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    src = ensure_source(root)
    patches = patch_source(root, src)
    dreamer, _, tools, wrappers = import_upstream(src)
    set_seed(SEED)
    config = configure(dreamer, root, debug=True)
    writer = __import__("tensorflow").summary.create_file_writer(str(config.logdir))
    datadir = config.logdir / "episodes"
    env = make_train_env(dreamer, wrappers, config, root, writer, SEED)
    agent = dreamer.Dreamer(config, datadir, env.action_space, writer)
    agent.load(config.logdir / "variables.pkl")
    restore_rng(root)
    rows = evaluate(agent, wrappers, [2001], root, tag="resume_check")
    append_metric(root, {"phase": "resume_check", "environment_steps": tools.count_steps(datadir, config), "wall_time": elapsed(root), "loaded": True, "eval_return": rows[0]["return"]})
    if hasattr(env, "close"):
        env.close()
    write_json(root / "resume_check.json", {"status": "passed", "rows": rows})
    log_line(root, "Checkpoint restore passed in a fresh Python process.")


def train(root: Path) -> None:
    if not stage_open(root, TRAIN_END_MINUTES, reserve_seconds=60):
        log_line(root, "Training stage skipped because its wall-clock boundary passed.")
        return
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    src = ensure_source(root)
    patches = patch_source(root, src)
    dreamer, _, tools, wrappers = import_upstream(src)
    set_seed(SEED)
    config = configure(dreamer, root, debug=False)
    config.logdir.mkdir(parents=True, exist_ok=True)
    writer = __import__("tensorflow").summary.create_file_writer(str(config.logdir))
    datadir = config.logdir / "episodes"
    env = make_train_env(dreamer, wrappers, config, root, writer, SEED)
    agent = dreamer.Dreamer(config, datadir, env.action_space, writer)
    checkpoint = config.logdir / "variables.pkl"
    if checkpoint.exists():
        agent.load(checkpoint)
        if (root / "rng_state.pkl").exists():
            restore_rng(root)
    state = None
    last_eval = -1
    while stage_open(root, TRAIN_END_MINUTES, reserve_seconds=60):
        step = tools.count_steps(datadir, config)
        if step >= TARGET_ENV_STEPS:
            break
        if step // config.eval_every > last_eval:
            rows = evaluate(agent, wrappers, [2000, 2001, 2002, 2003, 2004], root, tag="periodic")
            append_metric(root, {"phase": "periodic_eval", "environment_steps": step, "decision_steps": step // ACTION_REPEAT, "gradient_updates": None, "wall_time": elapsed(root), "mean_return": sum(row["return"] for row in rows) / len(rows)})
            last_eval = step // config.eval_every
            agent.save(checkpoint)
            save_rng(root)
        remaining = min(config.eval_every, TARGET_ENV_STEPS - step)
        state = tools.simulate(agent, [env], steps=remaining // ACTION_REPEAT, state=state)
        step = tools.count_steps(datadir, config)
        agent.save(checkpoint)
        save_rng(root)
        append_metric(root, {"phase": "train", "environment_steps": step, "decision_steps": step // ACTION_REPEAT, "gradient_updates": None, "wall_time": elapsed(root)})
    if hasattr(env, "close"):
        env.close()
    manifest(root, src, "train_complete", patches)
    log_line(root, f"Training stage ended at {tools.count_steps(datadir, config)} environment steps.")


def final_eval(root: Path) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    src = ensure_source(root)
    patches = patch_source(root, src)
    dreamer, _, tools, wrappers = import_upstream(src)
    set_seed(SEED)
    config = configure(dreamer, root, debug=False)
    writer = __import__("tensorflow").summary.create_file_writer(str(config.logdir))
    datadir = config.logdir / "episodes"
    env = make_train_env(dreamer, wrappers, config, root, writer, SEED)
    agent = dreamer.Dreamer(config, datadir, env.action_space, writer)
    checkpoint = config.logdir / "variables.pkl"
    if checkpoint.exists():
        agent.load(checkpoint)
    rows = evaluate(agent, wrappers, list(range(1000, 1010)), root, tag="final")
    append_metric(root, {"phase": "final_eval", "environment_steps": tools.count_steps(datadir, config), "decision_steps": tools.count_steps(datadir, config) // ACTION_REPEAT, "wall_time": elapsed(root), "mean_return": sum(row["return"] for row in rows) / len(rows)})
    if hasattr(env, "close"):
        env.close()
    write_json(root / "final_eval.json", {"status": "passed", "episodes": rows})
    log_line(root, "Final evaluation completed with fixed seeds 1000-1009.")


def export_summary(root: Path) -> None:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.stat().st_size < 20 * 1024 * 1024:
            files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    summary = {
        "status": "completed" if (root / "final_eval.json").exists() else "partial",
        "upstream_commit": UPSTREAM_COMMIT,
        "historical_commit": HISTORICAL_COMMIT,
        "elapsed_seconds": elapsed(root),
        "wall_clock_budget_seconds": FINAL_END_MINUTES * 60,
        "files": files,
        "limitations": ["single training seed", "pilot is not a paper reproduction", "GPU quota usage must be checked in Kaggle UI"],
    }
    write_json(root / "summary.json", summary)
    log_line(root, f"Exported summary with {len(files)} files; status={summary['status']}.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["preflight", "debug", "resume_check", "train", "final_eval", "export", "all"], required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == "preflight" and not (root / "started_at.txt").exists():
        (root / "started_at.txt").write_text(str(now()), encoding="utf-8")
    try:
        if args.phase in ("preflight", "all"):
            preflight(root)
        if args.phase in ("debug", "all"):
            debug(root)
            resume_check(root)
        if args.phase in ("train", "all"):
            train(root)
        if args.phase in ("final_eval", "all"):
            final_eval(root)
        if args.phase in ("export", "all"):
            export_summary(root)
        return 0
    except Exception as exc:
        write_json(root / "failure.json", {"status": "failed", "phase": args.phase, "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": elapsed(root) if (root / "started_at.txt").exists() else None})
        log_line(root, f"FAILED in {args.phase}: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
