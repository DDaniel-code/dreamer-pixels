#!/usr/bin/env python3
"""Bounded Dreamer V1 PyTorch pilot and long-budget runner for Kaggle.

This adapter reuses the upstream Dreamer PyTorch implementation and rlpyt
submodule. It does not install or replace PyTorch. Compatibility changes are
limited to the evidenced PyTorch Transform contract shim. Long runs remain
single-seed, independently recorded, and wall-clock bounded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import site
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np


UPSTREAM_URL = "https://github.com/juliusfrost/dreamer-pytorch.git"
UPSTREAM_COMMIT = "ccea6ae4a397a94c328891bd78e81d52dd156cb6"
RLPYT_COMMIT = "f04f23db1eb7b5915d88401fca67869968a07a37"
LICENSE = "MIT"
ROOT = Path(os.environ.get("DREAMER_PILOT_ROOT", "/kaggle/working/dreamer_pytorch_pilot"))
UPSTREAM = ROOT / "upstream_dreamer_pytorch"
RUN_ID = os.environ.get("DREAMER_RUN_ID", ROOT.name)
DEFAULT_SEED = 0
ACTION_REPEAT = 2
IMAGE_SIZE = (64, 64)
EPISODE_DECISION_STEPS = 500
UPSTREAM_TRAIN_EVERY = 1000
UPSTREAM_TRAIN_STEPS = 100
# Explicit formal-run capacity.  At 64x64 RGB uint8, 60k decision transitions
# are roughly 0.7 GiB for observations before replay metadata/allocator overhead;
# this leaves substantial headroom in a Kaggle RAM session while covering the
# 50k-decision target instead of the old 1,060-transition debug-sized buffer.
FORMAL_REPLAY_CAPACITY = 60000


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def progress_checkpoint_updates(progress_path: Path | None, decision_steps: int) -> int | None:
    """Read an optimizer count only when the progress log has that exact row."""
    if progress_path is None or not progress_path.exists():
        return None
    with progress_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                logged_steps = int(float(row["Diagnostics/CumSteps"]))
            except (KeyError, TypeError, ValueError):
                continue
            if logged_steps == decision_steps:
                value = row.get("Diagnostics/CumUpdates")
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return None
    return None


def checkpoint_metadata(path: Path, *, progress_path: Path | None = None,
                        optimizer_updates: int | None = None) -> dict:
    """Record the exact evaluated snapshot and its counter semantics."""
    import torch

    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint is not a dict: {path}")
    itr = payload.get("itr")
    cum_steps = payload.get("cum_steps")
    decision_steps = int(cum_steps) + 1 if cum_steps is not None else None
    exact_logged_updates = (
        progress_checkpoint_updates(progress_path, decision_steps)
        if decision_steps is not None else None
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "itr": int(itr) if itr is not None else None,
        "cum_steps": int(cum_steps) if cum_steps is not None else None,
        "decision_steps": decision_steps,
        "environment_steps": decision_steps * ACTION_REPEAT if decision_steps is not None else None,
        "optimizer_updates": (
            int(optimizer_updates)
            if optimizer_updates is not None
            else exact_logged_updates
        ),
        "optimizer_updates_source": (
            "runtime_counter" if optimizer_updates is not None
            else "exact_progress_row" if exact_logged_updates is not None
            else "not_available"
        ),
    }


def progress_boundary_metadata(progress_path: Path, decision_steps: int) -> dict:
    """Describe whether the run stopped between logged progress boundaries."""
    result = {"path": str(progress_path), "exists": progress_path.exists()}
    if not progress_path.exists():
        return result
    last = None
    with progress_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            last = row
    if last is None:
        result.update({"last_logged_decision_steps": None, "last_logged_updates": None})
        return result
    try:
        logged_steps = int(float(last["Diagnostics/CumSteps"]))
    except (KeyError, TypeError, ValueError):
        logged_steps = None
    try:
        logged_updates = int(float(last["Diagnostics/CumUpdates"]))
    except (KeyError, TypeError, ValueError):
        logged_updates = None
    result.update({
        "last_logged_decision_steps": logged_steps,
        "last_logged_updates": logged_updates,
        "non_log_boundary": logged_steps is not None and logged_steps != decision_steps,
    })
    return result


def export_minimal_bundle(path: Path | None = None) -> Path:
    """Archive only auditable run outputs, never the upstream checkout."""
    ROOT.mkdir(parents=True, exist_ok=True)
    path = path or (ROOT / "backups" / f"minimal_{int(time.time())}.zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = []
    exact_names = {
        "config.json", "source_manifest.json", "compatibility_patches.json",
        "initial_policy.pkl", "initial_policy_metadata.json",
        "initial_eval_episodes.csv", "initial_eval_summary.json",
        "debug_summary.json", "train_summary.json", "summary.json",
        "resume_audit.json", "metrics.jsonl", "commands.log", "pilot.log",
        "persistence_probe.json", "persistence_probe.csv",
        "persistence_probe.pt", "persistence_probe_subprocess.log",
    }
    for item in ROOT.rglob("*"):
        if not item.is_file() or item == path or UPSTREAM in item.parents:
            continue
        rel = item.relative_to(ROOT)
        if item.name in exact_names or item.name.endswith("_subprocess.log"):
            selected.append(item)
        elif rel.parts and rel.parts[0] in {"checkpoints", "videos", "figures"}:
            selected.append(item)
        elif "run_0" in rel.parts and item.suffix in {".csv", ".json", ".jsonl", ".pkl", ".log"}:
            selected.append(item)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(set(selected)):
            archive.write(item, item.relative_to(ROOT))
    log(f"minimal_bundle={path} files={len(selected)} bytes={path.stat().st_size}")
    return path


def persistence_probe() -> None:
    """Exercise save/load in a fresh process with a tiny, downloadable artifact."""
    import torch

    probe = ROOT / "persistence_probe"
    probe.mkdir(parents=True, exist_ok=True)
    seed_all(0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    checkpoint = probe / "persistence_probe.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "loss": float(loss.item()), "run_id": RUN_ID}, checkpoint)
    (probe / "persistence_probe.csv").write_text(
        "run_id,loss\n" + f"{RUN_ID},{float(loss.item())}\n", encoding="utf-8")
    loader = (
        "import torch,sys; p=torch.load(sys.argv[1],map_location='cpu'); "
        "assert 'model' in p and 'optimizer' in p; "
        "print('fresh_process_load=PASS run_id=' + str(p['run_id']))"
    )
    result = subprocess.run([sys.executable, "-c", loader, str(checkpoint)],
                            capture_output=True, text=True)
    (probe / "persistence_probe_subprocess.log").write_text(
        f"RETURN: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        encoding="utf-8",
    )
    result.check_returncode()
    write_json(probe / "persistence_probe.json", {
        "run_id": RUN_ID, "checkpoint": str(checkpoint),
        "fresh_process_load": True, "loss_finite": bool(np.isfinite(loss.item())),
    })
    export_minimal_bundle(ROOT / "persistence_probe_bundle.zip")
    log("persistence_probe=PASS")


def log(message: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(text, flush=True)
    with (ROOT / "pilot.log").open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def run_cmd(args: list[str], cwd: Path | None = None, timeout: int = 600) -> str:
    log("$ " + " ".join(args))
    result = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout)
    with (ROOT / "commands.log").open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(args)}\n{result.stdout}\n")
    print(result.stdout, end="", flush=True)
    if result.returncode:
        raise RuntimeError(f"command failed with {result.returncode}: {' '.join(args)}")
    return result.stdout


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_upstream() -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    if not (UPSTREAM / ".git").exists():
        run_cmd(["git", "clone", "--recurse-submodules", UPSTREAM_URL, str(UPSTREAM)], timeout=600)
    run_cmd(["git", "fetch", "--all", "--tags"], cwd=UPSTREAM, timeout=300)
    run_cmd(["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=UPSTREAM, timeout=120)
    actual = run_cmd(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, timeout=120).strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit mismatch: {actual} != {UPSTREAM_COMMIT}")
    run_cmd(["git", "submodule", "update", "--init", "--recursive"], cwd=UPSTREAM, timeout=300)
    submodule = run_cmd(["git", "-C", "rlpyt", "rev-parse", "HEAD"], cwd=UPSTREAM, timeout=120).strip()
    if submodule != RLPYT_COMMIT:
        raise RuntimeError(f"rlpyt submodule mismatch: {submodule} != {RLPYT_COMMIT}")
    (UPSTREAM / "LICENSE").read_text(encoding="utf-8")
    write_json(ROOT / "source_manifest.json", {
        "repository": UPSTREAM_URL,
        "commit": UPSTREAM_COMMIT,
        "rlpyt_commit": RLPYT_COMMIT,
        "license": LICENSE,
        "source_license_path": str(UPSTREAM / "LICENSE"),
        "algorithm": "Dreamer V1 PyTorch adaptation; not V2/V3",
    })
    return UPSTREAM


def import_upstream(upstream: Path):
    # Kaggle's notebook kernel and child interpreter can expose different
    # site-package paths even when they use the same executable.  Register the
    # interpreter's package locations before importing dm-control.  This is a
    # path-visibility fix only; it does not install or change dependencies.
    for package_path in site.getsitepackages():
        site.addsitedir(package_path)
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "rlpyt"))
    import torch
    from torch.distributions import constraints
    from dreamer.models.distribution import TanhBijector

    # The pinned 2020 code predates the Transform domain/codomain contract
    # enforced by newer PyTorch releases.  This is a runtime compatibility
    # shim only; it does not change the Dreamer objective or model equations.
    patches = []
    if not hasattr(TanhBijector, "domain"):
        TanhBijector.domain = constraints.real
        TanhBijector.codomain = constraints.interval(-1.0, 1.0)
        patches.append({
            "symbol": "dreamer.models.distribution.TanhBijector",
            "change": "add Transform domain/codomain constraints",
            "reason": "required by current PyTorch TransformedDistribution",
            "status": "applied",
        })
    write_json(ROOT / "compatibility_patches.json", {
        "upstream_commit": UPSTREAM_COMMIT,
        "torch_version": torch.__version__,
        "patches": patches,
    })
    from dm_control import suite
    from rlpyt.envs.base import Env
    from dreamer.envs.env import EnvInfo
    from dreamer.envs.action_repeat import ActionRepeat
    from dreamer.envs.normalize_actions import NormalizeActions
    from dreamer.envs.time_limit import TimeLimit
    from dreamer.envs.wrapper import EnvWrapper

    class SeededDeepMindControl(Env):
        """Small environment adapter; model and algorithm remain upstream."""

        def __init__(self, name, seed=None, size=IMAGE_SIZE, camera=0):
            domain, task = name.split("_", 1)
            if domain == "cup":
                domain = "ball_in_cup"
            kwargs = {"task_kwargs": {"random": int(seed)}} if seed is not None else {}
            self._env = suite.load(domain, task, **kwargs)
            self._size = tuple(size)
            self._camera = camera

        @property
        def observation_space(self):
            from rlpyt.spaces.int_box import IntBox
            return IntBox(low=0, high=255, shape=(3,) + self._size, dtype="uint8")

        @property
        def action_space(self):
            from rlpyt.spaces.float_box import FloatBox
            spec = self._env.action_spec()
            return FloatBox(low=spec.minimum, high=spec.maximum)

        def step(self, action):
            timestep = self._env.step(action)
            obs = self.render()
            reward = timestep.reward or 0.0
            done = timestep.last()
            info = EnvInfo(np.array(timestep.discount, np.float32), None, done)
            return obs, reward, done, info

        def reset(self):
            self._env.reset()
            return self.render()

        def render(self, *args, **kwargs):
            if kwargs.get("mode", "rgb_array") != "rgb_array":
                raise ValueError("Only rgb_array rendering is supported")
            return self._env.physics.render(*self._size, camera_id=self._camera).transpose(2, 0, 1).copy()

        def close(self):
            self._env.close()

    class SeededActionRepeat(ActionRepeat):
        pass

    class SeededNormalizeActions(NormalizeActions):
        pass

    class SeededTimeLimit(TimeLimit):
        pass

    def make_env(name="cartpole_swingup", seed=None):
        env = SeededDeepMindControl(name=name, seed=seed)
        env = SeededActionRepeat(env, amount=ACTION_REPEAT)
        env = SeededNormalizeActions(env)
        env = SeededTimeLimit(env, duration=EPISODE_DECISION_STEPS)
        return env

    return torch, make_env


def preflight(upstream: Path, seed: int) -> None:
    torch, make_env = import_upstream(upstream)
    seed_all(seed)
    log(f"python={sys.executable}")
    log(f"torch={torch.__version__}; cuda_available={torch.cuda.is_available()}; cuda={torch.version.cuda}")
    log(f"device_count={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle PyTorch reports no CUDA device")
    log(f"device_0={torch.cuda.get_device_name(0)}")
    env = make_env(seed=seed)
    try:
        first = env.reset()
        if first.shape != (3, 64, 64) or first.dtype != np.uint8:
            raise RuntimeError(f"unexpected pixel observation: shape={first.shape} dtype={first.dtype}")
        action = env.action_space.null_value()
        observations = [first]
        decisions = 0
        traj_done = False
        for _ in range(EPISODE_DECISION_STEPS + 2):
            obs, reward, done, info = env.step(action)
            observations.append(obs)
            decisions += 1
            if getattr(info, "traj_done", False) or done:
                traj_done = True
                break
        after_reset = env.reset()
        if not traj_done:
            raise RuntimeError("episode reset/time-limit contract did not trigger")
        if after_reset.shape != (3, 64, 64) or after_reset.dtype != np.uint8:
            raise RuntimeError("reset returned an invalid pixel observation")
        log(json.dumps({
            "pixel_shape": list(first.shape), "pixel_dtype": str(first.dtype),
            "action_shape": list(env.action_space.shape),
            "action_repeat": ACTION_REPEAT, "decision_steps_until_reset": decisions,
            "env_steps_contract": decisions * ACTION_REPEAT,
            "pixel_min": int(np.min(first)), "pixel_max": int(np.max(first)),
            "preflight": "PASS",
        }, sort_keys=True))
    finally:
        env.close()


class CountingDreamer:
    """Factory for an upstream Dreamer with observable bounded updates."""

    @staticmethod
    def build(upstream, make_env, seed, *, batch_size, batch_length, prefill,
              train_every, train_steps, n_steps, log_dir, checkpoint_name,
              replay_capacity=None,
              log_interval_steps=None, disable_anomaly=False,
              disable_progress=False, max_train_seconds=None,
              checkpoint_dir=None, checkpoint_decision_steps=None):
        import torch
        from rlpyt.samplers.collections import TrajInfo
        from rlpyt.samplers.serial.sampler import SerialSampler
        from rlpyt.runners.minibatch_rl import MinibatchRl
        from rlpyt.utils.logging import logger
        from rlpyt.utils.logging.context import logger_context
        from dreamer.agents.dmc_dreamer_agent import DMCDreamerAgent
        import dreamer.algos.dreamer_algo as dreamer_algo_module
        from dreamer.algos.dreamer_algo import Dreamer
        from dreamer.envs.wrapper import make_wapper

        # The pinned source enables anomaly detection globally and wraps every
        # imagination update in tqdm. Keep both for the small debug gate, but
        # disable them for formal training so diagnostics do not dominate the
        # bounded Kaggle session or its captured output.
        if disable_anomaly:
            torch.autograd.set_detect_anomaly(False)
        if disable_progress:
            dreamer_algo_module.tqdm = lambda iterable, **kwargs: iterable

        class InstrumentedDreamer(Dreamer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.real_updates = 0
                self.update_records = []
                self.initial_parameters = None

            def optim_initialize(self, rank=0):
                super().optim_initialize(rank)
                self.initial_parameters = {
                    "world_model": [p.detach().cpu().clone() for m in self.model_modules for p in m.parameters()],
                    "actor": [p.detach().cpu().clone() for m in self.actor_modules for p in m.parameters()],
                    "critic": [p.detach().cpu().clone() for m in self.value_modules for p in m.parameters()],
                }

            def optimize_agent(self, *args, **kwargs):
                info = super().optimize_agent(*args, **kwargs)
                count = len(getattr(info, "loss", []))
                self.real_updates += count
                for index in range(count):
                    record = {"update": self.real_updates - count + index + 1}
                    for field in ("loss", "grad_norm_model", "grad_norm_actor", "grad_norm_value",
                                  "model_loss", "actor_loss", "value_loss"):
                        values = getattr(info, field, [])
                        if isinstance(values, list) and index < len(values):
                            record[field] = float(values[index])
                    self.update_records.append(record)
                    write_jsonl(ROOT / "metrics.jsonl", record)
                self.update_counter = self.real_updates
                return info

            def parameter_change(self):
                output = {}
                current = {
                    "world_model": [p.detach().cpu() for m in self.model_modules for p in m.parameters()],
                    "actor": [p.detach().cpu() for m in self.actor_modules for p in m.parameters()],
                    "critic": [p.detach().cpu() for m in self.value_modules for p in m.parameters()],
                }
                for name in current:
                    diffs = [(a - b).abs().max().item() for a, b in zip(self.initial_parameters[name], current[name])]
                    output[name] = {"max_abs_parameter_change": max(diffs), "changed": bool(max(diffs) > 0)}
                return output

        action_repeat = ACTION_REPEAT
        factory = make_wapper(
            make_env,
            [lambda env, amount=action_repeat: env],
            [dict()],
        )
        # make_env already applies the action-repeat/time-limit wrappers; the
        # identity wrapper keeps the upstream sampler contract unchanged.
        sampler = SerialSampler(
            EnvCls=factory,
            TrajInfoCls=TrajInfo,
            env_kwargs=dict(name="cartpole_swingup", seed=seed),
            eval_env_kwargs=dict(name="cartpole_swingup", seed=seed),
            batch_T=1, batch_B=1, max_decorrelation_steps=0,
            eval_n_envs=0,
        )
        replay_capacity = int(replay_capacity if replay_capacity is not None
                              else max(500, prefill + batch_length + 10))
        algo = InstrumentedDreamer(
            batch_size=batch_size, batch_length=batch_length,
            train_every=train_every, train_steps=train_steps, pretrain=prefill,
            prefill=prefill, replay_size=replay_capacity,
            horizon=15, log_video=False,
        )
        agent = DMCDreamerAgent(
            train_noise=0.3, eval_noise=0.0,
            expl_type="additive_gaussian", expl_min=None, expl_decay=None,
        )
        class InitialSnapshotRunner(MinibatchRl):
            """Run upstream loop and preserve the formal pre-update snapshot."""

            def train(self):
                n_itr = self.startup()
                train_started = time.time()
                checkpoint_dir_path = Path(checkpoint_dir) if checkpoint_dir else None
                if checkpoint_dir_path:
                    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
                checkpoint_steps = set(checkpoint_decision_steps or [])
                stopped_for_deadline = False
                last_itr = -1
                last_backup = time.time()
                for itr in range(n_itr):
                    last_itr = itr
                    logger.set_iteration(itr)
                    with logger.prefix(f"itr #{itr} "):
                        self.agent.sample_mode(itr)
                        samples, traj_infos = self.sampler.obtain_samples(itr)
                        self.agent.train_mode(itr)
                        opt_info = self.algo.optimize_agent(itr, samples)
                        self.store_diagnostics(itr, traj_infos, opt_info)
                        if itr == prefill - 1 and checkpoint_name != "debug_gate":
                            self.save_itr_snapshot(itr)
                            source = Path(logger.get_snapshot_dir()) / "params.pkl"
                            target = ROOT / "initial_policy.pkl"
                            if not source.exists():
                                raise RuntimeError(f"prefill snapshot missing: {source}")
                            shutil.copy2(source, target)
                            write_json(ROOT / "initial_policy_metadata.json", {
                                "source": "formal_training_prefill_snapshot",
                                "checkpoint": str(target),
                                "itr": itr,
                                "cum_steps": prefill,
                                "updates": 0,
                            })
                            log(f"saved formal initial checkpoint at itr={itr}, decision_steps={prefill}")
                        if (itr + 1) % self.log_interval_itrs == 0:
                            self.log_diagnostics(itr)
                        decision_steps = itr + 1
                        if checkpoint_dir_path and decision_steps in checkpoint_steps:
                            torch.save(
                                self.get_itr_snapshot(itr),
                                checkpoint_dir_path / f"decision_{decision_steps:06d}_env_{decision_steps * ACTION_REPEAT:06d}.pkl",
                            )
                            log(f"saved periodic checkpoint decision_steps={decision_steps}")
                            export_minimal_bundle(
                                ROOT / "backups" / f"decision_{decision_steps:06d}.zip"
                            )
                            last_backup = time.time()
                        elif checkpoint_dir_path and time.time() - last_backup >= 30 * 60:
                            backup_path = checkpoint_dir_path / (
                                f"backup_decision_{decision_steps:06d}_env_{decision_steps * ACTION_REPEAT:06d}.pkl"
                            )
                            torch.save(self.get_itr_snapshot(itr), backup_path)
                            export_minimal_bundle(
                                ROOT / "backups" / f"wallclock_{decision_steps:06d}.zip"
                            )
                            last_backup = time.time()
                        if max_train_seconds is not None and time.time() - train_started >= max_train_seconds:
                            stopped_for_deadline = True
                            break
                if checkpoint_dir_path and last_itr >= 0:
                    torch.save(
                        self.get_itr_snapshot(last_itr),
                        checkpoint_dir_path / "latest_before_shutdown.pkl",
                    )
                self.deadline_stop = stopped_for_deadline
                self.actual_decision_steps = max(0, last_itr + 1)
                self.shutdown()

        runner = InitialSnapshotRunner(
            algo=algo, agent=agent, sampler=sampler,
            n_steps=n_steps, seed=seed,
            log_interval_steps=log_interval_steps or n_steps,
            affinity=dict(cuda_idx=0),
        )
        config = {
            "run_id": RUN_ID,
            "upstream_commit": UPSTREAM_COMMIT, "rlpyt_commit": RLPYT_COMMIT,
            "task": "cartpole_swingup", "seed": seed,
            "image": "64x64 RGB", "action_repeat": ACTION_REPEAT,
            "episode_env_steps": 1000, "episode_decision_steps": EPISODE_DECISION_STEPS,
            "batch_size": batch_size, "batch_length": batch_length,
            "prefill_decision_steps": prefill, "train_every": train_every,
            "train_steps_per_itr": train_steps, "n_steps": n_steps,
            "replay_capacity_decision_transitions": replay_capacity,
            "replay_capacity_env_steps": replay_capacity * ACTION_REPEAT,
            "replay_capacity_source": "explicit_ram_budget",
            "replay_estimated_pixel_mib": replay_capacity * 3 * 64 * 64 / (1024 ** 2),
            "iteration_equals_decision_steps": True,
            "decision_steps_per_environment_step": 1 / ACTION_REPEAT,
            "environment_steps_per_decision_step": ACTION_REPEAT,
            "first_update_at_decision_step": prefill,
            "checkpoint": checkpoint_name, "anomaly_detection": not disable_anomaly,
            "nested_tqdm": not disable_progress,
            "max_train_seconds": max_train_seconds,
            "periodic_checkpoint_decision_steps": sorted(checkpoint_decision_steps or []),
        }
        write_json(ROOT / "config.json", config)
        train_started = time.time()
        with logger_context(str(log_dir), 0, checkpoint_name, config,
                            snapshot_mode="last", override_prefix=True,
                            use_summary_writer=False):
            runner.train()
        train_wall_time = time.time() - train_started
        checkpoint = Path(log_dir) / "run_0" / "params.pkl"
        if not checkpoint.exists():
            raise RuntimeError(f"checkpoint was not saved: {checkpoint}")
        actual_decision_steps = getattr(runner, "actual_decision_steps", n_steps)
        progress_path = Path(log_dir) / "run_0" / "progress.csv"
        final_checkpoint = checkpoint
        latest_checkpoint = (
            Path(checkpoint_dir) / "latest_before_shutdown.pkl"
            if checkpoint_dir else None
        )
        if latest_checkpoint is not None and latest_checkpoint.exists():
            final_checkpoint = latest_checkpoint
        final_metadata = checkpoint_metadata(
            final_checkpoint,
            progress_path=progress_path,
            optimizer_updates=algo.real_updates,
        )
        if final_metadata["itr"] != actual_decision_steps - 1:
            raise RuntimeError(
                "final checkpoint does not match the runner boundary: "
                f"itr={final_metadata['itr']} actual_decision_steps={actual_decision_steps}"
            )
        boundary = progress_boundary_metadata(progress_path, actual_decision_steps)
        if checkpoint_name != "debug_gate" and not (ROOT / "initial_policy.pkl").exists():
            raise RuntimeError("formal pre-update checkpoint was not saved")
        changes = algo.parameter_change()
        summary_path = ROOT / ("debug_summary.json" if checkpoint_name == "debug_gate" else "train_summary.json")
        write_json(summary_path, {
            "real_updates": algo.real_updates,
            "parameter_changes": changes,
            "finite_update_metrics": all(np.isfinite(list(r.values())[1:]).all() for r in algo.update_records if len(r) > 1),
            "checkpoint": str(final_checkpoint),
            "checkpoint_metadata": final_metadata,
            "progress_boundary": boundary,
            "actual_decision_steps": actual_decision_steps,
            "actual_environment_steps": actual_decision_steps * ACTION_REPEAT,
            "stopped_for_wall_clock": bool(getattr(runner, "deadline_stop", False)),
            "training_wall_time_sec": train_wall_time,
        })
        for name, item in changes.items():
            if not item["changed"]:
                raise RuntimeError(f"{name} parameters did not change")
        if algo.real_updates < 10:
            raise RuntimeError(f"only {algo.real_updates} real updates completed")
        return final_checkpoint


def evaluate_checkpoint(upstream: Path, checkpoint: Path, seeds: list[int], policy_name: str,
                        video_dir: Path | None = None,
                        video_seeds: set[int] | None = None) -> list[dict]:
    """Evaluate one policy with an explicit RNG/env reset per episode.

    In eval mode the pinned tanh-normal actor uses its mode, while the RSSM
    posterior still updates recurrent state from each pixel observation. The
    state is reset once at every episode boundary and the previous action is
    reset to the environment action-space null value.
    """
    torch, make_env = import_upstream(upstream)
    from dreamer.agents.dmc_dreamer_agent import DMCDreamerAgent
    rows = []
    for seed in seeds:
        seed_all(seed)
        env = make_env(seed=seed)
        try:
            if policy_name == "random":
                agent = None
            else:
                params = torch.load(checkpoint, map_location="cpu")
                agent = DMCDreamerAgent(train_noise=0.0, eval_noise=0.0,
                                        expl_type="additive_gaussian", expl_min=None,
                                        expl_decay=None)
                agent.initialize(env.spaces)
                agent.load_state_dict(params["agent_state_dict"])
                agent.to_device(0)
                agent.eval_mode(0)
            obs = env.reset()
            if agent is not None:
                agent.reset()
            prev_action = env.action_space.null_value()
            total = 0.0
            decisions = 0
            frames = []
            save_video = video_dir is not None and video_seeds is not None and seed in video_seeds and policy_name == "dreamer"
            if save_video:
                frames.append(np.transpose(obs, (1, 2, 0)))
            while decisions < EPISODE_DECISION_STEPS:
                if agent is None:
                    action = env.action_space.sample()
                else:
                    obs_t = torch.from_numpy(obs[None]).to(agent.device)
                    action_t = torch.from_numpy(prev_action[None]).to(agent.device)
                    reward_t = torch.zeros(1, device=agent.device)
                    step = agent.step(obs_t, action_t, reward_t)
                    action = step.action.detach().cpu().numpy()[0]
                obs, reward, done, info = env.step(action)
                total += float(reward)
                decisions += 1
                if save_video:
                    frames.append(np.transpose(obs, (1, 2, 0)))
                if done or getattr(info, "traj_done", False):
                    break
                prev_action = action
            row = {"policy": policy_name, "seed": seed, "return": total,
                   "decision_steps": decisions, "env_steps": decisions * ACTION_REPEAT}
            if save_video:
                video_dir.mkdir(parents=True, exist_ok=True)
                mp4_path = video_dir / f"{policy_name}_seed_{seed}.mp4"
                try:
                    import imageio.v2 as imageio
                    imageio.mimsave(mp4_path, frames, fps=20, macro_block_size=1)
                    row["video"] = str(mp4_path)
                except Exception as exc:
                    gif_path = video_dir / f"{policy_name}_seed_{seed}.gif"
                    import imageio.v2 as imageio
                    imageio.mimsave(gif_path, frames, duration=0.05, loop=0)
                    row["video"] = str(gif_path)
                    row["video_fallback"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        finally:
            env.close()
    return rows


EVAL_FIELDS = [
    "policy", "seed", "return", "decision_steps", "env_steps",
    "video", "video_fallback",
]


def _read_eval_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_eval_row(output_dir: Path, row: dict, rows: list[dict]) -> None:
    """Persist one episode before allowing the next episode to run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "eval_episodes.csv"
    jsonl_path = output_dir / "eval_episodes.jsonl"
    status_path = output_dir / "eval_status.json"
    serializable = {field: row.get(field, "") for field in EVAL_FIELDS}
    if csv_path.exists() and csv_path.stat().st_size:
        with csv_path.open("r", encoding="utf-8") as handle:
            current_fields = next(csv.reader(handle), [])
        if current_fields != EVAL_FIELDS:
            tmp_path = csv_path.with_suffix(".csv.tmp")
            with tmp_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=EVAL_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            os.replace(tmp_path, csv_path)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVAL_FIELDS, extrasaction="ignore")
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(serializable)
    write_jsonl(jsonl_path, serializable)
    rows.append(serializable)
    write_json(status_path, {
        "status": "in_progress",
        "completed_episodes": len(rows),
        "completed_keys": sorted(f"{item.get('policy')}:{item.get('seed')}" for item in rows),
        "last_completed": serializable,
    })


def resumable_evaluate_checkpoint(upstream: Path, checkpoint: Path, seeds: list[int],
                                  output_dir: Path, initial_checkpoint: Path | None = None,
                                  include_random: bool = True,
                                  include_initial: bool = True,
                                  video_dir: Path | None = None) -> list[dict]:
    """Evaluate policies with independent per-episode persistence and deduplication."""
    existing = _read_eval_rows(output_dir / "eval_episodes.csv")
    completed = {(row.get("policy"), str(row.get("seed"))) for row in existing}
    rows = list(existing)

    jobs = []
    if include_random:
        jobs.append(("random", checkpoint, False))
    if include_initial and initial_checkpoint is not None and initial_checkpoint.exists():
        jobs.append(("initial", initial_checkpoint, False))
    jobs.append(("dreamer", checkpoint, True))
    for policy_name, policy_checkpoint, with_video in jobs:
        pending = [seed for seed in seeds if (policy_name, str(seed)) not in completed]
        if not pending:
            continue
        episode_rows = evaluate_checkpoint(
            upstream, policy_checkpoint, pending, policy_name,
            video_dir=video_dir if with_video else None,
            video_seeds={1000, 1001, 1002} if with_video else None,
        )
        for row in episode_rows:
            key = (row.get("policy"), str(row.get("seed")))
            if key in completed:
                continue
            _append_eval_row(output_dir, row, rows)
            completed.add(key)
    write_json(output_dir / "eval_status.json", {
        "status": "completed",
        "completed_episodes": len(rows),
        "completed_keys": sorted(f"{item.get('policy')}:{item.get('seed')}" for item in rows),
    })
    return rows


def save_initial_policy(upstream: Path, seed: int) -> Path:
    torch, make_env = import_upstream(upstream)
    from dreamer.agents.dmc_dreamer_agent import DMCDreamerAgent
    env = make_env(seed=seed)
    try:
        agent = DMCDreamerAgent(train_noise=0.0, eval_noise=0.0,
                                expl_type="additive_gaussian", expl_min=None,
                                expl_decay=None)
        agent.initialize(env.spaces)
        agent.to_device(0)
        agent.eval_mode(0)
        path = ROOT / "initial_policy.pkl"
        torch.save({"itr": 0, "cum_steps": 0,
                    "agent_state_dict": agent.state_dict()}, path)
        return path
    finally:
        env.close()


def eval_and_export(upstream: Path, checkpoint: Path, initial_checkpoint: Path | None = None,
                    video_dir: Path | None = None, output_dir: Path | None = None,
                    progress_path: Path | None = None, include_random: bool = True,
                    include_initial: bool = True) -> None:
    eval_started = time.time()
    output_dir = output_dir or ROOT
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1000, 1010))
    initial_path = (
        initial_checkpoint
        if initial_checkpoint is not None and initial_checkpoint.exists()
        else None
    )
    rows = resumable_evaluate_checkpoint(
        upstream, checkpoint, seeds, output_dir,
        initial_checkpoint=initial_path,
        include_random=include_random,
        include_initial=include_initial,
        video_dir=video_dir,
    )
    checkpoint_info = checkpoint_metadata(checkpoint, progress_path=progress_path)
    progress_info = (
        progress_boundary_metadata(progress_path, checkpoint_info["decision_steps"])
        if progress_path is not None and checkpoint_info["decision_steps"] is not None
        else None
    )
    initial_info = (
        checkpoint_metadata(initial_path, progress_path=progress_path)
        if initial_path is not None else None
    )
    summary = {
        "status": "bounded_pytorch_pilot",
        "task": "cartpole_swingup", "seeds": seeds,
        "checkpoint": str(checkpoint),
        "initial_checkpoint": str(initial_path) if initial_path is not None else None,
        "evaluated_checkpoint": checkpoint_info,
        "evaluated_initial_checkpoint": initial_info,
        "evaluation_protocol": {
            "environment_seeds": seeds,
            "action_repeat": ACTION_REPEAT,
            "episode_decision_steps": EPISODE_DECISION_STEPS,
            "episode_environment_steps": EPISODE_DECISION_STEPS * ACTION_REPEAT,
            "eval_noise": 0.0,
            "recurrent_state_reset_per_episode": True,
        },
        "random_mean_return": (
            float(np.mean([float(r["return"]) for r in rows if r["policy"] == "random"]))
            if include_random else None
        ),
        "random_sample_std_return": (
            float(np.std([float(r["return"]) for r in rows if r["policy"] == "random"], ddof=1))
            if include_random else None
        ),
        "dreamer_mean_return": float(np.mean([float(r["return"]) for r in rows if r["policy"] == "dreamer"])),
        "dreamer_sample_std_return": float(np.std([float(r["return"]) for r in rows if r["policy"] == "dreamer"], ddof=1)),
        "evaluation_wall_time_sec": time.time() - eval_started,
        "evaluated_checkpoint_progress_boundary": progress_info,
        "paper_reproduction_claim": False,
    }
    if include_initial and initial_path is not None:
        initial_returns = [float(r["return"]) for r in rows if r["policy"] == "initial"]
        summary["initial_mean_return"] = float(np.mean(initial_returns))
        summary["initial_sample_std_return"] = float(np.std(initial_returns, ddof=1))
    write_json(output_dir / "summary.json", summary)


def resume_audit(checkpoint: Path, initial_checkpoint: Path | None = None) -> None:
    """Report what the pinned snapshot contains without claiming full resume."""
    import torch
    records = []
    for label, path in (("final", checkpoint), ("initial", initial_checkpoint)):
        if path is None or not path.exists():
            records.append({"label": label, "path": str(path) if path else None, "exists": False})
            continue
        payload = torch.load(path, map_location="cpu")
        keys = sorted(payload.keys()) if isinstance(payload, dict) else []
        records.append({
            "label": label,
            "path": str(path),
            "exists": True,
            "keys": keys,
            "has_agent_state_dict": "agent_state_dict" in payload,
            "has_optimizer_state_dict": "optimizer_state_dict" in payload,
            "has_replay": any("replay" in k.lower() for k in keys),
            "has_rng_state": any("rng" in k.lower() or "random" in k.lower() for k in keys),
            "itr": payload.get("itr") if isinstance(payload, dict) else None,
            "cum_steps": payload.get("cum_steps") if isinstance(payload, dict) else None,
        })
    write_json(ROOT / "resume_audit.json", {
        "classification": "model_and_optimizer_only; replay_rng_environment_state_not_verified",
        "records": records,
    })
    log(json.dumps(records, sort_keys=True))


def export_bundle() -> Path:
    path = ROOT / "dreamer_pilot_artifacts.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in ROOT.rglob("*"):
            if item.is_file() and item != path:
                archive.write(item, item.relative_to(ROOT))
    log(f"export_bundle={path} bytes={path.stat().st_size}")
    return path


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["persistence_probe", "preflight", "initial_eval", "debug", "train", "train_long", "resume_audit", "eval", "export", "export_minimal", "all"], required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--max-train-minutes", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--progress-path", type=Path)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--skip-initial", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    started = time.time()
    if args.phase == "persistence_probe":
        persistence_probe()
        log(f"phase={args.phase} elapsed_sec={time.time() - started:.1f}")
        return
    upstream = ensure_upstream()
    if args.phase in ("preflight", "debug", "train", "train_long", "all"):
        preflight(upstream, args.seed)
    checkpoint = args.checkpoint
    initial_checkpoint = ROOT / "initial_policy.pkl"
    if args.phase in ("initial_eval", "all"):
        initial_checkpoint = save_initial_policy(upstream, args.seed)
        rows = evaluate_checkpoint(upstream, initial_checkpoint, list(range(1000, 1010)), "initial")
        with (ROOT / "initial_eval_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        write_json(ROOT / "initial_eval_summary.json", {
            "policy": "initial_untrained_policy",
            "label": "training-before baseline",
            "mean_return": float(np.mean([r["return"] for r in rows])),
            "sample_std_return": float(np.std([r["return"] for r in rows], ddof=1)),
            "checkpoint": str(initial_checkpoint),
        })
        export_bundle()
    if args.phase in ("debug", "all"):
        checkpoint = CountingDreamer.build(
            upstream, import_upstream(upstream)[1], args.seed,
            batch_size=2, batch_length=10, prefill=60, train_every=1, train_steps=1,
            n_steps=70, log_dir=ROOT / "debug_run", checkpoint_name="debug_gate",
        )
        subprocess.run([sys.executable, __file__, "--phase", "eval",
                        "--checkpoint", str(checkpoint)], check=True)
    if args.phase in ("train", "train_long"):
        long_run = args.phase == "train_long"
        checkpoint_steps = [50000, 100000, 150000, 250000] if long_run else []
        run_name = (
            f"cartpole_swingup_seed{args.seed}_long_{args.n_steps * ACTION_REPEAT}_env"
            if long_run else "cartpole_swingup_baseline_100k_env"
        )
        run_root = ROOT / ("long_train_run" if long_run else "train_run")
        checkpoint = CountingDreamer.build(
            upstream, import_upstream(upstream)[1], args.seed,
            batch_size=16, batch_length=50, prefill=1000,
            train_every=UPSTREAM_TRAIN_EVERY, train_steps=UPSTREAM_TRAIN_STEPS,
            n_steps=args.n_steps, log_dir=run_root,
            checkpoint_name=run_name,
            replay_capacity=60000 if long_run else FORMAL_REPLAY_CAPACITY,
            log_interval_steps=1000,
            disable_anomaly=True, disable_progress=True,
            max_train_seconds=(args.max_train_minutes * 60 if args.max_train_minutes else None),
            checkpoint_dir=ROOT / "checkpoints" if long_run else None,
            checkpoint_decision_steps=checkpoint_steps,
        )
        subprocess.run([sys.executable, __file__, "--phase", "eval",
                        "--checkpoint", str(checkpoint),
                        "--initial-checkpoint", str(initial_checkpoint)], check=True)
    if args.phase == "resume_audit":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for resume_audit")
        resume_audit(checkpoint, args.initial_checkpoint)
    if args.phase == "eval":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for eval")
        output_dir = args.output_dir or ROOT
        video_dir = None if args.no_video else output_dir / "videos"
        eval_and_export(
            upstream,
            checkpoint,
            args.initial_checkpoint if args.initial_checkpoint is not None
            else (initial_checkpoint if initial_checkpoint.exists() else None),
            video_dir=video_dir,
            output_dir=output_dir,
            progress_path=args.progress_path,
            include_random=not args.skip_random,
            include_initial=not args.skip_initial,
        )
    if args.phase == "export":
        export_bundle()
    if args.phase == "export_minimal":
        export_minimal_bundle()
    log(f"phase={args.phase} elapsed_sec={time.time() - started:.1f}")


if __name__ == "__main__":
    phase = sys.argv[sys.argv.index("--phase") + 1] if "--phase" in sys.argv else "unknown"
    started = time.time()
    try:
        _main()
    except Exception as exc:
        write_json(ROOT / f"status_{phase}.json", {
            "run_id": RUN_ID, "phase": phase, "status": "failed",
            "return_code": 1, "elapsed_sec": time.time() - started,
            "error_type": type(exc).__name__, "error": str(exc),
        })
        raise
    else:
        write_json(ROOT / f"status_{phase}.json", {
            "run_id": RUN_ID, "phase": phase, "status": "completed",
            "return_code": 0, "elapsed_sec": time.time() - started,
        })
