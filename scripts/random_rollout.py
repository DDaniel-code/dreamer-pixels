"""Run a short random-policy pixel rollout in DeepMind Control Suite.

This is an environment and observation-pipeline check, not a Dreamer training
entry point. It intentionally keeps the default budget small enough for a
developer laptop and records both environment steps and decision steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="cartpole")
    parser.add_argument("--task", default="swingup")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-env-steps", type=int, default=200)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/random_rollout"),
    )
    return parser


def make_montage(frames: list[np.ndarray], output_path: Path) -> None:
    if not frames:
        return
    columns = min(5, len(frames))
    rows = (len(frames) + columns - 1) // columns
    width, height = frames[0].shape[1], frames[0].shape[0]
    canvas = Image.new("RGB", (columns * width, rows * (height + 18)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, frame in enumerate(frames):
        x = (index % columns) * width
        y = (index // columns) * (height + 18)
        canvas.paste(Image.fromarray(frame), (x, y))
        draw.text((x + 2, y + height), f"t={index}", fill="black")
    canvas.save(output_path)


def run(args: argparse.Namespace) -> dict:
    from dm_control import suite
    from dm_control.suite.wrappers import pixels

    if args.episodes < 1 or args.max_env_steps < 1 or args.action_repeat < 1:
        raise ValueError("episodes, max-env-steps, and action-repeat must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    frames: list[np.ndarray] = []
    episode_returns: list[float] = []
    total_env_steps = 0
    total_decision_steps = 0
    first_observation_shape: list[int] | None = None
    action_shape: list[int] | None = None

    for episode in range(args.episodes):
        env = suite.load(
            args.domain,
            args.task,
            task_kwargs={"random": int(args.seed + episode)},
        )
        env = pixels.Wrapper(
            env,
            pixels_only=True,
            render_kwargs={
                "height": args.height,
                "width": args.width,
                "camera_id": 0,
            },
        )
        time_step = env.reset()
        action_spec = env.action_spec()
        action_shape = list(action_spec.shape)
        observation = np.asarray(time_step.observation["pixels"])
        first_observation_shape = list(observation.shape)
        episode_return = 0.0
        episode_env_steps = 0
        episode_decision_steps = 0

        while episode_env_steps < args.max_env_steps and not time_step.last():
            action = rng.uniform(
                low=action_spec.minimum,
                high=action_spec.maximum,
                size=action_spec.shape,
            ).astype(action_spec.dtype)
            for _ in range(args.action_repeat):
                time_step = env.step(action)
                episode_env_steps += 1
                total_env_steps += 1
                episode_return += float(time_step.reward or 0.0)
                if len(frames) < 12:
                    frames.append(np.asarray(time_step.observation["pixels"]).copy())
                if time_step.last() or episode_env_steps >= args.max_env_steps:
                    break
            episode_decision_steps += 1
            total_decision_steps += 1
        episode_returns.append(episode_return)
        env.close()

    summary = {
        "status": "passed",
        "environment": f"{args.domain}_{args.task}",
        "episodes": args.episodes,
        "seed": args.seed,
        "action_repeat": args.action_repeat,
        "max_env_steps_per_episode": args.max_env_steps,
        "total_environment_steps": total_env_steps,
        "total_decision_steps": total_decision_steps,
        "observation_key": "pixels",
        "observation_shape": first_observation_shape,
        "action_shape": action_shape,
        "episode_returns": episode_returns,
        "mean_return": float(np.mean(episode_returns)),
        "note": "Random policy only; this does not demonstrate learning or paper reproduction.",
    }
    summary_path = args.output_dir / "cartpole_swingup_random_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if frames:
        make_montage(frames, args.output_dir / "cartpole_swingup_random_frames.png")
        imageio.mimsave(
            args.output_dir / "cartpole_swingup_random.gif",
            frames,
            duration=0.1,
            loop=0,
        )
    return summary


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run(args)
    except Exception as exc:  # Keep a precise, machine-readable failure record.
        print(f"random rollout failed: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
