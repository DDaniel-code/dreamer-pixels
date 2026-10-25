# Seed-0 300k checkpoint evaluation protocol

- Run: `seed0_overnight_20261024_01`.
- Checkpoint: `decision_150000_env_300000.pkl` from the overnight run's periodic checkpoint directory.
- Task: DeepMind Control Suite `cartpole_swingup`.
- Observation: camera 0, 64x64 RGB pixels.
- Action repeat: 2; 500 decision steps / 1,000 environment steps per episode.
- Evaluation environment seeds: 1000--1009, one episode per seed.
- Policy: Dreamer checkpoint only. This stage evaluation notebook intentionally skipped random and training-initial controls.
- Each row in `eval_episodes.csv` is an actual row from the downloaded Kaggle `stage_evaluation_returns.csv`, filtered to `checkpoint_env_steps=300000`.
- The aggregate source file is retained as `source_stage_evaluation_returns.csv`.
- The 300k evaluation was not concatenated with the old 100k run or with the 330,732-step shutdown result.

The available downloaded artifact was the aggregate stage CSV rather than a checkpoint-specific `summary.json`. The summary in this directory is therefore a derived, independently recomputed summary of the ten downloaded 300k rows; it does not claim that a checkpoint-specific Kaggle summary was downloaded.
