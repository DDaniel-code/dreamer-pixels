# Cluster setup checklist

This phase does not connect to the server or execute SSH, `sbatch`, or `srun`; it also does not assume Slurm exists.

Confirm the following minimum information with the university:

- SSH host, username, and whether a VPN, jump host, or additional connection step is required;
- whether the SSH login node is only a login node and whether GPUs require a separate allocation;
- whether GPUs are used directly or must be requested through a scheduler;
- whether Slurm is used; if so, confirm the account, partition, and GPU request syntax;
- the recommended Python/Conda/module/container workflow;
- the CUDA, NVIDIA driver, and MuJoCo/EGL environment;
- project directory, scratch path, quota, and checkpoint storage rules;
- networking, data-transfer, and log-retention restrictions on compute nodes.

## Planned scheduler-independent interfaces

- `scripts/env_check.py`: checks the local or node environment only; it does not train.
- Upstream `python3 -m dreamer.scripts.train ...`: short debug and formal training entry point, pending dependency installation.
- Upstream logdir: stores episodes, checkpoints, and metrics; the formal path is pending cluster confirmation.
- Evaluation: use the upstream test planner and independent evaluation episodes; the command is pending source and dependency validation.
- Recovery: use the upstream checkpoint/logdir mechanism; exact flags must be checked during execution.

## Slurm (optional, unverified)

Do not submit a job without the account, partition, and GPU request parameters. This project will not store an executable submission script with placeholders.
