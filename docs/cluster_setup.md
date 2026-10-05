# Cluster setup checklist

本阶段不连接服务器、不执行 SSH、`sbatch` 或 `srun`，也不假定存在 Slurm。

向学校确认最少信息：

- SSH 主机、用户名，以及是否需要 VPN、跳板机或额外连接步骤；
- SSH 登录节点是否只是登录节点，GPU 是否需要另行分配；
- GPU 是直接使用还是必须通过调度器；
- 是否使用 Slurm；若使用，确认 account、partition、GPU 请求方式；
- Python/Conda/module/容器的推荐方式；
- CUDA、NVIDIA 驱动和 MuJoCo/EGL 环境；
- 项目目录、scratch 路径、配额和 checkpoint 存储规则；
- 节点上的联网、数据传输和日志保留限制。

## 计划中的无调度器接口

- `scripts/env_check.py`：只检查本机/节点环境，不训练。
- 上游 `python3 -m dreamer.scripts.train ...`：短 debug 和正式训练入口，待依赖安装后验证。
- 上游 logdir：保存 episode、checkpoint 和指标；正式路径待集群确认。
- 评估：使用上游 test planner 和独立评估 episode；命令待源码和依赖确认。
- 恢复：使用上游 checkpoint/logdir 机制；具体 flag 待运行核对。

## Slurm（可选、未验证）

不填写 account、partition 和 GPU 请求参数时不得提交任务。本项目不会保存带占位符的可执行提交脚本。
