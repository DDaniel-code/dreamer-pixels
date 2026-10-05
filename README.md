# Dreamer from Pixels — CMU 11-785

第一阶段项目：本地准备、Dreamer V1 复现核查和 GPU 集群工作流准备。

## 当前状态

- 已选定论文作者的原始 TensorFlow 1 实现：`google-research/dreamer`，固定到 commit `d517542f9777c53864ff6f2683cea88be9ea98ac`。
- 首个验证任务：DeepMind Control Suite `cartpole_swingup`，像素观测。
- 后续任务：`walker_walk`、`ball_in_cup catch`。
- 当前阶段不连接学校服务器、不提交远程任务、不进行长时间训练。
- 尚未完成 GPU 训练，因此不能声称已复现论文结果。

## 上游来源和许可证

本项目复用上游代码时保留作者署名和 Apache-2.0 许可证。上游源码应按 `docs/paper_to_code.md` 中记录的 commit 获取；本仓库新增的文档、检查脚本和配置由本项目作者维护。

上游：

- 论文：https://arxiv.org/abs/1912.01603
- 原始实现：https://github.com/google-research/dreamer
- TensorFlow 2 参考实现：https://github.com/danijar/dreamer
- 环境：https://github.com/google-deepmind/dm_control

## 本地准备

本机是 Apple Silicon macOS，当前 Python 为 3.14.6。原始实现依赖 TensorFlow 1.13.1，尚未在本机安装或训练；正式运行环境应优先使用 Linux GPU 环境，具体 CUDA、驱动和 Python 版本待学校确认。

可先运行不依赖第三方包的检查：

```bash
python3 scripts/env_check.py
python3 -m pytest -q
```

若后续在 Linux GPU 节点上获取了上游源码，训练入口以作者实现为准：

```bash
python3 -m dreamer.scripts.train --logdir ./logdir/cartpole_swingup \
  --params '{defaults: [dreamer], tasks: [cartpole_swingup]}'
```

该命令尚未在本地或集群验证，不应直接提交长时间任务。

## 文档

- [复现协议](docs/reproduction_protocol.md)
- [实现差异](docs/implementation_differences.md)
- [论文到代码](docs/paper_to_code.md)
- [集群准备清单](docs/cluster_setup.md)
- [交接记录](docs/HANDOFF.md)
