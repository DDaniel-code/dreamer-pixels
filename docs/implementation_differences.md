# Implementation differences

## 论文与上游

论文描述的是 Dreamer V1：RSSM world model 结合图像重建、reward model、actor 和 critic，通过 latent imagination 学习行为。原始仓库是作者提供的 TensorFlow 1 实现，README 明确给出 Ubuntu 18、TensorFlow GPU 1.13.1、TensorFlow Probability 0.6.0 和 dm_control 等依赖。

当前固定上游版本：`google-research/dreamer@d517542f9777c53864ff6f2683cea88be9ea98ac`。该版本用于复现基线，不改写算法。

## 与本项目的差异

1. 本项目新增文档、配置记录和无依赖环境检查脚本；不复制或重新许可上游源代码。
2. 本项目首个任务固定为 `cartpole_swingup`，而作者 README 的示例默认包括 `walker_run`。
3. 本项目区分 macOS 开发检查和 Linux GPU 训练；不为了让 TensorFlow 1 在 macOS 上完整训练而修改算法。
4. 调度脚本不预设 Slurm；若之后需要，会作为明确标记的可选模板加入。
5. 论文结果、上游默认配置和本地测量结果分开记录。

## 正确性风险

原始依赖年代较旧，当前 Python 3.14/macOS ARM 环境很可能不兼容；直接改动依赖或替换为 V2/V3 会破坏复现目标。因此正式运行前应优先建立 Linux GPU 环境并锁定可安装的旧版依赖组合。
