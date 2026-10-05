# Paper to code map

以下路径对应固定的 `google-research/dreamer` commit；本仓库暂不复制上游代码。

| 概念 | 上游代码位置 | 作用 |
|---|---|---|
| RSSM | `dreamer/models/rssm.py` | 用 recurrent belief 和 stochastic state 表示潜在动力学，并提供 posterior/transition。 |
| World model | `dreamer/training/define_model.py`, `dreamer/models/` | 编码像素、预测潜状态、重建图像并预测 reward。 |
| Actor | `dreamer/control/planning.py`, `dreamer/scripts/configs.py` | 在 imagined latent states 上产生连续动作；训练时采样，测试时取 mode。 |
| Critic | `dreamer/scripts/configs.py`, `dreamer/training/` | 估计 imagined trajectory 的 value，帮助超越 imagination horizon。 |
| Replay buffer | `dreamer/tools/numpy_episodes.py` | 保存 episode 数据并按窗口加载训练序列。 |
| Imagination | `dreamer/control/planning.py`, `dreamer/training/trainer.py` | 从 posterior state 出发，只用 learned transition 展开未来，再更新 actor/value。 |
| 训练入口 | `dreamer/scripts/train.py` | 解析 logdir/params 并启动训练流程。 |

这些映射来自仓库文件树和配置源码；待 Linux 环境可获取源码后，再补充实际函数名与行号，避免在未下载代码时臆造位置。
