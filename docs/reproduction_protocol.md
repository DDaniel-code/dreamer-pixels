# Reproduction protocol

## 目标

先以 `cartpole_swingup` 建立像素输入、训练入口、评估和日志链路；随后扩展到 `walker_walk` 与 `ball_in_cup_catch`。论文主比较对象是 Appendix G 的连续控制分数和 Figure 12 的 action-repeat 敏感性。第一阶段不把任何本地运行结果当作论文复现结果。

论文报告的参考值包括：`Cartpole Swingup` Dreamer 为 833.66，连续控制平均值为 823.39；这些是源材料中的参考，不是本项目测得的结果。

## 固定设置

| 项目 | 约定 | 状态 |
|---|---|---|
| 观测 | 64×64×3 RGB pixels | 论文确认；运行待验证 |
| action repeat | R=2 | 论文确认；wrapper 语义待验证 |
| episode | 1000 environment steps | 论文确认 |
| 决策步 | action repeat 后的 agent decisions | 需用实际 wrapper 核对 |
| 训练预算 | 目标 1,000,000 environment steps | 论文/图表口径；未运行 |
| 任务 | 首先 cartpole_swingup | 已确定 |
| 随机种子 | 记录每个 seed；具体 seed 集合 TBD | 未确定 |
| 评价 | episodic return、learning curve | 已确定 |
| 其他指标 | 不使用 classification accuracy 描述控制表现 | 已确定 |

## 训练与评估协议

正式复现前，先在 Linux GPU 上做依赖检查、随机策略像素检查、极短 debug run，再做小预算预检。正式实验应记录源码 commit、配置、环境版本、GPU、seed、环境步数和日志路径。评估应使用无探索噪声或上游 test planner，并固定评估 episode 数量；具体数量在读取并运行上游接口后补齐，当前不编造。

## 尚未验证

当前没有任何训练、像素渲染或论文结果复现通过记录。

## 尚未确认

- 论文附录中的全部默认超参数与当前上游配置的逐项差异。
- dm_control 版本、MuJoCo 渲染后端和 Linux CUDA 组合。
- 训练日志中 environment steps 与 decision steps 的精确计数位置。
- 学校 GPU 小时实际批准额度及作业调度方式。
