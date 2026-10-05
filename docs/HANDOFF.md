# Handoff

## 已完成

- 建立本地项目目录和 Git 忽略规则。
- 核查论文、原始作者实现、TensorFlow 2 参考实现和 dm_control。
- 固定原始 Dreamer V1 上游 commit：`d517542f9777c53864ff6f2683cea88be9ea98ac`。
- 固定首个任务 `cartpole_swingup`，记录像素、action repeat 和环境步数口径。
- 加入无依赖环境检查和项目元数据测试。

## 已验证

- 当前主机：Apple Silicon macOS，Python 3.14.6。
- 项目文档存在性测试和“未伪造结果”检查：已通过（`/opt/anaconda3/bin/python3 -m pytest -q`）。
- 环境清单脚本已运行；使用 `/opt/anaconda3/bin/python3` 得到 Python 3.14.6，默认 `/usr/bin/python3` 为 Python 3.9.6。
- 依赖探测：TensorFlow、TensorFlow Probability、dm_control、dm_env 均未安装；因此未运行像素环境或训练入口。

## 未验证

- TensorFlow 1.13.1、TensorFlow Probability 0.6.0 与当前 macOS/Python 的兼容性。
- 默认 `python3` 与 Conda Python 的选择差异，尚未通过项目级环境文件统一。
- dm_control/MuJoCo 像素渲染、随机策略短视频和任何训练结果。
- Linux GPU 依赖、CUDA、EGL、checkpoint 恢复和正式评估。
- GitHub 账户身份、仓库创建、push 和 PR。

## 阻塞项

- 本地没有 `gh` CLI，Shell 网络无法解析 GitHub；无法安全确认 GitHub 账户或创建 private repository。
- 学校 GPU 权限、SSH 主机和调度系统尚未确认。

## 下一步

1. 用户在浏览器完成 GitHub 登录/授权，确认账户身份。
2. 获取 Linux GPU 环境信息后安装并固定上游依赖。
3. 先做短 debug run，再做 cartpole_swingup 小规模预检。
4. 通过后在 `codex/bootstrap` 分支推进正式复现入口，并创建 PR；暂不合并。
