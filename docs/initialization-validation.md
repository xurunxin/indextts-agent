# 初始化验收（2026-09-11）

Windows / NVIDIA CUDA 实测从空的 `managed-runtime` 目录安装 Python 3.11、独立环境、固定版本上游源码、主模型和辅助模型；原有 `upstream` 环境和 checkpoints 保留。`itt init --check` 返回全部检查通过、ready=true。

模型下载使用隔离的 huggingface-hub 1.10.1，不修改上游冻结的推理依赖。网络中断后重复初始化复用下载缓存；清单查询有限重试。固定 revision 的真实文件清单不包含旧环境的 pinyin.vocab，因此不把它当作部署必需品。

新环境启动后真实生成任务 `672d80e1d37541559f6e5737ce1a2d2d` 成功，参考音频来自 QVD 新环境生成结果。输出 `outputs/init-managed-check.wav`：4.1448 秒、22.05 kHz、单声道，生成耗时 5.88 秒，峰值 GPU 分配 5383.3 MB，SHA-256 `b7a185d933639b255148b9f17550647537093de66562c352a87abdf16dc233e5`。

首次启动超过 55 秒等待窗口，随后服务正常 ready；推荐 `itt init --start --wait 600`，超时可用 `itt service status` 查询后台加载状态。验收后停止服务释放显存，运行时配置保持可用。

本次未执行全新操作系统、Linux、Docker 或远程 GPU 验收；CPU 初始化明确不支持当前上游 CUDA 锁定环境。系统 NVIDIA 驱动由用户预装。
