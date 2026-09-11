# 验收记录

日期：2026-09-11。模型为真实IndexTTS-2.5，CUDA/BF16，Windows RTX 3070 8GB，Torch 2.8.0+cu128。使用官方demo参考人声，不涉及用户自定义音色。

## 已执行

- 10项自动化测试通过：请求校验、认证、重复素材、并发幂等、幂等冲突、结果哈希、排队/运行取消、中断恢复、真实TCP客户端与服务端目录隔离、停止时完成当前任务并保留队列。
- TCP测试使用显式test后端验证协议和生命周期；该测试不作为模型质量证据。
- 原生Windows模型服务：后台启动、loading/ready、任务提交、进度、下载、停止、再次启动、历史任务持久化均已执行。
- 下表为另行执行的真实GPU生成；输出均为22.05kHz、PCM16、单声道WAV，并已在客户端下载后校验SHA-256。
- CLI wheel已构建；CLI环境不安装PyTorch。中文help和JSON/NDJSON输出已检查。
- Ruff静态检查通过；bootstrap重复安装通过；轻量CLI向模型Python转交前台serve入口的实际启停通过。
- Docker Compose配置解析通过；Docker镜像构建和容器GPU推理尚未执行。
- 没有指定独立远程GPU主机，因此尚未做跨机器、TLS反代、GPU驱动/容器直通验收；已用真实TCP及独立客户端/服务端目录验证远程协议。

## 实际生成记录

| 场景 | 音频秒数 | 生成耗时秒 | RTF | PyTorch峰值分配MiB |
|---|---:|---:|---:|---:|
| 首条开心向量 | 4.9923 | 6.510 | 1.304 | 5365.9 |
| 相同参考音频热运行 | 4.9923 | 3.551 | 0.711 | 5362.4 |
| 1.25倍语速 | 3.9938 | 3.131 | 0.784 | 5340.9 |
| 悲伤向量 | 6.4203 | 4.218 | 0.657 | 5391.6 |
| 独立悲伤参考录音 | 3.5875 | 2.627 | 0.732 | 5332.3 |
| 拼音发音标注 | 3.3901 | 2.363 | 0.697 | 5328.6 |
| 上游默认3 beams | 4.9110 | 3.488 | 0.710 | 5430.6 |
| Qwen文字情绪指导 | 3.2740 | 6.056 | 1.850 | 6463.0 |
| Qwen从台词自动推断 | 3.0302 | 4.676 | 1.543 | 6457.6 |
| 英文跨语言音色迁移、最终默认3 beams | 4.5421 | 4.882 | 1.075 | 5362.0 |

RTF=生成耗时/音频时长，越低越快；该耗时不含模型首次加载与网络上传，显存为PyTorch分配峰值，不是整个桌面GPU占用。样本规模有限，不是吞吐基准或质量排名。首条和热运行生成的WAV哈希一致；1.25倍速度对应0.8时长倍率，实测时长比约0.800。1与3 beams耗时接近，最终默认保留上游3。

所有记录未捕获模型截断警告。上述验证证明真实模型完成了生成和控制参数传递；尚未做盲听评价、情绪分类器评分或逐字ASR验收，不能据此宣称所有语言、所有情绪或发音标注均达到生产音质。

原始参数、任务ID、音频SHA-256和时间见 `outputs/acceptance.json`、`outputs/text-emotion-acceptance.json`；可试听 `outputs/happy-warm.wav`、`outputs/sad-vector.wav`、`outputs/sad-reference.wav`、`outputs/text-emotion.wav`、`outputs/auto-emotion.wav`。

## 复现

```powershell
itt service start --wait 600
.venv/Scripts/python.exe scripts/acceptance.py --run-id your-new-run
itt service stop --wait 300
itt service start --qwen-emotion --wait 600
.venv/Scripts/python.exe scripts/acceptance.py --text-emotion-only --run-id your-new-run
itt service stop --wait 300
```

同run-id复用任务；需要重新生成时明确换run-id。测试会保留结果，模型测试使用官方示例音色。服务停止后释放本次模型进程持有的显存。
