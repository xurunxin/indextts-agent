# IndexTTS-2.5 实现方案调研

核实日期：2026-09-11。官方源码固定为 `ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c`，主模型固定为 `c39ce5ba981572cb187443877ff559dfb246ce63`。

## 方案选择

| 方案 | 收益 | 本机限制 | 结论 |
|---|---|---|---|
| 每次 CLI 启动官方推理脚本 | 实现简单 | 每条语音重载权重，无持久化任务和远程协议 | 不采用 |
| 官方 Python 推理常驻 + HTTP + SQLite 队列 | 保留完整情绪/语速/发音控制，复用参考音频缓存；Windows/Linux通用 | 每张卡串行推理 | 当前实施 |
| 同一个服务放进 CUDA Docker | 环境隔离，便于迁移Linux GPU服务器 | 镜像较大；当前Docker仅8GB系统内存 | 提供Dockerfile/Compose及CLI启动入口 |
| vLLM-Omni双阶段服务 | 支持OpenAI语音API和生产服务集成 | 官方声明低显存部署配置未验证；当前需要较新的vLLM-Omni | 适合后续在更大显存远程机进行吞吐评估 |

本机核实：Windows、RTX 3070 8GB、驱动616.64；Torch 2.8.0+cu128识别CUDA和BF16。CLI本身使用Python标准库，无PyTorch依赖；远程客户端不需要GPU。模型服务单进程、单模型工作线程，HTTP在模型加载/推理期间仍可查询。SQLite WAL持久化任务；排队任务重启后继续，正在运行的任务重启后标记interrupted，防止无声重复生成。

效率选择：BF16默认启用、GPT KV cache采用上游行为；实际比较num_beams=1与3耗时接近，因此默认保留上游3，仍开放1供其他硬件评估。不默认开启DeepSpeed、FlashAttention或torch.compile，避免在Windows增加编译失败和冷启动负担。音频素材按内容哈希存储，同一参考文件总是同一路径，上游speaker/emotion缓存可命中。多个任务排队而非并行抢占8GB显存。该选择是基于部署条件的判断，不是对所有GPU的性能排名。

## 与模型原生能力的对应

| CLI/API | 官方推理参数 |
|---|---|
| speaker / --voice / --speaker-audio | spk_audio_prompt，声音身份参考 |
| emotion_audio | emo_audio_prompt，独立情绪参考 |
| emotion_vector / --emotion | emo_vector，八维混合 |
| emotion_text / emotion_auto | emo_text + use_emo_text，需use_qwen_emo |
| emotion_alpha | emo_alpha，0..1 |
| emotion_random | use_random，会降低音色还原度 |
| duration_factor / --speed | duration_factor；speed取倒数，无波形重采样 |
| 原样保留文本标注 | `<文字|拼音>`、CMU音素、日语假名 |
| lang | ZH/EN/JA/ES/AR；ZHEN为中英混合预处理模式 |
| max_text_tokens_per_segment / interval_silence | 上游分段与分段静音 |

情绪向量顺序固定为 happy, angry, sad, afraid, disgusted, melancholic, surprised, calm。为避免多源优先级悄悄覆盖用户意图，API拒绝同时指定多个情绪来源。原生混合公式使用 `1 - sum(vector)` 保留参考情绪，因此本工具要求非负向量且总和不大于1，建议0.6..0.8。文本情绪需要额外QwenEmotion模型，服务未加载时明确报错，不做关键词伪装替代。

语速大于1为更快；时长倍率大于1为更慢。时长倍率不是精确秒数控制。长文由上游拆段，段间韵律不连续；进度来自上游阶段回调，是阶段估计，不能当作剩余秒数。取消在下一模型回调或当前推理返回时生效，不能立即抢占正在执行的CUDA算子。

## 部署边界

所有素材先通过HTTP上传，服务只接收内容哈希，不接收客户端文件路径或任意URL；音频输出归服务目录管理，通过下载接口获取并校验SHA-256。客户端与服务端可使用完全不同的目录。远程连接配置仅保存URL与密钥环境变量名。服务API使用Bearer认证，默认绑定loopback；外网卡监听要求显式设置密钥。远程部署使用HTTPS反代、VPN或SSH端口转发。

服务必须由部署机上的CLI、Docker、systemd或SSH启动。已经关闭的远程HTTP服务不能通过自身HTTP端点启动；本地CLI可管理远程任务并请求有序停止。Compose有restart策略时，用部署机的 `docker compose stop` 保持停止状态。

## 官方来源

- [Hugging Face模型卡](https://huggingface.co/IndexTeam/IndexTTS-2.5)：能力、硬件需求、限制、许可。
- [官方仓库](https://github.com/index-tts/index-tts)：安装、Windows与DeepSpeed说明。
- [固定版本推理源码](https://github.com/index-tts/index-tts/blob/ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c/indextts/infer_v2_5.py)：推理签名、缓存、情绪混合与进度回调。
- [vLLM官方2.5部署配方](https://recipes.vllm.ai/IndexTeam/IndexTTS-2.5)：远程推理、双阶段配置和低显存验证范围。

模型遵循官方Bilibili模型许可；代码和模型文件未打包进CLI wheel，bootstrap按固定版本下载。
