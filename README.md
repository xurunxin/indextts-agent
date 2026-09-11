# IndexTTS Agent CLI

面向 agent 的 IndexTTS-2.5 工具。轻量 CLI + HTTP API + 常驻模型 + SQLite 任务队列。同一 CLI 可连接 Windows 本地进程、Linux GPU 服务器或 Docker；远程客户端不需要 CUDA、PyTorch或模型权重。

方案比较和依据见 [实现调研](docs/implementation-research.md)，实测记录见 [验收说明](docs/validation.md)。

可迁移文件：`dist/indextts_agent-0.1.0-py3-none-any.whl` 用于纯客户端；`dist/indextts-agent-0.1.0-source.zip` 解压到远程GPU机器后运行bootstrap。两者均不包含密钥、模型权重或本地任务数据。运行 `python scripts/package.py` 可重新生成。

## 安装与本地启动

使用 `itt init` 将官方源码、依赖和固定模型安装到状态目录下的隔离运行时；它不会接管本目录现有的 `upstream/.venv` 或 `upstream/checkpoints`。

```powershell
Set-Location G:\Projects\AIGC\_Tools\IndexTTS2.5
itt init --check
itt init
itt service start --wait 600
itt service status
itt capabilities
```

`init --check` 只读检查源码、运行环境、模型清单和 CUDA，并返回计划路径。首次部署固定使用 GitHub 源码 ZIP，不需要 Git；默认目录为 `<home>/runtime`。可以先用 `--dry-run` 查看计划，使用 `--install-dir DIR` 改变运行时位置。默认初始化不会启动 GPU 服务，完整模型准备好后可显式使用 `--start --wait 600`。

```powershell
itt init --dry-run --install-dir D:\IndexTTS-runtime
itt init --install-dir D:\IndexTTS-runtime --start --wait 600
```

固定锁使用 CUDA PyTorch。当前版本对 `--device cpu` 明确报错并且不会创建任何文件；不会自动安装或更新 NVIDIA 驱动。缺少 uv 时自动从官方 PyPI 平台 wheel 下载独立二进制并校验 SHA-256。模型下载使用单独的 Hugging Face Hub 1.10.1 环境，避免推理环境的旧版网络依赖影响大文件下载；不修改上游固定推理依赖，也不依赖全局 `hf` 命令。使用固定 revision、专用缓存和断点重试。`--skip-models` 只准备源码和依赖，写入的 `runtime.json` 会保持 `ready=false`。

没有 Python、pip、uv 的 Windows 机器，可在源码包解压目录执行：

```powershell
pwsh.exe -NoProfile -File .\scripts\install.ps1 -InstallDir 'G:\TTS\itt-runtime'
```

该入口先装轻量 CLI，再调用 `itt init`。`-Package <wheel路径>` 可指定随身携带的 CLI wheel；依赖和模型仍需联网。无需管理员权限，不修改系统 Python；脚本最后打印 CLI 的绝对路径。uv 和 Python 安装依据见 [uv 官方文档](https://docs.astral.sh/uv/getting-started/installation/)。

初始化过程输出 NDJSON 阶段事件，详细日志实时写入 `<home>/init.log`；`init-state.json` 记录最近阶段。下载完成后按固定版本远端清单核验必要模型文件长度，保存本地 manifest；后续检查与完整模型复用不再联网。修复网络后可重新执行同一 init 命令。不要在此受管环境的服务运行时重新安装，先用 `itt service stop` 停止服务。旧环境不会被删除。

在另一台部署机首次安装可以直接安装本 CLI wheel，然后执行 `itt init`；若仍使用源码包，旧入口也转发到同一初始化流程：

```sh
python scripts/bootstrap.py --skip-models
# 等价于：itt init --skip-models
```

bootstrap 现在只是兼容入口，初始化固定官方源码和主/辅助模型版本，运行 `uv sync --frozen --no-dev --python 3.11`，追加服务依赖，不装 Gradio、DeepSpeed 或 FlashAttention。服务器需要能容纳官方推理模型的 GPU（模型卡约 6GB 显存；8GB 卡需给桌面程序留余量）、足够系统内存和磁盘空间。初次运行还可能下载辅助文本规范化资源。

所有全局选项放在子命令之前。默认状态目录为 `~/.indextts-agent`；可用 `--home DIR` 或 `ITT_HOME` 改变。配置、素材、SQLite、服务日志和输出均在状态目录，初始化配置写入 `runtime.json`，模型在 `<home>/runtime/models`。
初始化成功会把 `runtime_python`、`upstream`、`model_dir` 和 `device` 写入 `runtime.json`；`service start`、`serve` 与 `doctor` 会自动使用这些值。对应的显式参数优先，其次是 `ITT_RUNTIME_PYTHON`、`ITT_UPSTREAM`、`ITT_MODEL_DIR`、`ITT_DEVICE` 环境变量，最后才使用旧版 `upstream/.venv` 和 `upstream/checkpoints` 默认路径。

## 为 agent 生成语音

先注册一段干净人声。官方示例只用于验证，生产使用你有权使用的录音。

```powershell
itt voices add narrator --audio upstream/examples/voice_01.wav
itt jobs submit --voice narrator --text '太好了，我们终于成功了！' --emotion happy=0.6 --idempotency-key scene01-v1
itt jobs watch JOB_ID --wait 300
itt jobs download JOB_ID --output outputs/scene01.wav
```

提交后立即返回任务ID。`watch`只输出变化的状态，使用NDJSON；中断监控不会取消生成。也可以一步等待并下载：

```powershell
itt jobs submit --voice narrator --text '欢迎来到我们的世界。' --emotion happy=0.4,calm=0.2 --wait 300 --output outputs/welcome.wav --idempotency-key welcome-v1
```

若不注册音色，使用 `--speaker-audio local.wav` 自动上传，或 `--speaker ASSET_ID` 复用上传素材。音色和素材ID属于当前服务器。所有请求经HTTP上传，远程服务器不需要访问客户端目录。

Agent调用顺序：`service status`确认ready → `capabilities`取得当前Schema → 上传/注册素材 → 提交并保存ID与幂等键 → `jobs get/watch` → 成功后`download`。

```powershell
itt jobs list --limit 20
itt jobs get JOB_ID
itt jobs cancel JOB_ID
itt service logs --tail 50
itt service stop --wait 300
```

重试同一请求时复用幂等键。同键同请求返回原任务，改变请求则报`IDEMPOTENCY_CONFLICT`。`SUBMISSION_UNKNOWN`表示网络未确认提交结果，应使用错误响应里的幂等键重试。失败/取消/中断的任务不会自动重做，明确要重做时换键。

## 情绪、语速和发音

情绪来源每次选一种：

| 控制 | 示例 | 说明 |
|---|---|---|
| 八维混合 | `--emotion sad=0.6,calm=0.1` | 无额外情绪语言模型开销 |
| 八维数组 | `--emotion-vector 0.6,0,0,0,0,0,0,0.2` | happy, angry, sad, afraid, disgusted, melancholic, surprised, calm |
| 独立情绪录音 | `--emotion-audio emotion.wav --emotion-alpha 0.7` | 以speaker参考保持音色，从另一段音频取表达方式 |
| 文字指导 | `--emotion-text '压低声音，带着悲伤' --emotion-alpha 0.6` | 服务需`--qwen-emotion` |
| 从台词推断 | `--emotion-auto --emotion-alpha 0.6` | 服务需`--qwen-emotion` |

向量各值0..1且总和<=1，建议总强度0.6..0.8。`--emotion-random`会降低音色还原度，默认关闭。开启文字情绪会加载额外QwenEmotion，建议在显存更充裕的部署机使用。

```powershell
itt service stop --wait 300
itt service start --qwen-emotion --wait 600
itt jobs submit --voice narrator --text '我们终于等到这一天了。' --emotion-text '激动又有些哽咽' --emotion-alpha 0.6
```

语速 `--speed 1.25` 表示更快；底层映射 `duration_factor=0.8`。直接使用 `--duration-factor 1.25` 则是更慢。两者范围均0.5..2，不是精确的目标秒数，也不通过重采样改变音高。

语言选项为 `--lang ZH/EN/JA/ES/AR`；`ZHEN`是中英混合预处理。保留模型发音标注，例如 `他在银<行|HANG2>工作，喜欢步<行|XING2>回家。`、`<going|G OW1 . IH0 NG>`。长文用UTF-8 `--text-file script.txt`，上游分段生成，`--interval-silence 200`控制段间静音；跨段韵律不连续。

`--seed`默认42，跨设备/版本不保证逐字节一致。`--num-beams`默认3；其他采样/分段参数可用 `itt jobs submit --help` 查看。完整JSON请求用 `--request request.json`，字段由 `itt capabilities` 返回。

## 远程部署、本地调用

在远程GPU机器安装服务环境后，生成一个长随机密钥并通过环境变量设置。监听外网卡时必须提供密钥。以Linux为例：

```sh
export ITT_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
# 将密钥保存到服务器的私有环境配置，再通过安全方式配置客户端。
itt service start --host 0.0.0.0 --port 8095 --wait 600
```

Windows本地客户端仅安装CLI wheel或源码包，无需下载模型：

```powershell
uv tool install .\dist\indextts_agent-0.1.0-py3-none-any.whl
$env:ITT_API_KEY = '<远程服务密钥>'
itt config set --url https://tts.example.com --api-key-env ITT_API_KEY
itt service status
itt voices add narrator --audio D:\Audio\narrator.wav
itt jobs submit --voice narrator --text '这是远程生成的语音。' --wait 300 --output .\remote.wav
```

URL可用 `--url` 或 `ITT_URL` 临时覆盖。`config set`保存URL和密钥环境变量名，不保存远程密钥。推荐HTTPS反代、VPN或SSH隧道；SSH隧道可保持服务监听loopback：

```sh
ssh -N -L 18095:127.0.0.1:8095 user@gpu-host
```

然后本地连接 `http://127.0.0.1:18095` 并设置服务密钥。启动停止属于部署机职责：已经关停的远程HTTP服务无法通过自身HTTP重新启动；在远程机使用CLI、SSH、systemd或Docker启动。当前CLI的 `service start` 启动本机进程/本机Docker；`status`、任务命令与`stop`连接所配置的服务器。换回本地前将URL改为本地服务，并移除远程密钥环境变量，让CLI使用自动生成的本地密钥。

## Docker

部署机需要Linux容器和NVIDIA Container Toolkit/GPU直通。权重挂载为独立目录，不打入镜像。

```sh
docker build -t indextts-agent:local .
export ITT_API_KEY='已有的长随机密钥'
# 默认只发布到127.0.0.1；使用VPN/反代，或明确设置ITT_BIND_IP。
docker compose up -d
docker compose logs -f
docker compose stop
```

环境变量：`ITT_BIND_IP`、`ITT_PORT`、`ITT_DATA_DIR`、`ITT_MODEL_DIR`。Windows PowerShell使用 `$env:变量名='值'` 设置。

也可以让CLI管理一次性Docker服务：

```sh
itt service start --backend docker --image indextts-agent:local --wait 600
itt service stop --wait 300
```

Docker镜像默认使用官方BF16推理，未自动开启编译加速。需要文字情绪时对CLI启动加 `--qwen-emotion`；Compose可覆盖command为 `['--home','/data','serve','--host','0.0.0.0','--upstream','/opt/indextts','--model-dir','/opt/indextts/checkpoints','--qwen-emotion']`。

## 状态、接口与退出码

任务：queued → running → succeeded/failed/cancelled；服务异常退出的running任务重启后为interrupted，queued继续执行。取消在下一模型进度回调或当前推理返回时生效。停止服务会完成当前任务，保留队列；不会杀死其他进程。

HTTP接口：`/v1/status`、`/v1/capabilities`、`/v1/assets`、`/v1/voices`、`/v1/jobs`、`/v1/jobs/{id}`、`/cancel`、`/audio`、`/v1/service/stop`。认证后的 `/v1/openapi.json` 提供完整OpenAPI。公开 `/health`仅报告状态。接口版本为v1。

| 退出码 | 意义 |
|---|---|
| 0 | 命令成功；异步submit成功仅表示入队 |
| 1 | 连接/服务错误，或提交结果未知 |
| 2 | 参数、请求、认证或本地文件错误 |
| 3 | 等待超时；后台生成或启动继续 |
| 4 | 任务失败、取消或中断 |
| 130 | 用户中断监控，后台任务继续 |

结果为JSON，`watch`及`submit --wait`为NDJSON。模型日志写后台日志；普通CLI不在stdout输出模型日志。错误包含稳定 `error.code`。下载校验SHA-256，默认不覆盖已有文件。单素材上限40MiB、0.25..60秒、WAV/FLAC、单/双声道；上游仅使用前15秒。当前为单用户/可信团队服务，不提供多租户隔离、在线计费或自动清理任务。

## 验证和打包

```powershell
uv pip install --python .venv/Scripts/python.exe -e '.[server,test]'
.venv/Scripts/python.exe -m pytest -q
itt doctor
.venv/Scripts/python.exe scripts/acceptance.py
uv build --wheel
```

测试后端需显式 `--engine test`，只生成提示音；它绝不会在真实模型失败时自动替代模型。实际GPU验收脚本拒绝test后端。输出语音及生成参数保存在 `outputs/`。所有生成应遵循[官方模型许可](https://huggingface.co/IndexTeam/IndexTTS-2.5/blob/main/LICENSE)。
# 卸载受管环境

```powershell
itt uninstall --dry-run
itt service stop --wait 600
itt uninstall --yes
```

仅删除 `init` 标记归属的运行环境、上游源码及模型，并清除指向它的 `runtime.json`。保留任务、音色、日志、导出音频、共享 uv/Python/Hugging Face 缓存及全局客户端。用 `--install-dir <目录>` 可清理未完成的受管安装。陌生目录、符号链接/junction 和运行中的服务会被拒绝；请通过轻量全局客户端执行，不要从待卸载环境内运行。

`--dry-run` 只读预览，实际删除需 `--yes`。运行环境移除后，如还需卸载通过 uv 安装的客户端：`uv tool uninstall indextts-agent`。其他包管理器安装的客户端使用对应工具卸载。
