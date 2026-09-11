import argparse
import importlib.util
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import quote

from . import __version__
from . import init as init_command
from .client import Client
from .common import (
    EMOTIONS,
    ROOT,
    TERMINAL,
    Failure,
    InstanceLock,
    atomic_json,
    read_json,
)


def emit(data):
    print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise Failure("INVALID_ARGUMENT", message, 2)


def parser():
    p = Parser(prog="itt", description="IndexTTS-2.5 agent CLI：本地/远程模型服务、素材、情绪控制与持久化任务。结果为 JSON，watch 为 NDJSON。")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--home", default=os.getenv("ITT_HOME", str(Path.home() / ".indextts-agent")), help="CLI 配置和本地任务目录；环境变量 ITT_HOME")
    p.add_argument("--url", default=os.getenv("ITT_URL"), help="HTTP(S) 服务地址；也可配置 ITT_URL")
    p.add_argument("--api-key-env", default=None, help="存放 API Key 的环境变量名，默认 ITT_API_KEY；不把密钥放入命令行")
    p.add_argument("--timeout", type=float, default=30, help="单次 HTTP 超时秒数，默认30")
    subs = p.add_subparsers(dest="command", required=True)
    from .uninstall import arguments as uninstall_arguments
    uninstall_arguments(subs)
    c = subs.add_parser("config", help="保存远程连接设置（不保存远程密钥）")
    cs = c.add_subparsers(dest="action", required=True)
    setp = cs.add_parser("set")
    setp.add_argument("--url", dest="config_url", required=True)
    setp.add_argument("--api-key-env", dest="config_key_env", default="ITT_API_KEY")
    cs.add_parser("show")
    i = subs.add_parser("init", help="创建隔离的 IndexTTS 运行环境和固定模型")
    check_group = i.add_mutually_exclusive_group()
    check_group.add_argument("--check", action="store_true", help="只读检查当前或计划中的运行环境")
    check_group.add_argument("--dry-run", action="store_true", help="只输出计划和检查结果，不写文件")
    i.add_argument("--install-dir", type=Path, help="运行时目录，默认 <home>/runtime")
    i.add_argument("--skip-models", action="store_true", help="只安装源码和依赖，不声明模型 ready")
    i.add_argument("--device", choices=["cuda", "cpu"], default=None)
    i.add_argument("--start", action="store_true", help="安装完成后启动现有本地后台服务")
    i.add_argument("--wait", type=float, default=600, metavar="SECONDS")
    d = subs.add_parser("doctor", help="只读检查 CLI、CUDA 和模型文件")
    d.add_argument("--runtime-python", type=Path)
    d.add_argument("--upstream", type=Path, default=None)
    d.add_argument("--model-dir", type=Path, default=None)
    d.add_argument("--device", choices=["cuda", "cpu"], default=None)
    s = subs.add_parser("service", help="启停本机本地进程/Docker；status/stop 可连接远程")
    ss = s.add_subparsers(dest="action", required=True)
    start = ss.add_parser("start", help="启动本机后台服务，默认不等待模型完成加载")
    start.add_argument("--backend", choices=["local", "docker"], default="local")
    start.add_argument("--runtime-python", type=Path, help="已安装模型依赖的 Python；默认 upstream/.venv")
    start.add_argument("--image", default="indextts-agent:local", help="Docker 镜像，需先 build")
    runtime_arguments(start)
    start.add_argument("--wait", type=float, default=0, metavar="SECONDS", help="等待模型就绪；超时保留后台进程")
    ss.add_parser("status")
    stop = ss.add_parser("stop", help="完成当前任务后退出；保留排队任务")
    stop.add_argument("--wait", type=float, default=0, metavar="SECONDS")
    logs = ss.add_parser("logs", help="读取本机启动日志；远程日志在部署机读取")
    logs.add_argument("--tail", type=int, default=60)
    fg = subs.add_parser("serve", help="前台模型服务入口，供部署机/systemd/Docker使用")
    runtime_arguments(fg)
    fg.add_argument("--runtime-python", type=Path, help="转交给安装了模型依赖的 Python，默认 upstream/.venv")
    subs.add_parser("capabilities", help="查询模型特色、支持的控制参数和 JSON Schema")
    a = subs.add_parser("assets", help="上传参考音频供本地或远程服务复用")
    aa = a.add_subparsers(dest="action", required=True)
    aa.add_parser("upload").add_argument("audio", type=Path)
    v = subs.add_parser("voices", help="为已上传参考音频注册可复用音色名")
    vv = v.add_subparsers(dest="action", required=True)
    vv.add_parser("list")
    va = vv.add_parser("add")
    va.add_argument("name")
    va.add_argument("--audio", required=True, type=Path)
    j = subs.add_parser("jobs", help="提交/查询/监控/取消/下载任务")
    jj = j.add_subparsers(dest="action", required=True)
    submit = jj.add_parser("submit", help="异步提交；推荐提供稳定的 --idempotency-key",
                           epilog="示例：itt jobs submit --text '今天真开心！' --voice narrator --emotion happy=0.6 --idempotency-key scene01-v1")
    submit.add_argument("--request", type=Path, help="完整请求 JSON；路径以外的字段见 capabilities")
    tx = submit.add_mutually_exclusive_group()
    tx.add_argument("--text")
    tx.add_argument("--text-file", type=Path, help="UTF-8 文本，保留 <字|拼音> 发音标注")
    sp = submit.add_mutually_exclusive_group()
    sp.add_argument("--speaker", help="已上传素材 ID")
    sp.add_argument("--speaker-audio", type=Path, help="自动上传本地 WAV/FLAC")
    sp.add_argument("--voice", help="已注册音色名")
    submit.add_argument("--lang", choices=["ZH", "EN", "JA", "ES", "AR", "ZHEN"], default=None)
    emo = submit.add_mutually_exclusive_group()
    emo.add_argument("--emotion", help="八维混合：happy=0.6,calm=0.2；总强度<=1，建议0.6..0.8")
    emo.add_argument("--emotion-vector", help="按 happy,angry,sad,afraid,disgusted,melancholic,surprised,calm 的8个逗号分隔浮点数")
    emo.add_argument("--emotion-audio", type=Path, help="独立情绪参考录音；保留 speaker 音色")
    emo.add_argument("--emotion-text", help="自然语言情绪描述；服务需 --qwen-emotion")
    emo.add_argument("--emotion-auto", action="store_true", default=None, help="从朗读文本推断情绪；服务需 --qwen-emotion")
    submit.add_argument("--emotion-alpha", type=float, help="情绪强度0..1；文字情绪建议0.6")
    submit.add_argument("--emotion-random", action="store_true", default=None, help="随机情绪采样，会降低音色还原度")
    speed = submit.add_mutually_exclusive_group()
    speed.add_argument("--speed", type=float, help="0.5..2，>1更快；映射 duration_factor=1/speed")
    speed.add_argument("--duration-factor", type=float, help="0.5..2，>1更慢；模型原生时长倍率，并非精确秒数")
    for name, typ, description in [("seed", int, "随机种子，默认42；跨设备不保证逐字节一致"),
                                   ("num-beams", int, "1..5，默认3遵循上游；可按硬件对比1"),
                                   ("max-text-tokens-per-segment", int, "20..200，默认120"),
                                   ("interval-silence", int, "分段静音毫秒0..3000，默认200"),
                                   ("temperature", float, "采样温度0.1..2，默认0.8"),
                                   ("top-p", float, "0..1，默认0.8"), ("top-k", int, "1..100，默认30"),
                                   ("max-mel-tokens", int, "50..3000，默认1500")]:
        submit.add_argument("--" + name, type=typ, help=description)
    submit.add_argument("--no-text-normalization", action="store_true", help="关闭模型文本规范化")
    submit.add_argument("--idempotency-key", help="同键同请求返回原任务；不同请求报冲突，不自动重做失败任务")
    submit.add_argument("--wait", type=float, default=0, metavar="SECONDS")
    submit.add_argument("--output", type=Path, help="等待成功后下载的 WAV 路径，需 --wait")
    li = jj.add_parser("list")
    li.add_argument("--limit", type=int, default=50)
    for action in ("get", "watch", "cancel", "download"):
        item = jj.add_parser(action)
        item.add_argument("job_id")
        if action == "watch":
            item.add_argument("--wait", type=float, default=600)
            item.add_argument("--interval", type=float, default=1)
        if action == "download":
            item.add_argument("--output", type=Path, required=True)
            item.add_argument("--force", action="store_true")
    return p


def runtime_arguments(p):
    p.add_argument("--host", default="127.0.0.1", help="监听地址；远程部署用0.0.0.0并设置ITT_API_KEY")
    p.add_argument("--port", type=int, default=8095)
    p.add_argument("--upstream", type=Path, default=None)
    p.add_argument("--model-dir", type=Path, default=None, help="默认 upstream/checkpoints")
    p.add_argument("--device", default=None)
    p.add_argument("--no-bf16", action="store_true", help="关闭BF16；默认开启以降低显存")
    p.add_argument("--qwen-emotion", action="store_true", help="加载QwenEmotion开启文字/自动情绪，需要额外内存")
    p.add_argument("--cuda-kernel", action="store_true", help="启用BigVGAN自定义CUDA核，需要CUDA编译环境")
    p.add_argument("--compile", action="store_true", help="启用torch.compile；需要兼容Triton且首次编译较慢")
    p.add_argument("--engine", choices=["indextts", "test"], default="indextts", help="test仅产生测试音，不能作为模型验收")


def key_for(home, url, env_name):
    key = os.getenv(env_name, "")
    if key:
        return key
    local = read_json(home / "local-service.json", {})
    local_url = local.get("url", "http://127.0.0.1:8095")
    if url.rstrip("/") == local_url and (home / "local-key").exists():
        return (home / "local-key").read_text().strip()
    raise Failure("API_KEY_MISSING", f"请设置环境变量 {env_name}", 2)


def connection(args):
    home = Path(args.home).resolve()
    config = read_json(home / "client.json", {})
    local = read_json(home / "local-service.json", {})
    url = args.url or config.get("url") or local.get("url", "http://127.0.0.1:8095")
    env_name = args.api_key_env or config.get("api_key_env", "ITT_API_KEY")
    return Client(url, key_for(home, url, env_name), args.timeout)


def runtime_paths(args, home):
    """Resolve explicit CLI values, environment, persisted init profile, then legacy defaults."""
    home = Path(home).resolve()
    profile = read_json(home / "runtime.json", {})
    if not isinstance(profile, dict):
        profile = {}
    upstream = getattr(args, "upstream", None) or os.getenv("ITT_UPSTREAM") or profile.get("upstream") or str(ROOT / "upstream")
    upstream = Path(upstream).expanduser().resolve()
    model_dir = getattr(args, "model_dir", None) or os.getenv("ITT_MODEL_DIR") or profile.get("model_dir") or str(upstream / "checkpoints")
    model_dir = Path(model_dir).expanduser().resolve()
    device = getattr(args, "device", None) or os.getenv("ITT_DEVICE") or profile.get("device") or "cuda:0"
    runtime_python = getattr(args, "runtime_python", None) or os.getenv("ITT_RUNTIME_PYTHON") or profile.get("runtime_python")
    if runtime_python:
        runtime_python = Path(runtime_python).expanduser().resolve()
    else:
        runtime_python = upstream / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return upstream, model_dir, device, runtime_python


def runtime_config(args):
    home = Path(args.home).resolve()
    home.mkdir(parents=True, exist_ok=True)
    upstream, model_dir, device, runtime_python = runtime_paths(args, home)
    key = os.getenv(args.api_key_env or "ITT_API_KEY", "")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not key:
        raise Failure("API_KEY_MISSING", "监听外部网卡必须通过 ITT_API_KEY 或 --api-key-env 提供密钥", 2)
    if not key:
        key_path = home / "local-key"
        try:
            with key_path.open("x") as f:
                f.write(secrets.token_urlsafe(32))
            if os.name != "nt":
                key_path.chmod(0o600)
        except FileExistsError:
            pass
        key = key_path.read_text().strip()
    return {"data_dir": str(home), "upstream": str(upstream),
            "model_dir": str(model_dir), "runtime_python": str(runtime_python),
            "device": device, "bf16": not args.no_bf16, "qwen_emotion": args.qwen_emotion,
            "cuda_kernel": args.cuda_kernel, "compile": args.compile, "engine": args.engine,
            "host": args.host, "port": args.port, "api_key": key}


def background_start(args):
    home = Path(args.home).resolve()
    with InstanceLock(home / "start.lock"):
        cfg = runtime_config(args)
        old = read_json(home / "local-service.json", {})
        address = "127.0.0.1" if args.host in {"0.0.0.0", "localhost"} else "::1" if args.host == "::" else args.host
        authority = f"[{address}]" if ":" in address else address
        url = f"http://{authority}:{args.port}"
        client = Client(url, cfg["api_key"], 2)
        try:
            status = client.call("/v1/status")
        except Failure as e:
            if e.code != "CONNECTION_FAILED":
                raise
        else:
            if status.get("service") != "indextts-agent":
                raise Failure("PORT_IN_USE", "目标端口已被其他服务使用")
            if old.get("url") != url:
                raise Failure("SERVICE_NOT_OWNED", "该端口服务不属于当前 CLI home")
            return {"already_running": True, "url": url, **status}
        # Kernel lock detects another instance even while it is loading or draining.
        with InstanceLock(home / "service.lock"):
            pass
        forwarded = ["--home", str(home), "serve", "--host", args.host, "--port", str(args.port),
                     "--upstream", cfg["upstream"], "--model-dir", cfg["model_dir"], "--device", cfg["device"], "--engine", args.engine]
        for flag in ("no_bf16", "qwen_emotion", "cuda_kernel", "compile"):
            if getattr(args, flag):
                forwarded.append("--" + flag.replace("_", "-"))
        log_path = home / "service.log"
        if args.backend == "local":
            python = Path(cfg["runtime_python"])
            if not python.is_file():
                raise Failure("RUNTIME_MISSING", "缺少模型 Python，请运行 scripts/bootstrap.py 或指定 --runtime-python", 2)
            env = {**os.environ, "ITT_API_KEY": cfg["api_key"], "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
            creation = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
            with log_path.open("ab") as log:
                process = subprocess.Popen([str(python.resolve()), "-m", "indextts_agent.cli", *forwarded],
                                           cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env, **creation)
            info = {"backend": "local", "pid": process.pid, "url": url, "log": str(log_path)}
        else:
            name = "itt-" + __import__("hashlib").sha256(str(home).encode()).hexdigest()[:12]
            model_dir = Path(cfg["model_dir"])
            if not (model_dir / "config.yaml").is_file():
                raise Failure("MODEL_MISSING", "Docker启动前请下载模型", 2)
            docker_args = ["docker", "run", "--detach", "--rm", "--name", name, "--gpus", "all", "--init",
                           "--publish", f"{args.host}:{args.port}:8095", "--env", "ITT_API_KEY",
                           "--mount", f"type=bind,source={home},target=/data",
                           "--mount", f"type=bind,source={model_dir},target=/opt/indextts/checkpoints",
                           args.image, "--home", "/data", "serve", "--host", "0.0.0.0", "--port", "8095",
                           "--upstream", "/opt/indextts", "--model-dir", "/opt/indextts/checkpoints", "--device", args.device, "--engine", args.engine]
            for flag in ("no_bf16", "qwen_emotion", "cuda_kernel", "compile"):
                if getattr(args, flag):
                    docker_args.append("--" + flag.replace("_", "-"))
            result = subprocess.run(docker_args, env={**os.environ, "ITT_API_KEY": cfg["api_key"]}, capture_output=True, text=True, timeout=120, check=False)
            if result.returncode:
                raise Failure("DOCKER_START_FAILED", result.stderr.strip())
            info = {"backend": "docker", "container": name, "container_id": result.stdout.strip(), "url": url}
        atomic_json(home / "local-service.json", info)
    if args.wait:
        deadline = time.monotonic() + args.wait
        while time.monotonic() < deadline:
            try:
                status = client.call("/v1/status")
                if status["state"] == "ready":
                    return {**info, **status}
                if status["state"] == "failed":
                    raise Failure("MODEL_LOAD_FAILED", status["error"]["message"], details=info)
            except Failure as e:
                if e.code != "CONNECTION_FAILED":
                    raise
            if args.backend == "local" and process.poll() is not None:
                raise Failure("SERVICE_EXITED", "后台进程已退出；使用 service logs 查看原因", details=info)
            time.sleep(1)
        raise Failure("WAIT_TIMEOUT", "服务启动等待超时，后台进程仍保留；使用 service status/logs 检查", 3, info)
    return {**info, "state": "starting"}


def watch(client, jid, seconds, interval=1):
    deadline, previous = time.monotonic() + seconds, None
    while True:
        job = client.call("/v1/jobs/" + quote(jid, safe=""))
        fingerprint = (job["state"], job["progress"], job["phase"], job["cancel_requested"])
        if fingerprint != previous:
            emit({"ok": True, "event": "job", "job_id": jid,
                  **{k: job[k] for k in ("state", "progress", "phase", "cancel_requested", "result", "error")}})
            previous = fingerprint
        if job["state"] in TERMINAL:
            if job["state"] != "succeeded":
                raise Failure("JOB_" + job["state"].upper(), f"任务结束: {job['state']}", 4, {"job_id": jid, "error": job["error"]})
            return job
        if time.monotonic() >= deadline:
            raise Failure("WAIT_TIMEOUT", "等待超时，生成任务继续；用 jobs watch 恢复监控", 3, {"job_id": jid})
        time.sleep(max(0.1, min(interval, deadline - time.monotonic())))


def submit(args, client):
    if args.output and not args.wait:
        raise Failure("INVALID_ARGUMENT", "--output 需要 --wait 秒数", 2)
    body = read_json(args.request) if args.request else {}
    if not isinstance(body, dict):
        raise Failure("INVALID_REQUEST", "请求文件必须是 JSON 对象", 2)
    for key in ("text", "speaker", "lang", "emotion_text", "emotion_auto", "emotion_alpha", "emotion_random", "duration_factor",
                "seed", "num_beams", "max_text_tokens_per_segment", "interval_silence", "temperature", "top_p", "top_k", "max_mel_tokens"):
        value = getattr(args, key)
        if value is not None:
            body[key] = value
    if args.text_file:
        body["text"] = args.text_file.read_text(encoding="utf-8-sig")
    if args.speaker_audio:
        body["speaker"] = client.upload(args.speaker_audio)["id"]
    if args.voice:
        voices = client.call("/v1/voices")["voices"]
        voice = next((v for v in voices if v["name"] == args.voice), None)
        if voice is None:
            raise Failure("VOICE_NOT_FOUND", "找不到音色名；用 voices add 注册", 2)
        body["speaker"] = voice["asset"]
    if args.emotion_audio:
        body["emotion_audio"] = client.upload(args.emotion_audio)["id"]
    if args.emotion_vector:
        body["emotion_vector"] = [float(x) for x in args.emotion_vector.split(",")]
    if args.emotion:
        vector = [0.0] * 8
        seen = set()
        for part in args.emotion.split(","):
            name, value = part.split("=")
            if name not in EMOTIONS or name in seen:
                raise Failure("INVALID_EMOTION", "情绪名未知或重复；可用: " + ",".join(EMOTIONS), 2)
            seen.add(name)
            vector[EMOTIONS.index(name)] = float(value)
        body["emotion_vector"] = vector
    if args.speed is not None:
        if not 0.5 <= args.speed <= 2:
            raise Failure("INVALID_SPEED", "speed 必须在0.5..2", 2)
        body["duration_factor"] = 1 / args.speed
    if args.no_text_normalization:
        body["text_normalization"] = False
    idem = args.idempotency_key or uuid.uuid4().hex
    try:
        result = client.call("/v1/jobs", "POST", body, headers={"Idempotency-Key": idem})
    except Failure as e:
        if e.code == "CONNECTION_FAILED":
            raise Failure("SUBMISSION_UNKNOWN", "提交响应未确认；用同一请求和幂等键重试，勿创建新键", 1, {"idempotency_key": idem}) from None
        raise
    result["idempotency_key"] = idem
    if args.wait:
        emit({"ok": True, **result})
        job = watch(client, result["job"]["id"], args.wait)
        return client.download(job["id"], args.output) if args.output else {"job": job}
    return result


def run(args):
    home = Path(args.home).resolve()
    if args.command == "uninstall":
        from .uninstall import uninstall
        return uninstall(args)
    if args.timeout <= 0:
        raise Failure("INVALID_ARGUMENT", "HTTP timeout 必须大于0", 2)
    if args.command == "init":
        return init_command.run(args, emit=emit)
    if args.command == "config":
        if args.action == "set":
            Client(args.config_url, "")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.config_key_env):
                raise Failure("INVALID_ARGUMENT", "API Key 环境变量名不合法", 2)
            atomic_json(home / "client.json", {"url": args.config_url.rstrip("/"), "api_key_env": args.config_key_env})
        return read_json(home / "client.json", {})
    if args.command == "doctor":
        upstream, model_dir, device, runtime_python = runtime_paths(args, home)
        result = {"cli_version": __version__, "python": sys.executable, "upstream_exists": upstream.is_dir(),
                  "model_config_exists": (model_dir / "config.yaml").is_file(),
                  "upstream": str(upstream), "model_dir": str(model_dir),
                  "runtime_python": str(runtime_python), "device": device,
                  "executables": {n: shutil.which(n) for n in ("uv", "hf", "docker", "nvidia-smi")}}
        if shutil.which("nvidia-smi"):
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"], capture_output=True, text=True, timeout=15, check=False)
            result["gpu"] = gpu.stdout.strip()
        if runtime_python:
            check = subprocess.run([str(runtime_python), "-c", "import json,torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'bf16':torch.cuda.is_bf16_supported()}))"], capture_output=True, text=True, timeout=60, check=False)
            result["runtime"] = {"exit_code": check.returncode, "output": check.stdout.strip(), "error": check.stderr.strip()}
        return result
    if args.command == "serve":
        has_server = importlib.util.find_spec("fastapi") is not None
        has_model = args.engine == "test" or importlib.util.find_spec("torch") is not None
        if not (has_server and has_model):
            _upstream, _model_dir, _device, python = runtime_paths(args, home)
            if not python.is_file() or python.resolve() == Path(sys.executable).resolve():
                raise Failure("RUNTIME_MISSING", "当前Python缺少服务/模型依赖；请运行bootstrap或指定 --runtime-python", 2)
            result = subprocess.run([str(python.resolve()), "-m", "indextts_agent.cli", *sys.argv[1:]], check=False)
            if result.returncode:
                raise Failure("SERVICE_EXITED", "前台模型进程已退出", result.returncode)
            return None
        cfg = runtime_config(args)
        from .server import serve
        # Model logs are stderr; CLI stdout is reserved for JSON.
        sys.stdout = sys.stderr
        serve(cfg)
        return None
    if args.command == "service" and args.action == "start":
        return background_start(args)
    if args.command == "service" and args.action == "logs":
        info = read_json(home / "local-service.json", {})
        if info.get("backend") == "docker":
            out = subprocess.run(["docker", "logs", "--tail", str(max(1, args.tail)), info["container"]], capture_output=True, text=True, timeout=20, check=False)
            return {"logs": (out.stdout + out.stderr).splitlines()}
        path = home / "service.log"
        return {"logs": path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, args.tail):] if path.exists() else []}
    client = connection(args)
    if args.command == "service":
        if args.action == "status":
            return client.call("/v1/status")
        result = client.call("/v1/service/stop", "POST")
        if args.wait:
            deadline = time.monotonic() + args.wait
            while time.monotonic() < deadline:
                try:
                    client.call("/v1/status")
                except Failure as e:
                    if e.code == "CONNECTION_FAILED":
                        return {"state": "stopped"}
                    raise
                time.sleep(0.5)
            raise Failure("WAIT_TIMEOUT", "服务正在完成当前生成；稍后查询状态", 3)
        return result
    if args.command == "capabilities":
        return client.call("/v1/capabilities")
    if args.command == "assets":
        return client.upload(args.audio)
    if args.command == "voices":
        if args.action == "list":
            return client.call("/v1/voices")
        aid = client.upload(args.audio)["id"]
        return client.call("/v1/voices", "POST", {"name": args.name, "asset": aid})
    if args.command == "jobs":
        if args.action == "submit":
            return submit(args, client)
        if args.action == "list":
            return client.call("/v1/jobs?limit=" + str(args.limit))
        if args.action == "download":
            return client.download(args.job_id, args.output, args.force)
        if args.action == "watch":
            return {"job": watch(client, args.job_id, args.wait, args.interval)}
        path = "/v1/jobs/" + quote(args.job_id, safe="")
        return client.call(path + "/cancel", "POST") if args.action == "cancel" else client.call(path)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        result = run(parser().parse_args())
        if result is not None:
            emit({"ok": True, **result})
    except Failure as e:
        emit({"ok": False, "error": {"code": e.code, "message": str(e), "details": e.details}})
        return e.exit_code
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        emit({"ok": False, "error": {"code": "LOCAL_ERROR", "message": str(e)}})
        return 2
    except KeyboardInterrupt:
        emit({"ok": False, "error": {"code": "MONITOR_INTERRUPTED", "message": "停止等待；已提交任务继续运行"}})
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
