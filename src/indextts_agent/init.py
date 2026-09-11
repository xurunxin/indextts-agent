"""Create and inspect an isolated IndexTTS runtime.

The module deliberately keeps provisioning policy here and process/file helpers in
``provision``.  It can be run by the base CLI or by the Python interpreter inside
the newly-created runtime for model downloads::

    python -m indextts_agent.init --download-models --model-dir ...

The model downloader prints one JSON result to stdout and sends diagnostics to
stderr.  The parent initializer captures that output in ``init.log``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .common import (
    ROOT,
    Failure,
    InstanceLock,
    atomic_json,
    read_json,
)
from .provision import Installer, download, safe_claim, safe_extract_zip

PACKAGE_DIR = Path(__file__).resolve().parent
LOCK_DIR = PACKAGE_DIR / "locks"


def _lock_file(name: str) -> Path:
    """Use packaged locks first, with a source checkout fallback."""

    packaged = LOCK_DIR / name
    if packaged.is_file():
        return packaged
    return ROOT / name


def _read_lock(name: str) -> dict[str, Any]:
    path = _lock_file(name)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Failure("LOCK_MISSING", f"无法读取固定锁文件: {path}", 2) from exc
    if not isinstance(value, dict):
        raise Failure("LOCK_INVALID", f"锁文件必须是 JSON 对象: {path}", 2)
    return value


UPSTREAM_LOCK = _read_lock("upstream.lock.json")
MODELS_LOCK = _read_lock("models.lock.json")

UPSTREAM_REQUIRED = (
    "pyproject.toml",
    "uv.lock",
    "indextts/infer_v2_5.py",
)

# These are the files used by the pinned 2.5 inference path.  Checking their
# size catches a partially resumed snapshot while allowing harmless metadata
# files to vary between Hub clients.
MAIN_MODEL_REQUIRED = (
    "config.yaml",
    "codec.pth",
    "feat1.pt",
    "feat2.pt",
    "gpt.pth",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
    "qwen0.6bemo4-merge/config.json",
    "qwen0.6bemo4-merge/model.safetensors",
    "qwen0.6bemo4-merge/tokenizer.json",
    "qwen0.6bemo4-merge/tokenizer_config.json",
    "qwen0.6bemo4-merge/vocab.json",
    "qwen0.6bemo4-merge/merges.txt",
)

AUXILIARY_MODEL_REQUIRED = {
    "facebook/w2v-bert-2.0": (
        "hf_cache/w2v-bert-2.0/config.json",
        "hf_cache/w2v-bert-2.0/model.safetensors",
        "hf_cache/w2v-bert-2.0/preprocessor_config.json",
    ),
    "nvidia/bigvgan_v2_22khz_80band_256x": (
        "hf_cache/bigvgan/config.json",
        "hf_cache/bigvgan/bigvgan_generator.pt",
    ),
    "funasr/campplus": ("hf_cache/campplus_cn_common.bin",),
    "amphion/MaskGCT": ("hf_cache/semantic_codec_model.safetensors",),
}

AUXILIARY_REMOTE_FILES = {
    "facebook/w2v-bert-2.0": {
        "hf_cache/w2v-bert-2.0/config.json": "config.json",
        "hf_cache/w2v-bert-2.0/model.safetensors": "model.safetensors",
        "hf_cache/w2v-bert-2.0/preprocessor_config.json": "preprocessor_config.json",
    },
    "nvidia/bigvgan_v2_22khz_80band_256x": {
        "hf_cache/bigvgan/config.json": "config.json",
        "hf_cache/bigvgan/bigvgan_generator.pt": "bigvgan_generator.pt",
    },
    "funasr/campplus": {"hf_cache/campplus_cn_common.bin": "campplus_cn_common.bin"},
    "amphion/MaskGCT": {
        "hf_cache/semantic_codec_model.safetensors": "semantic_codec/model.safetensors",
    },
}
MODEL_MANIFEST = ".itt-model-manifest.json"


def _emit_default(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    except OSError:
        return False


def _file_check(root: Path, relative_paths: tuple[str, ...]) -> tuple[bool, list[str]]:
    missing = [name for name in relative_paths if not _nonempty(root / Path(name))]
    return not missing, missing


def _model_file_map() -> list[tuple[str, str, str]]:
    main_repo = UPSTREAM_LOCK["model"]
    files = [(main_repo, name, name) for name in MAIN_MODEL_REQUIRED]
    files.extend(
        (repo, remote, local)
        for repo, mapping in AUXILIARY_REMOTE_FILES.items()
        for local, remote in mapping.items()
    )
    return files


def _manifest_check(model_dir: Path) -> tuple[bool, dict[str, Any]]:
    path = model_dir / MODEL_MANIFEST
    if not _nonempty(path):
        return False, {"manifest": "missing"}
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False, {"manifest": "invalid JSON"}
    if not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("revisions") != MODELS_LOCK:
        return False, {"manifest": "revision or schema mismatch"}
    entries = value.get("files")
    if not isinstance(entries, dict):
        return False, {"manifest": "file entries missing"}
    mismatches: list[str] = []
    for repo, remote, local in _model_file_map():
        entry = entries.get(f"{repo}/{remote}")
        target = model_dir / Path(local)
        if not isinstance(entry, dict) or not isinstance(entry.get("size"), int):
            mismatches.append(f"{repo}/{remote}: manifest entry missing")
        elif not _nonempty(target) or target.stat().st_size != entry["size"]:
            mismatches.append(f"{local}: expected {entry['size']} bytes")
    if mismatches:
        return False, {"manifest": mismatches}
    return True, {"manifest": str(path)}


def _runtime_python(install_dir: Path) -> Path:
    return install_dir / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _resolve_plan(home: Path, args: Any) -> dict[str, Any]:
    existing = read_json(home / "runtime.json", {})
    if not isinstance(existing, dict):
        existing = {}
    requested_install = getattr(args, "install_dir", None)
    install_text = requested_install or os.getenv("ITT_INSTALL_DIR") or existing.get("install_dir")
    install_dir = Path(install_text).expanduser() if install_text else home / "runtime"
    install_dir = install_dir.resolve()
    upstream = install_dir / "upstream"
    model_dir = install_dir / "models"
    runtime_python = _runtime_python(install_dir)
    requested_device = getattr(args, "device", None)
    device = requested_device or os.getenv("ITT_DEVICE") or existing.get("device") or "cuda"
    if device not in {"cuda", "cpu"}:
        raise Failure("INVALID_ARGUMENT", "device 必须是 cuda 或 cpu", 2)
    return {
        "home": home,
        "install_dir": install_dir,
        "upstream": upstream,
        "model_dir": model_dir,
        "runtime_python": runtime_python,
        "device": device,
        "existing": existing,
    }


def _config_mismatches(plan: dict[str, Any]) -> dict[str, dict[str, str]]:
    existing = plan["existing"]
    if not existing:
        return {}
    mismatches: dict[str, dict[str, str]] = {}
    expected = {
        "install_dir": str(plan["install_dir"]),
        "upstream": str(plan["upstream"]),
        "model_dir": str(plan["model_dir"]),
        "runtime_python": str(plan["runtime_python"]),
        "device": plan["device"],
    }
    for key, value in expected.items():
        actual = existing.get(key)
        if actual is not None and str(actual) != value:
            mismatches[key] = {"configured": str(actual), "requested": value}
    return mismatches


def _ownership(plan: dict[str, Any]) -> tuple[bool, dict[str, str]]:
    """Read-only ownership check matching the values used by safe_claim."""

    marker_value = {"schema": 1, "revision": UPSTREAM_LOCK["revision"], "repository": UPSTREAM_LOCK["repository"]}
    details: dict[str, str] = {}
    install_dir = plan["install_dir"]
    marker = install_dir / ".itt-runtime.json"
    if marker.exists():
        if read_json(marker) != marker_value:
            details["install_dir"] = "marker conflict"
            return False, details
    elif install_dir.exists():
        try:
            if any(install_dir.iterdir()):
                details["install_dir"] = "non-empty directory is not owned by itt"
                return False, details
        except OSError:
            details["install_dir"] = "directory cannot be inspected"
            return False, details
    details["install_dir"] = "owned" if marker.exists() else "available"

    upstream = plan["upstream"]
    upstream_marker = upstream / ".itt-upstream.json"
    if upstream_marker.exists():
        if read_json(upstream_marker) != marker_value:
            details["upstream"] = "marker conflict"
            return False, details
    elif upstream.exists():
        try:
            if any(upstream.iterdir()):
                details["upstream"] = "non-empty directory is not owned by itt"
                return False, details
        except OSError:
            details["upstream"] = "directory cannot be inspected"
            return False, details
    details["upstream"] = "owned" if upstream_marker.exists() else "available"
    return True, details


def _runtime_probe(plan: dict[str, Any]) -> tuple[bool, bool, str]:
    python = plan["runtime_python"]
    if not _nonempty(python):
        return False, False, "runtime Python 不存在"
    code = (
        "import json,torch,fastapi,indextts; "
        "print(json.dumps({'torch': getattr(torch, '__version__', None), "
        "'cuda': bool(torch.cuda.is_available())}))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            cwd=plan["upstream"],
            env={**os.environ, "PYTHONPATH": str(plan["upstream"]), "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, False, f"runtime probe failed: {exc}"
    if result.returncode:
        message = (result.stderr or result.stdout).strip().splitlines()
        return False, False, message[-1] if message else f"exit code {result.returncode}"
    try:
        value = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return False, False, "runtime probe returned invalid JSON"
    return True, bool(value.get("cuda")), "ok"


def check_installation(plan: dict[str, Any]) -> dict[str, Any]:
    """Return an entirely read-only readiness report for a planned runtime."""

    owned, ownership_details = _ownership(plan)
    source_ok, source_missing = _file_check(plan["upstream"], UPSTREAM_REQUIRED)
    main_ok, main_missing = _file_check(plan["model_dir"], MAIN_MODEL_REQUIRED)
    aux_missing: dict[str, list[str]] = {}
    for repo, paths in AUXILIARY_MODEL_REQUIRED.items():
        ok, missing = _file_check(plan["model_dir"], paths)
        if not ok:
            aux_missing[repo] = missing
    manifest_ok, manifest_details = _manifest_check(plan["model_dir"])
    models_ok = main_ok and not aux_missing and manifest_ok
    runtime_ok, cuda_ok, runtime_message = _runtime_probe(plan)
    config_mismatches = _config_mismatches(plan)
    device_ok = plan["device"] == "cuda" and cuda_ok
    ready = owned and not config_mismatches and source_ok and models_ok and runtime_ok and device_ok
    return {
        "ready": ready,
        "checks": {
            "ownership": owned,
            "config_compatible": not config_mismatches,
            "upstream": source_ok,
            "runtime": runtime_ok,
            "models": models_ok,
            "cuda": cuda_ok,
            "device": device_ok,
        },
        "check_details": {
            "ownership": ownership_details,
            "config_conflicts": config_mismatches,
            "upstream_missing": source_missing,
            "main_model_missing": main_missing,
            "auxiliary_model_missing": aux_missing,
            "model_manifest": manifest_details,
            "runtime": runtime_message,
        },
    }


def planned_paths(plan: dict[str, Any]) -> dict[str, str]:
    return {
        "home": str(plan["home"]),
        "install_dir": str(plan["install_dir"]),
        "upstream": str(plan["upstream"]),
        "model_dir": str(plan["model_dir"]),
        "runtime_python": str(plan["runtime_python"]),
        "runtime_config": str(plan["home"] / "runtime.json"),
        "init_log": str(plan["home"] / "init.log"),
    }


def _archive_url(repository: str, revision: str) -> str:
    parsed = urlparse(repository)
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise Failure("UPSTREAM_UNTRUSTED", "固定上游必须来自 https://github.com", 2)
    path = parsed.path.rstrip("/")
    path = path.removesuffix(".git")
    if not path.count("/") == 2:
        raise Failure("UPSTREAM_UNTRUSTED", "固定上游 GitHub 仓库地址无效", 2)
    return f"https://github.com{path}/archive/{revision}.zip"


def _claim_values() -> dict[str, Any]:
    return {"schema": 1, "revision": UPSTREAM_LOCK["revision"], "repository": UPSTREAM_LOCK["repository"]}


def _prepare_upstream(plan: dict[str, Any], installer: Installer) -> None:
    upstream = plan["upstream"]
    value = _claim_values()
    safe_claim(plan["install_dir"], ".itt-runtime.json", value)
    safe_claim(upstream, ".itt-upstream.json", value)
    source_ok, _ = _file_check(upstream, UPSTREAM_REQUIRED)
    if source_ok:
        return
    cache_dir = plan["home"] / "init-cache"
    archive = cache_dir / f"upstream-{UPSTREAM_LOCK['revision']}.zip"
    if not _nonempty(archive):
        download(_archive_url(UPSTREAM_LOCK["repository"], UPSTREAM_LOCK["revision"]), archive)
    safe_extract_zip(archive, upstream, strip_root=True)
    source_ok, missing = _file_check(upstream, UPSTREAM_REQUIRED)
    if not source_ok:
        raise Failure("UPSTREAM_INCOMPLETE", f"源码解包不完整，缺少: {', '.join(missing)}", 1)


def _run_runtime_download(plan: dict[str, Any], installer: Installer) -> None:
    if _manifest_check(plan["model_dir"])[0]:
        return
    python = plan["runtime_python"]
    if not _nonempty(python):
        raise Failure("RUNTIME_MISSING", "模型环境 Python 不存在，无法下载模型", 2)
    main_ok, _ = _file_check(plan["model_dir"], MAIN_MODEL_REQUIRED)
    aux_ok = all(_file_check(plan["model_dir"], files)[0] for files in AUXILIARY_MODEL_REQUIRED.values())
    manifest_ok, _ = _manifest_check(plan["model_dir"])
    if main_ok and aux_ok and manifest_ok:
        installer.emit({"ok": True, "event": "init_stage", "stage": "models", "state": "succeeded", "reused": True})
        return
    env = {
        "HF_HOME": str(plan["model_dir"] / ".hf-cache"),
        "HF_HUB_CACHE": str(plan["model_dir"] / ".hf-cache" / "hub"),
    }
    # Keep modern Hub transport independent of the model's frozen old SDK.
    uv = installer.ensure_uv()
    wheel = installer.client_wheel("indextts_agent", "indextts-agent")
    installer.run(
        "models",
        [uv, "run", "--no-project", "--python", "3.11", "--with", "huggingface-hub==1.10.1", "--with", wheel,
         "python", "-m", "indextts_agent.init", "--download-models", "--model-dir", plan["model_dir"]],
        cwd=plan["upstream"],
        env=env,
    )


def _cuda_preflight() -> None:
    """Reject a missing NVIDIA driver before creating or downloading anything."""

    executable = shutil.which("nvidia-smi")
    if not executable:
        raise Failure(
            "CUDA_DRIVER_MISSING",
            "找不到 nvidia-smi；请先安装匹配的 NVIDIA 驱动后重试，init 不会自动安装驱动",
            2,
        )
    try:
        result = subprocess.run(
            [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Failure("CUDA_DRIVER_MISSING", f"无法检查 NVIDIA 驱动: {exc}", 2) from None
    if result.returncode or not result.stdout.strip():
        message = (result.stderr or result.stdout).strip().splitlines()
        detail = message[-1] if message else "nvidia-smi 无可用 GPU"
        raise Failure(
            "CUDA_DRIVER_MISSING",
            f"NVIDIA 驱动或 GPU 不可用: {detail}；init 不会自动安装驱动",
            2,
        )


def _install_runtime(plan: dict[str, Any], installer: Installer) -> None:
    uv = installer.ensure_uv()
    installer.run(
        "runtime",
        [uv, "sync", "--frozen", "--no-dev", "--python", "3.11"],
        cwd=plan["upstream"],
        env={"UV_PROJECT_ENVIRONMENT": str(plan["install_dir"] / ".venv")},
    )
    python = plan["runtime_python"]
    wheel = installer.client_wheel("indextts_agent", "indextts-agent")
    installer.run("client", [uv, "pip", "install", "--python", python, "--reinstall-package", "indextts-agent", f"{wheel}[server]"])


def initialize(plan: dict[str, Any], *, skip_models: bool, emit: Callable[[dict[str, Any]], None], wait: float = 600, start: bool = False) -> dict[str, Any]:
    """Provision one isolated runtime and persist its profile."""

    if plan["device"] == "cpu":
        # The pinned upstream lock selects CUDA wheels on Windows/Linux.  Do
        # not create a marker or a partial environment while pretending CPU is
        # supported; a future CPU lock can make this branch explicit.
        raise Failure("CPU_UNSUPPORTED", "当前固定 IndexTTS 运行锁使用 CUDA PyTorch；CPU 部署未实现，请使用 --device cuda", 2)
    conflicts = _config_mismatches(plan)
    if conflicts:
        raise Failure("INSTALL_CONFLICT", "runtime.json 与当前安装参数冲突，请沿用原路径或指定新的 home", 2, conflicts)
    owned, ownership_details = _ownership(plan)
    if not owned:
        conflict = any("conflict" in value for value in ownership_details.values())
        raise Failure(
            "INSTALL_CONFLICT" if conflict else "DIRECTORY_NOT_OWNED",
            "安装目录已有不匹配的标记或未被 itt 管理，不会接管",
            2,
            ownership_details,
        )
    _cuda_preflight()
    home = plan["home"]
    home.mkdir(parents=True, exist_ok=True)
    installer = Installer(home, "itt", emit)
    with InstanceLock(home / "init.lock"):
        _prepare_upstream(plan, installer)
        _install_runtime(plan, installer)
        if not skip_models:
            _run_runtime_download(plan, installer)
        report = check_installation(plan)
        if not report["checks"]["runtime"] or not report["checks"]["cuda"]:
            raise Failure(
                "RUNTIME_INVALID",
                "运行时导入或 CUDA 检查失败，请查看 init.log 并修复后重试",
                1,
                report["check_details"],
            )
        if not skip_models and not report["checks"]["models"]:
            raise Failure("MODEL_INCOMPLETE", "模型下载未完成，请检查 init.log 后重试", 1, report["check_details"])
        model_ready = bool(report["checks"]["models"] and not skip_models)
        runtime_config = {
            "schema_version": 1,
            "install_dir": str(plan["install_dir"]),
            "runtime_python": str(plan["runtime_python"]),
            "upstream": str(plan["upstream"]),
            "model_dir": str(plan["model_dir"]),
            "device": plan["device"],
            "upstream_revision": UPSTREAM_LOCK["revision"],
            "model_revisions": MODELS_LOCK,
            "models_ready": model_ready,
            "ready": bool(report["ready"] and model_ready),
            "updated_at": time.time(),
        }
        atomic_json(home / "runtime.json", runtime_config)
    result = {
        "ready": runtime_config["ready"],
        "runtime": runtime_config,
        "checks": report["checks"],
        "check_details": report["check_details"],
        "planned": planned_paths(plan),
    }
    if start:
        if skip_models or not runtime_config["ready"]:
            raise Failure("MODEL_NOT_READY", "--start 需要完整模型和可用 CUDA；请先完成 init", 2, result)
        from .cli import background_start

        service_args = argparse.Namespace(
            home=str(home),
            backend="local",
            host="127.0.0.1",
            port=8095,
            runtime_python=plan["runtime_python"],
            upstream=plan["upstream"],
            model_dir=plan["model_dir"],
            device=plan["device"],
            no_bf16=False,
            qwen_emotion=False,
            cuda_kernel=False,
            compile=False,
            engine="indextts",
            wait=wait,
            api_key_env=None,
            timeout=30,
            url=None,
        )
        result["service"] = background_start(service_args)
    return result


def run(args: Any, emit: Callable[[dict[str, Any]], None] = _emit_default) -> dict[str, Any]:
    home = Path(args.home).expanduser().resolve()
    plan = _resolve_plan(home, args)
    if plan["device"] == "cpu":
        raise Failure("CPU_UNSUPPORTED", "当前固定 IndexTTS 运行锁使用 CUDA PyTorch；CPU 部署未实现，请使用 --device cuda", 2)
    report = check_installation(plan)
    if args.check or args.dry_run:
        return {
            **report,
            "planned": planned_paths(plan),
            "dry_run": bool(args.dry_run),
        }
    return initialize(plan, skip_models=args.skip_models, emit=emit, wait=args.wait, start=args.start)


def _download_snapshot(repo: str, revision: str, local_dir: Path, *, cache_dir: Path, allow_patterns: list[str] | None = None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise Failure("HF_MISSING", "运行环境缺少 huggingface_hub，请重试 init", 1) from exc
    local_dir.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(1, 4):
        try:
            snapshot_download(
                repo_id=repo,
                revision=revision,
                local_dir=str(local_dir),
                cache_dir=str(cache_dir),
                allow_patterns=allow_patterns,
                max_workers=4,
            )
            return
        except Exception as exc:  # noqa: BLE001 - Hub transport errors vary.
            last = exc
            print(f"model download attempt {attempt}/3 failed for {repo}: {exc}", file=sys.stderr, flush=True)
            if attempt < 3:
                time.sleep(min(2**attempt, 8))
    raise Failure("MODEL_DOWNLOAD_FAILED", f"模型 {repo} 下载失败，已断点重试3次", 1, {"repo": repo, "revision": revision}) from last


def _model_manifest(model_dir: Path) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise Failure("HF_MISSING", "运行环境缺少 huggingface_hub，请重试 init", 1) from exc
    files: dict[str, dict[str, Any]] = {}
    api = HfApi()
    for repo, revision in MODELS_LOCK.items():
        required = [(remote, local) for item_repo, remote, local in _model_file_map() if item_repo == repo]
        for attempt in range(3):
            try:
                info = api.model_info(repo, revision=revision, files_metadata=True, timeout=30)
                break
            except Exception as exc:
                if attempt == 2:
                    raise Failure("MODEL_MANIFEST_FAILED", f"无法读取模型 {repo} 的固定文件清单，已重试3次", 1,
                                  {"repo": repo, "revision": revision, "error_type": type(exc).__name__}) from exc
                time.sleep(2 ** attempt)
        siblings = {getattr(item, "rfilename", None): getattr(item, "size", None) for item in (info.siblings or [])}
        for remote, local in required:
            size = siblings.get(remote)
            if not isinstance(size, int) or size <= 0:
                raise Failure("MODEL_MANIFEST_FAILED", f"模型 {repo} 缺少固定文件元数据: {remote}", 1)
            target = model_dir / Path(local)
            if not _nonempty(target) or target.stat().st_size != size:
                raise Failure("MODEL_SIZE_MISMATCH", f"模型文件大小不符: {local}", 1, {"expected": size, "actual": target.stat().st_size if target.exists() else 0})
            files[f"{repo}/{remote}"] = {"path": local, "size": size}
    return {"schema_version": 1, "revisions": MODELS_LOCK, "files": files, "updated_at": time.time()}


def download_models(model_dir: Path) -> dict[str, Any]:
    """Download the pinned main and auxiliary snapshots using this runtime."""

    model_dir = Path(model_dir).resolve()
    cache_dir = model_dir / ".hf-cache"
    manifest_ok, _ = _manifest_check(model_dir)
    if manifest_ok:
        return {"models_ready": True, "model_dir": str(model_dir), "revisions": MODELS_LOCK, "reused": True}
    # A local file may exist but have been truncated after the last download.
    # Repair only those artifacts, using the pinned remote source again.
    previous = read_json(model_dir / MODEL_MANIFEST, {})
    if isinstance(previous, dict) and previous.get("revisions") == MODELS_LOCK:
        from huggingface_hub import hf_hub_download

        for repo, remote, local in _model_file_map():
            entry = previous.get("files", {}).get(f"{repo}/{remote}", {})
            target = model_dir / local
            if isinstance(entry, dict) and isinstance(entry.get("size"), int) and (
                not _nonempty(target) or target.stat().st_size != entry["size"]
            ):
                source = hf_hub_download(repo, remote, revision=MODELS_LOCK[repo], cache_dir=cache_dir, force_download=True)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    main_repo = UPSTREAM_LOCK["model"]
    main_ok, _ = _file_check(model_dir, MAIN_MODEL_REQUIRED)
    if not main_ok:
        _download_snapshot(main_repo, MODELS_LOCK[main_repo], model_dir, cache_dir=cache_dir)
    for repo, paths in AUXILIARY_MODEL_REQUIRED.items():
        # Match the upstream loader's flat cache layout.  The source snapshot
        # for MaskGCT keeps its semantic codec in a nested directory; copy the
        # alias consumed by infer_v2_5 after the snapshot is complete.
        if repo == "facebook/w2v-bert-2.0":
            destination = model_dir / "hf_cache" / "w2v-bert-2.0"
            patterns = ["config.json", "model.safetensors", "preprocessor_config.json"]
        elif repo == "nvidia/bigvgan_v2_22khz_80band_256x":
            destination = model_dir / "hf_cache" / "bigvgan"
            patterns = ["config.json", "bigvgan_generator.pt"]
        elif repo == "funasr/campplus":
            destination = model_dir / "hf_cache" / "campplus"
            patterns = ["campplus_cn_common.bin"]
        else:
            destination = model_dir / "hf_cache" / "MaskGCT"
            patterns = ["semantic_codec/model.safetensors"]
        complete, _ = _file_check(model_dir, paths)
        if complete:
            continue
        _download_snapshot(repo, MODELS_LOCK[repo], destination, cache_dir=cache_dir, allow_patterns=patterns)
        if repo == "funasr/campplus":
            source = destination / "campplus_cn_common.bin"
            target = model_dir / "hf_cache" / "campplus_cn_common.bin"
            if _nonempty(source):
                shutil.copy2(source, target)
        elif repo == "amphion/MaskGCT":
            source = destination / "semantic_codec" / "model.safetensors"
            target = model_dir / "hf_cache" / "semantic_codec_model.safetensors"
            if _nonempty(source):
                shutil.copy2(source, target)
    atomic_json(model_dir / MODEL_MANIFEST, _model_manifest(model_dir))
    report = check_installation(
        {
            "home": model_dir.parent.parent,
            "install_dir": model_dir.parent,
            "upstream": model_dir.parent / "upstream",
            "model_dir": model_dir,
            "runtime_python": Path(sys.executable),
            "device": "cuda",
            "existing": {},
        }
    )
    if not report["checks"]["models"]:
        raise Failure("MODEL_INCOMPLETE", "模型快照下载后仍缺少必要文件", 1, report["check_details"])
    return {"models_ready": True, "model_dir": str(model_dir), "revisions": MODELS_LOCK}


def _module_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m indextts_agent.init")
    parser.add_argument("--download-models", "--download", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    return parser


def module_main(argv: list[str] | None = None) -> int:
    args = _module_parser().parse_args(argv)
    if not args.download_models:
        _emit_default({"ok": False, "error": {"code": "INVALID_ARGUMENT", "message": "需要 --download-models"}})
        return 2
    if not args.model_dir:
        _emit_default({"ok": False, "error": {"code": "INVALID_ARGUMENT", "message": "需要 --model-dir"}})
        return 2
    try:
        _emit_default({"ok": True, **download_models(args.model_dir)})
        return 0
    except Failure as exc:
        _emit_default({"ok": False, "error": {"code": exc.code, "message": str(exc), "details": exc.details}})
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(module_main())
