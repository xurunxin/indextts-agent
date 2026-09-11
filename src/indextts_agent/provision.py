"""Standard-library provisioning helpers. Also shipped with the standalone itt wheel."""
import base64
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .common import Failure, atomic_json, read_json

UV_VERSION = "0.12.9"


def safe_claim(directory, marker_name, value):
    directory = Path(directory).resolve()
    marker = directory / marker_name
    if marker.exists():
        if read_json(marker) != value:
            raise Failure("INSTALL_CONFLICT", f"安装配置不同，请选择新的 --install-dir: {directory}", 2)
    elif directory.exists() and any(directory.iterdir()):
        raise Failure("DIRECTORY_NOT_OWNED", f"非空目录没有安装标记，不会接管: {directory}", 2)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, value)


def safe_extract_zip(archive, destination, strip_root=True):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        entries = []
        for info in bundle.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename or ":" in info.filename:
                raise Failure("UNSAFE_ARCHIVE", "源码压缩包包含非法路径", 2)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise Failure("UNSAFE_ARCHIVE", "源码压缩包包含符号链接", 2)
            parts = path.parts[1:] if strip_root else path.parts
            if not parts or info.is_dir():
                continue
            target = destination.joinpath(*parts).resolve()
            if not target.is_relative_to(destination):
                raise Failure("UNSAFE_ARCHIVE", "源码压缩包路径越界", 2)
            entries.append((info, target))
        for info, target in entries:
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def download(url, target, expected_sha256=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, pending = tempfile.mkstemp(dir=target.parent, suffix=".part")
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(url, timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if expected_sha256 and digest.hexdigest() != expected_sha256:
            raise Failure("CHECKSUM_MISMATCH", "下载文件 SHA-256 不匹配")
        os.replace(pending, target)
    finally:
        Path(pending).unlink(missing_ok=True)


class Installer:
    def __init__(self, home, tool, emit, timeout=3600):
        self.home, self.tool, self.emit, self.timeout = Path(home).resolve(), tool, emit, timeout
        self.log = self.home / "init.log"

    def run(self, stage, command, cwd=None, env=None):
        self.home.mkdir(parents=True, exist_ok=True)
        self.emit({"ok": True, "event": "init_stage", "stage": stage, "state": "running", "log": str(self.log)})
        state = {"stage": stage, "state": "running", "updated": time.time()}
        atomic_json(self.home / "init-state.json", state)
        environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "UV_NO_PROGRESS": "1", **(env or {})}
        try:
            with self.log.open("ab") as output:
                output.write(f"\n[{stage}]\n".encode())
                output.flush()
                stage_start = output.tell()
                result = subprocess.run(list(map(str, command)), cwd=cwd, env=environment,
                                        stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                        timeout=self.timeout, check=False)
            with self.log.open("rb") as output:
                output.seek(max(stage_start, self.log.stat().st_size - 65536))
                text = output.read().decode("utf-8", errors="replace")
            if result.returncode:
                raise Failure("INIT_STAGE_FAILED", f"阶段 {stage} 失败（退出码 {result.returncode}）；查看 {self.log}，修复后重新执行 init", 1,
                              {"stage": stage, "returncode": result.returncode, "log": str(self.log)})
        except (OSError, subprocess.TimeoutExpired) as exc:
            atomic_json(self.home / "init-state.json", {**state, "state": "failed", "error_type": type(exc).__name__})
            raise Failure("INIT_STAGE_FAILED", f"阶段 {stage} 无法完成；检查网络/磁盘后重新执行 init", 1,
                          {"stage": stage, "log": str(self.log), "error_type": type(exc).__name__}) from None
        except Failure:
            atomic_json(self.home / "init-state.json", {**state, "state": "failed"})
            raise
        atomic_json(self.home / "init-state.json", {**state, "state": "succeeded"})
        self.emit({"ok": True, "event": "init_stage", "stage": stage, "state": "succeeded"})
        return text

    def ensure_uv(self):
        existing = shutil.which("uv")
        if existing:
            return existing
        private = self.home / "tools" / ("uv.exe" if os.name == "nt" else "uv")
        if private.is_file():
            return str(private)
        machine = platform.machine().lower()
        if sys.platform == "win32":
            suffix = "win_arm64.whl" if machine in {"arm64", "aarch64"} else "win_amd64.whl"
        elif sys.platform == "linux" and machine in {"x86_64", "amd64"}:
            suffix = "manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        elif sys.platform == "linux" and machine in {"arm64", "aarch64"}:
            suffix = "manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl"
        elif sys.platform == "darwin":
            suffix = "macosx_11_0_arm64.whl" if machine == "arm64" else "macosx_10_12_x86_64.whl"
        else:
            raise Failure("PLATFORM_UNSUPPORTED", "此平台需要手动安装 uv 后重试", 2)
        self.emit({"ok": True, "event": "init_stage", "stage": "uv", "state": "downloading"})
        with urllib.request.urlopen(f"https://pypi.org/pypi/uv/{UV_VERSION}/json", timeout=60) as response:
            metadata = json.load(response)
        wheel = next((item for item in metadata["urls"] if item["filename"].endswith(suffix)), None)
        if wheel is None:
            raise Failure("PLATFORM_UNSUPPORTED", "未找到此平台的 uv 二进制", 2)
        archive = private.parent / wheel["filename"]
        download(wheel["url"], archive, wheel["digests"]["sha256"])
        with zipfile.ZipFile(archive) as bundle:
            entry = next((name for name in bundle.namelist() if PurePosixPath(name).name == private.name), None)
            if entry is None:
                raise Failure("INVALID_UV_WHEEL", "uv wheel 不含预期程序")
            with bundle.open(entry) as source, private.open("xb") as output:
                shutil.copyfileobj(source, output)
        private.chmod(0o755)
        self.run("uv-verify", [private, "--version"])
        return str(private)

    def client_wheel(self, package, distribution):
        """Bundle current installed code, including editable installs, without build dependencies."""
        module = importlib.import_module(package)
        source = Path(module.__file__).parent
        dist = importlib.metadata.distribution(distribution)
        version = dist.version
        normalized = distribution.replace("-", "_")
        info = f"{normalized}-{version}.dist-info"
        target = self.home / "installers" / f"{normalized}-{version}-py3-none-any.whl"
        target.parent.mkdir(parents=True, exist_ok=True)
        records = []
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as wheel:
            def write(name, data):
                if isinstance(data, str):
                    data = data.encode("utf-8")
                wheel.writestr(name, data)
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                records.append((name, "sha256=" + digest, len(data)))
            for path in sorted(source.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json"}:
                    write(package + "/" + path.relative_to(source).as_posix(), path.read_bytes())
            write(info + "/METADATA", dist.read_text("METADATA"))
            write(info + "/WHEEL", "Wheel-Version: 1.0\nGenerator: cli-init\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
            entrypoints = dist.read_text("entry_points.txt")
            if entrypoints:
                write(info + "/entry_points.txt", entrypoints)
            output = io.StringIO(newline="")
            csv.writer(output).writerows([*records, (info + "/RECORD", "", "")])
            wheel.writestr(info + "/RECORD", output.getvalue())
        return target
