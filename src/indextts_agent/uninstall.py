"""Remove only a marked runtime, keeping user data and shared caches."""
import os
import shutil
import stat
import sys
from pathlib import Path

from .common import Failure, InstanceLock, read_json


def arguments(sub):
    p = sub.add_parser("uninstall", help="卸载受管运行环境和模型，保留音色、任务及客户端")
    p.add_argument("--install-dir", type=Path, help="默认使用 init 保存的目录；也可清理未完成的安装")
    p.add_argument("--dry-run", action="store_true", help="只读预览，不删除文件")
    p.add_argument("--yes", action="store_true", help="确认删除受管环境及模型")


def reject_links(path, ancestors=True):
    for part in (path, *path.parents) if ancestors else (path,):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise Failure("UNSAFE_PATH", "卸载路径包含符号链接或 junction", 2, {"path": str(part)})


def plan(args):
    home = Path(os.path.abspath(args.home))
    reject_links(home)
    reject_links(home / "runtime.json")
    reject_links(home / "init.lock")
    reject_links(home / "service.lock")
    reject_links(home / "start.lock")
    saved = read_json(home / "runtime.json", {})
    if not isinstance(saved, dict):
        raise Failure("INVALID_CONFIG", "runtime.json 必须是配置对象", 2)
    root = Path(os.path.abspath(args.install_dir or saved.get("install_dir", home / "runtime")))
    reject_links(root)
    root = root.resolve()
    protected = (home.resolve(), Path.home().resolve(), Path.cwd().resolve(), Path(sys.executable).resolve(), Path(__file__).resolve())
    if root == Path(root.anchor) or any(p == root or root in p.parents for p in protected):
        raise Failure("UNSAFE_PATH", "拒绝卸载数据目录、系统根目录或正在使用的程序目录", 2)
    if root.exists():
        reject_links(root / ".itt-runtime.json")
        marker = read_json(root / ".itt-runtime.json", {})
        if (not root.is_dir() or not isinstance(marker, dict) or marker.get("schema") != 1
                or marker.get("repository") != "https://github.com/index-tts/index-tts.git"
                or not isinstance(marker.get("revision"), str) or len(marker["revision"]) != 40):
            raise Failure("UNOWNED_DIRECTORY", "目录不属于 itt init，拒绝删除", 2)
        # Validate every descendant before removing anything; never traverse a junction.
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name in dirs + files:
                reject_links(Path(directory) / name, ancestors=False)
    remove_config = bool(saved.get("install_dir") and Path(saved["install_dir"]).resolve() == root)
    return {"install_dir": str(root), "exists": root.exists(), "remove_runtime_config": remove_config,
            "preserved": ["任务、音色和日志", "导出的音频", "共享 uv/Python/Hugging Face 缓存", "itt 客户端"],
            "client_uninstall": "uv tool uninstall indextts-agent"}


def uninstall(args):
    report = plan(args)
    if args.dry_run:
        return {"mode": "dry-run", **report}
    if not args.yes:
        raise Failure("CONFIRMATION_REQUIRED", "先用 uninstall --dry-run 预览，再加 --yes 执行删除", 2, report)
    home = Path(args.home).resolve()
    if not report["exists"] and not report["remove_runtime_config"]:
        return {"state": "already_uninstalled", **report}
    with InstanceLock(home / "init.lock"), InstanceLock(home / "start.lock"), InstanceLock(home / "service.lock"):
        report = plan(args)
        root = Path(report["install_dir"])
        if root.exists():
            # Keep ownership evidence until all payloads are gone, allowing retries.
            marker = root / ".itt-runtime.json"
            for child in root.iterdir():
                if child == marker:
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            marker.unlink()
            root.rmdir()
        if report["remove_runtime_config"]:
            (home / "runtime.json").unlink()
    return {"state": "uninstalled", **report}
