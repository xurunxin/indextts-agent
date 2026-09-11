import json
import os
import stat
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMOTIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


class Failure(Exception):
    def __init__(self, code, message, exit_code=1, details=None):
        super().__init__(message)
        self.code, self.exit_code, self.details = code, exit_code, details


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def is_reparse_point(path):
    """Return whether *path* is a symlink/junction (without following it)."""

    path = Path(path)
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    except FileNotFoundError:
        return False
    except OSError:
        # An unreadable path cannot be safely classified for a destructive
        # operation. Treat it as unsafe and let the caller report the path.
        return True


class InstanceLock:
    """Kernel-owned lock: automatically released after a process crash."""
    def __init__(self, path):
        self.path, self.file = Path(path), None

    def __enter__(self):
        if is_reparse_point(self.path):
            raise Failure("UNSAFE_PATH", "锁文件不能是符号链接或 junction", 2, {"path": str(self.path)})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.file = self.path.open("a+b")
            self.file.seek(0)
            if self.file.read(1) == b"":
                self.file.write(b"0")
                self.file.flush()
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if self.file is not None:
                self.file.close()
            raise Failure("SERVICE_LOCKED", "该数据目录已有服务运行")
        return self

    def __exit__(self, *args):
        self.file.close()
