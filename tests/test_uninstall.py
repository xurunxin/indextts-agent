import json
from pathlib import Path

import pytest

from indextts_agent.cli import parser, run
from indextts_agent.common import Failure, InstanceLock, atomic_json


def setup(tmp_path):
    home, root = tmp_path / "home", tmp_path / "runtime"
    atomic_json(root / ".itt-runtime.json", {"schema": 1, "repository": "https://github.com/index-tts/index-tts.git", "revision": "a" * 40})
    atomic_json(home / "runtime.json", {"install_dir": str(root)})
    (home / "voices.json").write_text("keep")
    (root / "model.bin").write_bytes(b"model")
    return home, root


def invoke(home, *extra):
    return run(parser().parse_args(["--home", str(home), "uninstall", *extra]))


def test_preview_confirm_remove_and_repeat(tmp_path):
    home, root = setup(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert invoke(home, "--dry-run")["exists"]
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(Failure, match="--yes"):
        invoke(home)
    assert root.exists()
    assert invoke(home, "--yes")["state"] == "uninstalled"
    assert not root.exists() and not (home / "runtime.json").exists()
    assert (home / "voices.json").read_text() == "keep"
    assert invoke(home, "--yes")["state"] == "already_uninstalled"


def test_unowned_and_protected(tmp_path):
    home, root = setup(tmp_path)
    (root / ".itt-runtime.json").unlink()
    with pytest.raises(Failure, match="不属于"):
        invoke(home, "--yes")
    with pytest.raises(Failure, match="拒绝卸载"):
        invoke(home, "--yes", "--install-dir", str(tmp_path))
    assert (root / "model.bin").exists()


def test_active_service_and_install_refused(tmp_path):
    home, root = setup(tmp_path)
    for lock in ("service.lock", "init.lock", "start.lock"):
        with InstanceLock(home / lock), pytest.raises(Failure):
            invoke(home, "--yes")
    assert root.exists()


def test_other_install_preserves_config(tmp_path):
    home, root = setup(tmp_path)
    other = tmp_path / "other"
    atomic_json(other / ".itt-runtime.json", json.loads((root / ".itt-runtime.json").read_text()))
    invoke(home, "--yes", "--install-dir", str(other))
    assert root.exists() and (home / "runtime.json").exists()


def test_link_rejected_before_delete(tmp_path, monkeypatch):
    home, root = setup(tmp_path)
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p.name == "model.bin" or original(p))
    with pytest.raises(Failure, match="链接"):
        invoke(home, "--yes")
    assert (root / "model.bin").exists()


def test_missing_home_preview_no_write(tmp_path):
    home = tmp_path / "missing"
    assert not invoke(home, "--dry-run")["exists"]
    assert not home.exists()


def test_partial_delete_keeps_ownership_for_retry(tmp_path, monkeypatch):
    home, root = setup(tmp_path)
    original = Path.unlink
    def fail_model(path, *args, **kwargs):
        if path.name == "model.bin":
            raise PermissionError("in use")
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_model)
        with pytest.raises(PermissionError):
            invoke(home, "--yes")
    assert (root / ".itt-runtime.json").exists()
    assert invoke(home, "--yes")["state"] == "uninstalled"
