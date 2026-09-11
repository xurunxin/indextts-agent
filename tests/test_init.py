import argparse
import json

import pytest

from indextts_agent import init
from indextts_agent.cli import runtime_paths
from indextts_agent.common import Failure


def init_args(home, **overrides):
    values = {
        "home": str(home),
        "install_dir": None,
        "device": None,
        "check": True,
        "dry_run": False,
        "skip_models": False,
        "start": False,
        "wait": 600,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_check_is_read_only_and_returns_planned_paths(tmp_path):
    home = tmp_path / "home"
    result = init.run(init_args(home))

    assert result["ready"] is False
    assert result["planned"]["install_dir"] == str((home / "runtime").resolve())
    assert not home.exists()


def test_cpu_is_rejected_before_any_write(tmp_path):
    home = tmp_path / "home"

    with pytest.raises(Failure) as caught:
        init.run(init_args(home, check=False, device="cpu"))

    assert caught.value.code == "CPU_UNSUPPORTED"
    assert not home.exists()


def test_check_reports_changed_runtime_profile(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    configured = home / "runtime"
    (home / "runtime.json").write_text(
        json.dumps(
            {
                "install_dir": str(configured),
                "upstream": str(configured / "upstream"),
                "model_dir": str(configured / "models"),
                "runtime_python": str(configured / ".venv" / "Scripts" / "python.exe"),
                "device": "cuda",
            }
        ),
        encoding="utf-8",
    )

    result = init.run(init_args(home, install_dir=tmp_path / "other"))

    assert result["checks"]["config_compatible"] is False
    assert "install_dir" in result["check_details"]["config_conflicts"]
    assert not (tmp_path / "other").exists()


def test_runtime_profile_precedence(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    profile = tmp_path / "profile-runtime"
    (home / "runtime.json").write_text(
        json.dumps(
            {
                "upstream": str(profile / "upstream"),
                "model_dir": str(profile / "models"),
                "runtime_python": str(profile / ".venv" / "Scripts" / "python.exe"),
                "device": "cuda",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ITT_UPSTREAM", str(tmp_path / "env-upstream"))
    monkeypatch.setenv("ITT_MODEL_DIR", str(tmp_path / "env-models"))
    monkeypatch.setenv("ITT_DEVICE", "cpu")

    args = argparse.Namespace(upstream=tmp_path / "explicit-upstream", model_dir=None, device="cuda", runtime_python=None)
    upstream, model_dir, device, _runtime = runtime_paths(args, home)

    assert upstream == (tmp_path / "explicit-upstream").resolve()
    assert model_dir == (tmp_path / "env-models").resolve()
    assert device == "cuda"
