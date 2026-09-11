"""Compatibility launcher; deployment logic lives in itt init, including uv bootstrap."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from indextts_agent.provision import Installer

p = argparse.ArgumentParser(description="安装 CLI 并调用 itt init；无 Python 的 Windows 请用 install.ps1")
p.add_argument("--home", type=Path, default=Path.home() / ".indextts-agent")
p.add_argument("--cpu", action="store_true")
p.add_argument("--examples", action="store_true", help="兼容旧参数；示例音频不属于初始化模型清单")
p.add_argument("--skip-cli", action="store_true", help="通过 uv tool run 临时安装启动器，不注册全局 CLI")
args, remaining = p.parse_known_args()
if args.examples:
    print("--examples 不再自动下载；可使用已有参考音频验证模型。", file=sys.stderr)
installer = Installer(args.home, "itt", lambda event: print(json.dumps(event), flush=True))
uv = installer.ensure_uv()
if args.skip_cli:
    command = [uv, "tool", "run", "--from", str(ROOT), "itt"]
else:
    installer.run("cli", [uv, "tool", "install", "--force", "--editable", ROOT])
    directory = subprocess.check_output([uv, "tool", "dir", "--bin"], text=True).strip()
    command = [str(Path(directory) / ("itt.exe" if os.name == "nt" else "itt"))]
command.extend(["--home", str(args.home), "init"])
if args.cpu:
    command.extend(["--device", "cpu"])
raise SystemExit(subprocess.run([*command, *remaining], check=False).returncode)
