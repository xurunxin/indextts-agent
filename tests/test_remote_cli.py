"""Real TCP boundary with different client/server directories; test engine only."""
import json
import os
import socket
import subprocess
import sys
import time

import pytest
from test_contract import wav_bytes

from indextts_agent.client import Client
from indextts_agent.common import Failure


@pytest.fixture
def remote(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "ITT_API_KEY": "remote-contract-secret", "PYTHONUTF8": "1"}
    url = f"http://127.0.0.1:{port}"
    log = (tmp_path / "remote.log").open("w")
    server_home = tmp_path / "server"
    proc = subprocess.Popen([sys.executable, "-m", "indextts_agent.cli", "--home", str(server_home),
                             "serve", "--port", str(port), "--engine", "test"], env=env, stdout=log, stderr=log)
    client = Client(url, env["ITT_API_KEY"], 1)
    try:
        for _ in range(100):
            try:
                if client.call("/v1/status")["state"] == "ready":
                    break
            except Failure:
                pass
            time.sleep(0.05)
        else:
            pytest.fail("TCP server did not start")
        def cli(*args, expected=0):
            result = subprocess.run([sys.executable, "-m", "indextts_agent.cli", "--home", str(tmp_path / "client"),
                                     "--url", url, *map(str, args)], env=env, capture_output=True, text=True, encoding="utf-8", timeout=15, check=False)
            assert result.returncode == expected, (result.stdout, result.stderr)
            return [json.loads(line) for line in result.stdout.splitlines()]
        yield client, cli, proc
    finally:
        if proc.poll() is None:
            try:
                client.call("/v1/service/stop", "POST")
                proc.wait(10)
            except (Failure, subprocess.TimeoutExpired):
                proc.kill()
                proc.wait(10)
        log.close()


def test_remote_cli_upload_monitor_download_and_cancel(remote, tmp_path):
    client, cli, proc = remote
    audio = tmp_path / "only-on-client.wav"
    audio.write_bytes(wav_bytes())
    assert cli("voices", "add", "narrator", "--audio", audio)[0]["name"] == "narrator"
    submit = cli("jobs", "submit", "--voice", "narrator", "--text", "你好", "--emotion", "happy=0.6",
                 "--speed", "1.25", "--idempotency-key", "remote-v1")[0]
    jid = submit["job"]["id"]
    assert submit["job"]["request"]["duration_factor"] == 0.8
    cli("jobs", "watch", jid, "--wait", "5", "--interval", "0.1")
    target = tmp_path / "client-output.wav"
    cli("jobs", "download", jid, "--output", target)
    assert target.read_bytes()[:4] == b"RIFF"
    assert cli("jobs", "download", jid, "--output", target, expected=2)[0]["error"]["code"] == "OUTPUT_EXISTS"
    # Retry doesn't synthesize again, even after completion.
    retry = cli("jobs", "submit", "--voice", "narrator", "--text", "你好", "--emotion", "happy=0.6",
                "--speed", "1.25", "--idempotency-key", "remote-v1")[0]
    assert retry["replayed"] and retry["job"]["id"] == jid
    assert cli("jobs", "submit", "--voice", "narrator", "--text", "不同文本", "--idempotency-key", "remote-v1", expected=2)[0]["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    cancelled = client.call("/v1/jobs", "POST", {"text": "取消", "speaker": submit["job"]["request"]["speaker"]})["job"]["id"]
    client.call(f"/v1/jobs/{cancelled}/cancel", "POST")
    cli("jobs", "watch", cancelled, "--wait", "5", "--interval", "0.1", expected=4)
    cli("service", "stop", "--wait", "5")
    proc.wait(5)


def test_shutdown_drains_active_job_and_preserves_queue(remote, tmp_path):
    client, _cli, proc = remote
    aid = client.call("/v1/assets", "POST", raw=wav_bytes())["id"]
    first = client.call("/v1/jobs", "POST", {"text": "active", "speaker": aid})["job"]["id"]
    second = client.call("/v1/jobs", "POST", {"text": "queued", "speaker": aid})["job"]["id"]
    for _ in range(50):
        if client.call("/v1/status")["active_job"]:
            break
        time.sleep(0.01)
    stopped = client.call("/v1/service/stop", "POST")
    assert stopped["active_job"] == first
    proc.wait(10)
    from indextts_agent.store import Store
    store = Store(tmp_path / "server")
    assert store.get(first)["state"] == "succeeded"
    assert store.get(second)["state"] == "queued"
