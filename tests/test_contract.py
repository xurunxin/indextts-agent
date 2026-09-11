import hashlib
import io
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from indextts_agent.schema import JobRequest
from indextts_agent.server import create_app
from indextts_agent.store import Store


def wav_bytes():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 22050)
    return buf.getvalue()


@pytest.fixture
def api(tmp_path):
    config = {"data_dir": str(tmp_path), "api_key": "test-secret", "engine": "test", "qwen_emotion": False}
    with TestClient(create_app(config)) as client:
        client.headers["Authorization"] = "Bearer test-secret"
        yield client


def test_authenticated_upload_job_download_and_idempotency(api):
    assert api.get("/v1/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
    audio = wav_bytes()
    upload = api.post("/v1/assets", content=audio)
    assert upload.status_code == 200
    aid = upload.json()["id"]
    assert aid == hashlib.sha256(audio).hexdigest()
    assert api.post("/v1/assets", content=audio).json()["id"] == aid
    assert api.post("/v1/voices", json={"name": "narrator", "asset": aid}).status_code == 200
    request = {"text": "测试", "speaker": aid, "emotion_vector": [0.6, 0, 0, 0, 0, 0, 0, 0.2]}
    headers = {"Idempotency-Key": "scene-01-v1"}
    first = api.post("/v1/jobs", json=request, headers=headers).json()
    jid = first["job"]["id"]
    assert api.post("/v1/jobs", json=request, headers=headers).json()["replayed"]
    assert api.post("/v1/jobs", json={**request, "text": "different"}, headers=headers).status_code == 409
    for _ in range(100):
        job = api.get("/v1/jobs/" + jid).json()
        if job["state"] == "succeeded":
            break
        time.sleep(0.02)
    assert job["state"] == "succeeded"
    result = api.get(job["result"]["audio_url"])
    assert hashlib.sha256(result.content).hexdigest() == job["result"]["sha256"]
    assert job["result"]["backend"] == "test"
    assert job["result"]["warnings"]


def test_input_validation_and_missing_qwen(api):
    aid = api.post("/v1/assets", content=wav_bytes()).json()["id"]
    assert api.post("/v1/assets", content=b"not audio").status_code == 422
    req = {"text": "hello", "speaker": aid}
    assert api.post("/v1/jobs", json={**req, "emotion_auto": True}).json()["error"]["code"] == "TEXT_EMOTION_DISABLED"
    assert api.post("/v1/jobs", json={**req, "output_path": "/tmp/overwrite"}).status_code == 422
    assert api.post("/v1/jobs", json={**req, "speaker": "../../secret"}).status_code == 422
    assert api.post("/v1/jobs", json={**req, "duration_factor": 2.1}).status_code == 422
    assert api.post("/v1/jobs", json={**req, "emotion_text": "happy", "emotion_vector": [0] * 8}).status_code == 422


def test_concurrent_idempotency_claim_cancel_and_restart(tmp_path):
    store = Store(tmp_path)
    request = {"text": "test"}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: store.submit(request, "same-key"), range(16)))
    assert len({r[0]["id"] for r in results}) == 1
    running = store.claim()
    queued = store.submit(request, "second")[0]
    assert store.cancel(queued["id"])["state"] == "cancelled"
    assert store.cancel(running["id"])["cancel_requested"]
    Store(tmp_path).recover()
    assert store.get(running["id"])["state"] == "interrupted"
    assert store.claim() is None


@pytest.mark.parametrize("patch", [{"emotion_vector": [0.5] * 8}, {"emotion_vector": [float('nan')] * 8},
                                    {"text": "   "}, {"speaker": "file:///private.wav"}, {"lang": "YUE"}])
def test_schema_rejects_invalid_controls(patch):
    with pytest.raises(ValidationError):
        JobRequest.model_validate({"text": "test", "speaker": "a" * 64, **patch})
