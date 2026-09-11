"""Real model acceptance; requires an already running authenticated service."""
import argparse
import json
import time
from pathlib import Path

from indextts_agent.cli import connection, parser
from indextts_agent.common import TERMINAL, atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-emotion-only", action="store_true")
    p.add_argument("--run-id", default="acceptance-v2")
    args = p.parse_args()
    client = connection(parser().parse_args(["service", "status"]))
    status = client.call("/v1/status")
    assert status["state"] == "ready" and status["backend"] == "indextts", status
    speaker = client.upload(ROOT / "upstream/examples/voice_01.wav")["id"]
    emotion = client.upload(ROOT / "upstream/examples/emo_sad.wav")["id"]
    text = "你好，欢迎使用语音生成工具。今天真是令人开心的一天！"
    vector = [0.6, 0, 0, 0, 0, 0, 0, 0]
    cases = [("happy-warm", {"text": text, "emotion_vector": vector}),
             ("happy-fast", {"text": text, "emotion_vector": vector, "duration_factor": 0.8}),
             ("sad-vector", {"text": text, "emotion_vector": [0, 0, 0.6, 0, 0, 0, 0, 0]}),
             ("sad-reference", {"text": "对不起，我真的很想念你。", "emotion_audio": emotion, "emotion_alpha": 0.7}),
             ("pronunciation", {"text": "他在银<行|HANG2>工作，喜欢步<行|XING2>回家。"}),
             ("beams-three", {"text": text, "emotion_vector": vector, "num_beams": 3})]
    if args.text_emotion_only:
        cases = [("text-emotion", {"text": "太好了，我们终于成功了！", "emotion_text": "开心而兴奋", "emotion_alpha": 0.6})]
    records = []
    for name, controls in cases:
        body = {"speaker": speaker, "lang": "ZH", "seed": 42, "num_beams": 1, **controls}
        response = client.call("/v1/jobs", "POST", body, headers={"Idempotency-Key": args.run_id + "-" + name})
        jid = response["job"]["id"]
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            job = client.call("/v1/jobs/" + jid)
            if job["state"] in TERMINAL:
                break
            time.sleep(0.5)
        if job["state"] != "succeeded":
            raise RuntimeError(json.dumps(job, ensure_ascii=False))
        path = ROOT / "outputs" / (name + ".wav")
        downloaded = client.download(jid, path, force=True)
        record = {"case": name, "job_id": jid, "request": job["request"], "result": downloaded}
        records.append(record)
        print(json.dumps({"case": name, "job_id": jid, "result": downloaded}, ensure_ascii=False), flush=True)
    target = ROOT / "outputs" / ("text-emotion-acceptance.json" if args.text_emotion_only else "acceptance.json")
    atomic_json(target, {"status": status, "run_id": args.run_id, "cases": records})


if __name__ == "__main__":
    main()
