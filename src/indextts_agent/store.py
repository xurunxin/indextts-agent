import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .common import TERMINAL, Failure


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "jobs.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, idem TEXT UNIQUE, fingerprint TEXT NOT NULL,
                    request TEXT NOT NULL, state TEXT NOT NULL, created REAL NOT NULL,
                    started REAL, finished REAL, progress REAL DEFAULT 0,
                    phase TEXT DEFAULT 'queued', cancel_requested INTEGER DEFAULT 0,
                    error TEXT, result TEXT);
                CREATE TABLE IF NOT EXISTS assets (id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS voices (name TEXT PRIMARY KEY, asset TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def recover(self):
        with self.connect() as db:
            db.execute("UPDATE jobs SET state='interrupted', finished=?, error=? WHERE state='running'",
                       (time.time(), json.dumps({"code": "WORKER_INTERRUPTED", "message": "服务在生成时退出；请使用新幂等键重试"})))

    def submit(self, request, idem):
        raw = json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False)
        fingerprint = hashlib.sha256(raw.encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if idem:
                previous = db.execute("SELECT * FROM jobs WHERE idem=?", (idem,)).fetchone()
                if previous:
                    if previous["fingerprint"] != fingerprint:
                        raise Failure("IDEMPOTENCY_CONFLICT", "同一幂等键对应不同请求", 2)
                    return self.decode(previous), True
            count = db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]
            if count >= 1000:
                raise Failure("QUEUE_FULL", "队列已达到 1000 个任务上限")
            jid = uuid.uuid4().hex
            db.execute("INSERT INTO jobs (id,idem,fingerprint,request,state,created) VALUES (?,?,?,?,'queued',?)",
                       (jid, idem, fingerprint, raw, time.time()))
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()), False

    @staticmethod
    def decode(row):
        data = dict(row)
        for key in ("request", "error", "result"):
            data[key] = json.loads(data[key]) if data[key] else None
        data["cancel_requested"] = bool(data["cancel_requested"])
        return data

    def get(self, jid):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if row is None:
            raise Failure("JOB_NOT_FOUND", "找不到任务", 2)
        return self.decode(row)

    def list(self, limit=50):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,))]

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            if not row:
                return None
            db.execute("UPDATE jobs SET state='running',started=?,phase='starting' WHERE id=?", (time.time(), row[0]))
        return self.get(row[0])

    def progress(self, jid, value, phase):
        with self.connect() as db:
            db.execute("UPDATE jobs SET progress=MAX(progress,?),phase=? WHERE id=? AND state='running'",
                       (min(0.99, max(0, value)), phase, jid))

    def finish(self, jid, state, result=None, error=None):
        with self.connect() as db:
            db.execute("UPDATE jobs SET state=?,finished=?,progress=?,phase=?,result=?,error=? WHERE id=?",
                       (state, time.time(), 1 if state == "succeeded" else 0, state,
                        json.dumps(result) if result else None, json.dumps(error) if error else None, jid))

    def cancel(self, jid):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state FROM jobs WHERE id=?", (jid,)).fetchone()
            if row is None:
                raise Failure("JOB_NOT_FOUND", "找不到任务", 2)
            if row[0] not in TERMINAL:
                db.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (jid,))
                if row[0] == "queued":
                    db.execute("UPDATE jobs SET state='cancelled',finished=?,phase='cancelled' WHERE id=?", (time.time(), jid))
        return self.get(jid)

    def add_asset(self, metadata):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO assets VALUES (?,?)", (metadata["id"], json.dumps(metadata)))

    def asset(self, aid):
        with self.connect() as db:
            row = db.execute("SELECT metadata FROM assets WHERE id=?", (aid,)).fetchone()
        if not row:
            raise Failure("ASSET_NOT_FOUND", "请先上传参考音频", 2)
        return json.loads(row[0])

    def voices(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM voices ORDER BY name")]

    def add_voice(self, name, asset):
        self.asset(asset)
        with self.connect() as db:
            row = db.execute("SELECT asset FROM voices WHERE name=?", (name,)).fetchone()
            if row and row[0] != asset:
                raise Failure("VOICE_EXISTS", "音色名已存在，请使用另一个名称", 2)
            db.execute("INSERT OR IGNORE INTO voices VALUES (?,?)", (name, asset))
        return {"name": name, "asset": asset}
