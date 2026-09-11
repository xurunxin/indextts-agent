import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .common import Failure


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Client:
    def __init__(self, url, key, timeout=30):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise Failure("INVALID_URL", "服务 URL 必须是无凭据、无查询参数的 HTTP(S) 地址", 2)
        self.url, self.key, self.timeout = url.rstrip("/"), key, timeout
        self.opener = urllib.request.build_opener(NoRedirect())

    def call(self, path, method="GET", body=None, raw=None, headers=None, binary=False):
        data = raw if raw is not None else json.dumps(body, ensure_ascii=False, allow_nan=False).encode() if body is not None else None
        request_headers = {"Authorization": "Bearer " + self.key, "Content-Type": "application/octet-stream" if raw is not None else "application/json"}
        request_headers.update(headers or {})
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=request_headers)
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                result = response.read()
            return result if binary else json.loads(result)
        except urllib.error.HTTPError as e:
            try:
                result = json.loads(e.read())
                error = result.get("error", {})
                if not isinstance(error, dict):
                    error = {}
            except (ValueError, AttributeError):
                error = {}
            raise Failure(error.get("code", "HTTP_ERROR"), error.get("message", f"HTTP {e.code}"), 2, {"http_status": e.code}) from None
        except (urllib.error.URLError, OSError, TimeoutError):
            raise Failure("CONNECTION_FAILED", "无法连接服务或请求超时；检查 URL、网络与服务状态") from None

    def upload(self, path):
        path = Path(path).resolve()
        if not path.is_file():
            raise Failure("FILE_NOT_FOUND", f"找不到音频: {path}", 2)
        if path.stat().st_size > 40 * 1024 * 1024:
            raise Failure("ASSET_TOO_LARGE", "参考音频上限 40 MiB", 2)
        return self.call("/v1/assets", "POST", raw=path.read_bytes())

    def download(self, jid, output, force=False):
        job = self.call("/v1/jobs/" + urllib.parse.quote(jid, safe=""))
        if job["state"] != "succeeded":
            raise Failure("RESULT_NOT_READY", f"任务状态: {job['state']}", 2)
        target = Path(output).resolve()
        if target.exists() and not force:
            raise Failure("OUTPUT_EXISTS", "结果文件已存在；选择新路径或使用 --force", 2)
        audio = self.call(job["result"]["audio_url"], binary=True)
        if hashlib.sha256(audio).hexdigest() != job["result"]["sha256"]:
            raise Failure("CHECKSUM_MISMATCH", "下载结果的 SHA-256 校验失败")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(audio)
            if force:
                os.replace(tmp, target)
            else:
                # Exclusive creation prevents clobbering a file created during download.
                # Hardlink is atomic and same-filesystem because tmp lives beside target.
                os.link(tmp, target)
        finally:
            Path(tmp).unlink(missing_ok=True)
        return {"job_id": jid, "output": str(target), **job["result"]}
