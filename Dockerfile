FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONUTF8=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git libsndfile1 ffmpeg libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /bin/
ARG INDEXTTS_REVISION=ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c
RUN git clone --no-checkout --filter=blob:none https://github.com/index-tts/index-tts.git /opt/indextts && cd /opt/indextts && git checkout --detach ${INDEXTTS_REVISION}
WORKDIR /opt/indextts
RUN uv sync --frozen --no-dev --python 3.11
COPY pyproject.toml /opt/agent/pyproject.toml
COPY src /opt/agent/src
RUN uv pip install --python /opt/indextts/.venv/bin/python '/opt/agent[server]'
ENV PATH=/opt/indextts/.venv/bin:$PATH
EXPOSE 8095
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m CMD python -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8095/health'))['state']=='ready'"
ENTRYPOINT ["itt"]
CMD ["--home", "/data", "serve", "--host", "0.0.0.0", "--upstream", "/opt/indextts", "--model-dir", "/opt/indextts/checkpoints"]
