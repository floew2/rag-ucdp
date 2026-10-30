# syntax=docker/dockerfile:1

# ---- Build stage: resolve dependencies with uv from the lockfile -------------
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies first, so code changes do not invalidate this layer.
# The locked torch wheel for Linux pulls in several GB of CUDA libraries
# (nvidia-*, cuda-*, triton). The image runs on CPU, so those are skipped and
# torch is installed from PyTorch's CPU-only index at the locked version, on
# both amd64 and arm64. The exported list is complete and pinned, so --no-deps
# stops resolution from pulling the CUDA build back in through torch's dependents.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-dev --no-emit-project --no-hashes -o /tmp/requirements.txt \
    && torch_pin="$(grep -oE '^torch==[^ ;]+' /tmp/requirements.txt)" \
    && grep -vE '^(nvidia-|cuda-|triton==|torch==)' /tmp/requirements.txt > /tmp/requirements-cpu.txt \
    && uv venv /app/.venv \
    && uv pip install --python /app/.venv/bin/python --no-deps -r /tmp/requirements-cpu.txt \
    && uv pip install --python /app/.venv/bin/python --no-deps \
         --index-url https://download.pytorch.org/whl/cpu "$torch_pin"

# The project itself, installed in editable mode: paths in config.yaml are
# resolved relative to the repository root, i.e. /app.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python --no-deps -e .


# ---- Runtime stage ------------------------------------------------------------
FROM python:3.12-slim-bookworm

RUN useradd --create-home --uid 1000 app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --chown=app:app src ./src
COPY --chown=app:app app ./app
COPY --chown=app:app config ./config
COPY --chown=app:app data/processed ./data/processed

RUN mkdir -p data/chroma_db data/bm25 /home/app/.cache/huggingface \
    && chown -R app:app data /home/app/.cache

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"

CMD ["rag-ucdp", "app", "--server.address=0.0.0.0", "--server.port=8501"]
