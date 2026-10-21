# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG UID=1000
ARG GID=1000
ARG TORCH_VARIANT=cpu
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    XDG_CACHE_HOME=/tmp/.cache \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install locked third-party dependencies first so application-only changes
# keep the expensive PyTorch/Ultralytics layer cached.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "${TORCH_VARIANT}" = "cpu" ]; then \
        uv venv /app/.venv \
        && uv pip install --python /app/.venv/bin/python \
            --default-index "${TORCH_CPU_INDEX_URL}" \
            "torch==${TORCH_VERSION}" \
            "torchvision==${TORCHVISION_VERSION}" \
        && uv export --frozen --no-dev --no-hashes --no-emit-project \
            --prune torch --prune torchvision \
            --output-file /tmp/requirements.lock \
        && uv pip install --python /app/.venv/bin/python \
            --requirements /tmp/requirements.lock; \
    elif [ "${TORCH_VARIANT}" = "cuda" ]; then \
        uv sync --frozen --no-dev --no-install-project; \
    else \
        echo "TORCH_VARIANT must be 'cpu' or 'cuda'" >&2; \
        exit 2; \
    fi

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "${TORCH_VARIANT}" = "cpu" ]; then \
        uv pip install --python /app/.venv/bin/python --no-deps . \
        && uv pip check --python /app/.venv/bin/python; \
    else \
        uv sync --frozen --no-dev --no-editable; \
    fi

RUN groupadd --gid "${GID}" app \
    && useradd --uid "${UID}" --gid "${GID}" --create-home app \
    && mkdir -p /app/data /app/outputs /tmp/Ultralytics/Ultralytics \
    && chown -R app:app /app/outputs /home/app /tmp/Ultralytics

USER app

VOLUME ["/app/data", "/app/outputs"]

# Override this command with evaluate, render-preview, demo, etc.
CMD ["evaluate", "--help"]
