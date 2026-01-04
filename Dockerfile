# Builder stage: Install dependencies with uv package manager
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

# UV package manager configuration
# UV_COMPILE_BYTECODE: Pre-compile Python files to .pyc for faster startup
# UV_LINK_MODE: Copy files instead of hardlinking for reliability
# UV_PYTHON_DOWNLOADS: Don't download Python (use system Python)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install dependencies only (cached layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-workspace --no-dev

# Copy application code
COPY pyproject.toml uv.lock /app/
COPY frontend /app/frontend

# Install workspace package
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Runtime stage: Minimal image
FROM python:3.11-slim-bookworm AS runtime

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/frontend /app/frontend

# Python runtime configuration
# PATH: Add virtualenv binaries to PATH
# PYTHONPATH: Python module search paths
# PYTHONUNBUFFERED: Disable output buffering for real-time logs
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1

WORKDIR /app

EXPOSE 8501

# Start Streamlit on all interfaces
CMD ["streamlit", "run", \
     "frontend/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
