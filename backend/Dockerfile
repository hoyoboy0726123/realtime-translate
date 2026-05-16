# Backend image — FastAPI + uvicorn, dependencies installed with uv.
# Covers the "cloud" and "mock" engines. The on-device "local" engine
# (MLX-Whisper) needs Apple Silicon and is best run natively (see README).
FROM python:3.12-slim

# Bring in the uv binary from its official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
