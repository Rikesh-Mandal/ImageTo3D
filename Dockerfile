
# Python runtime for FastAPI and ImageTo3D.
FROM python:3.12-slim-bookworm

# Python and CPU execution settings.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=2 \
    U2NET_HOME=/app/.u2net

WORKDIR /app

# System libraries required by Open3D/OpenCV.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 \
        libgl1 \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency declarations first for build caching.
COPY requirements.txt ./requirements.txt

# Install CPU-only PyTorch.
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        torch==2.5.1 \
        torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu

# Install FastAPI, Open3D, Rembg and other dependencies.
RUN python -m pip install -r requirements.txt

# Copy only application and source code.
COPY app/ ./app/
COPY src/ ./src/

# Create an unprivileged application user.
RUN groupadd --system --gid 10001 appuser \
    && useradd \
        --system \
        --uid 10001 \
        --gid 10001 \
        --home-dir /app \
        appuser \
    && mkdir -p \
        /app/saved_models \
        /app/outputs/api \
        /app/.u2net \
    && chown -R appuser:appuser \
        /app/saved_models \
        /app/outputs \
        /app/.u2net

USER appuser

EXPOSE 8001

# One worker preserves the single-model,
# single-CPU-reconstruction design.
CMD ["python","-m","uvicorn","app.main:app","--host","0.0.0.0","--port","8001","--workers","1"]
