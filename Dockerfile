# WhisperX forced-alignment worker for RunPod Serverless (built by RunPod from GitHub).
# Every version is pinned: base image by digest, Python packages in requirements.txt.
FROM --platform=linux/amd64 runpod/base:1.4.0-cuda1281-ubuntu2204@sha256:6a1a66d92c5a9946c2748415c923321ed816c26f5d331a2389f4646145611965

# RUNPOD_LOG_LEVEL: the SDK defaults to DEBUG, which logs every handler output.
ENV PYTHONUNBUFFERED=1 \
    RUNPOD_LOG_LEVEL=INFO \
    MODEL_DIR=/models \
    NLTK_DATA=/models/nltk_data \
    PATH=/opt/venv/bin:$PATH

# Own venv on the base's Python 3.11 (ffmpeg comes with the base image).
RUN python3.11 -m venv /opt/venv
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-deps -r /app/requirements.txt && pip check

# Model weights baked in at build time, never fetched on cold start.
COPY download_models.py /app/download_models.py
RUN python /app/download_models.py

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

WORKDIR /app
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
