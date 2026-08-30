FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/hf \
    TORCH_HOME=/app/.cache/torch \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    DETECTOR_CHECKPOINT=/app/best.pt \
    DETECTOR_DEVICE=cpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      libjpeg62-turbo libpng16-16 && rm -rf /var/lib/apt/lists/*

# CPU-only torch — keeps the image ~2 GB smaller than the CUDA bundle.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
      torch==2.6.0 torchvision==0.21.0

COPY space/requirements.txt .
RUN pip install -r requirements.txt

COPY detector ./detector
COPY webapp ./webapp
COPY best.pt ./best.pt

# Bake the CLIP-B download into the image and prove the checkpoint loads, so
# a broken build fails here rather than at first request.
RUN python -c "from detector.model import load_checkpoint; load_checkpoint('best.pt', device='cpu'); print('warmup ok')"

EXPOSE 7860
CMD ["uvicorn", "webapp.server:app", "--host", "0.0.0.0", "--port", "7860"]
