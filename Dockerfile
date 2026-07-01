# RunPod serverless worker для lip-sync (Content Factory).
# Первичная модель — MuseTalk (audio-driven, realtime). LivePortrait — опционально (A/B).
#
# ВНИМАНИЕ: точные версии torch/cuda и путь realtime-inference у MuseTalk чувствительны —
# ПРОВЕРИТЬ на первом GPU-прогоне (см. runpod/README_RUNPOD.md, раздел «Проверить на GPU»).
# Логика handler'а от этого не зависит: команду inference можно переопределить env
# MUSETALK_CMD/MUSETALK_DIR без пересборки логики.

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- MuseTalk -----------------------------------------------------------------
# Клонируем и ставим зависимости модели. Веса тянем скриптом репозитория.
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /opt/MuseTalk
WORKDIR /opt/MuseTalk
RUN pip install --no-cache-dir -r requirements.txt || true
# Веса моделей (musetalk, sd-vae, whisper, dwpose и т.п.). Если скрипт называется иначе —
# поправить здесь по README MuseTalk.
RUN bash ./download_weights.sh || python -m scripts.download_weights || true

# --- слой handler'а -----------------------------------------------------------
RUN pip install --no-cache-dir runpod boto3 httpx

# Болванки-аватары. DIMA.mp4 должен лежать в build-контексте: runpod/avatars/dima.mp4
# (либо подключить RunPod network volume в /avatars и не бейкать в образ).
RUN mkdir -p /avatars
COPY avatars/ /avatars/

COPY scripts/runpod_handler.py /opt/runpod_handler.py

ENV AVATARS_DIR=/avatars \
    MUSETALK_DIR=/opt/MuseTalk \
    RUNPOD_DEFAULT_MODEL=musetalk

CMD ["python", "-u", "/opt/runpod_handler.py"]
