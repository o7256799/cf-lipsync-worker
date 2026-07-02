# RunPod serverless worker для lip-sync — LatentSync 1.6 (ByteDance), 512px, резкий рот.
# CUDA 11.8 (а не 12.1): часть GPU-хостов RunPod имеют драйверы старее, чем требует cu121,
# и падают "no kernel image is available". cu118 совместим с гораздо более старыми драйверами.
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 wget curl ca-certificates \
        build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/bytedance/LatentSync.git /opt/LatentSync
WORKDIR /opt/LatentSync

# зависимости модели. requirements пинят torch==2.5.1 c индексом cu121 — переводим на cu118,
# чтобы torch/torchvision встали под cuda 11.8 (совпадает с базой), а не притащили cu121.
RUN sed -i 's|whl/cu121|whl/cu118|g' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall torch==2.5.1+cu118 torchvision==0.20.1+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

# веса: latentsync_unet.pt (~5 ГБ) + whisper/tiny.pt. Жёстко проверяем.
RUN set -eux; \
    huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints; \
    huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt     --local-dir checkpoints; \
    ls -la checkpoints checkpoints/whisper; \
    test -f checkpoints/latentsync_unet.pt; \
    test -f checkpoints/whisper/tiny.pt

# слой handler'а
# runpod ПИНИМ на 1.7.10: версии >=1.7.11 имеют баг балансировки (задачи виснут в очереди,
# всё уходит на один воркер) — runpod-python#432. Именно это ловили как "RunPod не отдаёт задачи".
RUN pip install --no-cache-dir "runpod==1.7.10" boto3 httpx

# полный статический ffmpeg первым в PATH (conda-ffmpeg без libx264 ломает кодирование)
RUN wget -q https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz && \
    tar xf ffmpeg-master-latest-linux64-gpl.tar.xz && \
    mv ffmpeg-master-latest-linux64-gpl /opt/ffmpeg-static && \
    rm -f ffmpeg-master-latest-linux64-gpl.tar.xz

RUN mkdir -p /avatars
COPY avatars/ /avatars/
COPY scripts/runpod_handler.py /opt/runpod_handler.py

ENV PATH="/opt/ffmpeg-static/bin:${PATH}" \
    AVATARS_DIR=/avatars \
    LATENTSYNC_DIR=/opt/LatentSync \
    RUNPOD_DEFAULT_MODEL=latentsync

CMD ["python", "-u", "/opt/runpod_handler.py"]
