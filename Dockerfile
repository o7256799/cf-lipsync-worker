# RunPod serverless worker для lip-sync (Content Factory) — MuseTalk v1.5.
# Установка по официальному README MuseTalk: requirements + mmlab(openmim) + веса + ffmpeg.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs ffmpeg libgl1 libglib2.0-0 wget curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /opt/MuseTalk
WORKDIR /opt/MuseTalk

# 1) базовые зависимости модели
RUN pip install --no-cache-dir -r requirements.txt

# 2) mmlab-стек для DWPose (ставится через openmim, иначе не собирается)
RUN pip install --no-cache-dir -U openmim && \
    mim install "mmengine" "mmcv==2.1.0" "mmdet==3.2.0" "mmpose==1.2.0"

# 3) утилиты загрузчика весов
RUN pip install --no-cache-dir "huggingface_hub[cli]" gdown

# 4) статический ffmpeg, которого ждёт realtime_inference по умолчанию
RUN wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar xf ffmpeg-release-amd64-static.tar.xz && \
    mv ffmpeg-*-amd64-static ffmpeg-4.4-amd64-static && \
    rm -f ffmpeg-release-amd64-static.tar.xz

# 5) веса — качаем КАЖДЫЙ файл отдельной командой (позиционно). Не используем download_weights.sh:
#    там multi-include (`hf download repo --include "a" "b"`) отдаёт не все файлы (мелкие .json теряются).
RUN pip install --no-cache-dir gdown && set -eux; \
    mkdir -p models/musetalk models/musetalkV15 models/sd-vae models/whisper models/dwpose models/syncnet models/face-parse-bisent; \
    hf download TMElyralab/MuseTalk musetalk/musetalk.json          --local-dir models; \
    hf download TMElyralab/MuseTalk musetalk/pytorch_model.bin      --local-dir models; \
    hf download TMElyralab/MuseTalk musetalkV15/musetalk.json       --local-dir models; \
    hf download TMElyralab/MuseTalk musetalkV15/unet.pth            --local-dir models; \
    hf download stabilityai/sd-vae-ft-mse config.json               --local-dir models/sd-vae; \
    hf download stabilityai/sd-vae-ft-mse diffusion_pytorch_model.bin --local-dir models/sd-vae; \
    hf download openai/whisper-tiny config.json                     --local-dir models/whisper; \
    hf download openai/whisper-tiny pytorch_model.bin               --local-dir models/whisper; \
    hf download openai/whisper-tiny preprocessor_config.json        --local-dir models/whisper; \
    hf download yzd-v/DWPose dw-ll_ucoco_384.pth                    --local-dir models/dwpose; \
    hf download ByteDance/LatentSync latentsync_syncnet.pt          --local-dir models/syncnet || true; \
    gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth || true; \
    curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth -o models/face-parse-bisent/resnet18-5c106cde.pth
# ЖЁСТКО проверяем критичные веса — если чего-то нет, сборка падает здесь, не на GPU.
RUN set -eux; \
    ls -la models/dwpose models/musetalkV15 models/musetalk models/sd-vae models/whisper; \
    test -f models/dwpose/dw-ll_ucoco_384.pth; \
    test -f models/musetalk/musetalk.json; \
    test -f models/musetalk/pytorch_model.bin; \
    test -f models/musetalkV15/musetalk.json; \
    test -f models/musetalkV15/unet.pth; \
    test -f models/sd-vae/config.json; \
    test -f models/sd-vae/diffusion_pytorch_model.bin; \
    test -f models/whisper/config.json; \
    test -f models/whisper/pytorch_model.bin

# 6) слой handler'а
RUN pip install --no-cache-dir runpod boto3 httpx

# 7) КРИТИЧНО: download_weights.sh делает `pip install -U huggingface_hub[cli]` и тащит
#    hf-hub 1.x, который ломает transformers (ImportError WhisperModel). Пинуем обратно <1.0.
RUN pip install --no-cache-dir "huggingface_hub==0.25.2"

# болванки-аватары
RUN mkdir -p /avatars
COPY avatars/ /avatars/
COPY scripts/runpod_handler.py /opt/runpod_handler.py

ENV AVATARS_DIR=/avatars \
    MUSETALK_DIR=/opt/MuseTalk \
    RUNPOD_DEFAULT_MODEL=musetalk

CMD ["python", "-u", "/opt/runpod_handler.py"]
