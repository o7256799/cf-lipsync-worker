# RunPod lip-sync — runbook деплоя

Что это: serverless-воркер, который делает lip-sync (озвучка → губы аватара) на GPU.
Заменяет платный Pixverse (~$0.55/ролик) на свой RunPod (~$0.03–0.05/ролик).

Файлы:
- `scripts/runpod_handler.py` — entry point воркера (контракт с `app/services/lipsync/runpod_provider.py`).
- `runpod/Dockerfile.runpod` — образ GPU-воркера (MuseTalk).
- `runpod/avatars/` — болванки-аватары, `dima.mp4` = `input_videos/DIMA.mp4`.

## Контракт (не менять без синхронизации с runpod_provider.py)
Вход:  `input = {source_id, audio_url, model}` — audio_url это presigned mp3, видео не гоняем.
Выход: `output = {video_url, s3_key, model, source_id, bytes}` — presigned mp4 в S3 фабрики.

## Шаги деплоя

### 1. Подготовить болванку в build-контекст
```
cp /opt/content-factory/input_videos/DIMA.mp4 runpod/avatars/dima.mp4
```
`source_id="dima"` → `/avatars/dima.mp4` внутри воркера.

### 2. Собрать и запушить образ
Собирать НЕ на vietnam (там нет GPU/места) — на машине с Docker + доступом в registry
(Docker Hub / GHCR). Из корня репо:
```
docker build -f runpod/Dockerfile.runpod -t <registry>/cf-lipsync:latest .
docker push <registry>/cf-lipsync:latest
```

### 3. Создать RunPod Serverless Endpoint
RunPod → Serverless → New Endpoint:
- Container image: `<registry>/cf-lipsync:latest`
- GPU: 1× (24 ГБ, напр. L4/A5000 — под MuseTalk хватает)
- Env-переменные эндпоинта (секреты берём из vietnam `/opt/content-factory/.env`):
  - `S3_ENDPOINT`, `S3_REGION`, `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`
  - (опц.) `MUSETALK_CMD`, `MUSETALK_DIR` — если путь inference другой
- Скопировать **Endpoint ID**.

### 4. Прописать в фабрике (.env на vietnam) — АДДИТИВНО, чужое не трогать
```
RUNPOD_API_KEY=<ключ из RunPod Settings>
RUNPOD_ENDPOINT_ID=<endpoint id>
```
Переключить нишу на RunPod (в `config/niches/lawyers_ru.yaml`) ТОЛЬКО после успешного теста;
Pixverse оставить `fallback_provider`.

### 5. Тест одного ролика
```
python -m app.cli render-video <script_id> --sync   # с provider=runpod в niche
```
или дёрнуть endpoint напрямую curl'ом с presigned audio_url.

## Проверить на GPU (то, что нельзя проверить без RunPod)
1. Версии torch/cuda в Dockerfile совместимы с MuseTalk (может понадобиться cu118 + конкретный torch).
2. Имя скрипта весов (`download_weights.sh` vs другой) и realtime-inference
   (`scripts.realtime_inference` — уточнить по README MuseTalk, поправить `MUSETALK_CMD`).
3. Формат вывода MuseTalk (папка кадров vs mp4) — при необходимости досборка в handler.
4. Cold-start время (образ большой) — держать endpoint тёплым на батч.

## Модели
- `musetalk` — primary (быстрый, audio-driven).
- `liveportrait` — для A/B (пункт 10 роадмапа): добавить установку в Dockerfile + `LIVEPORTRAIT_CMD`.
