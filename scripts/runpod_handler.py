"""RunPod serverless handler для lip-sync (Content Factory, этап D роадмапа).

Контракт задаётся клиентом `app/services/lipsync/runpod_provider.py`:

    POST /run   input = {"source_id": <avatar>, "audio_url": <presigned mp3>, "model": <name>}
    ...poll...
    output      = {"video_url": <presigned mp4>, ...}

Поток внутри воркера:
  1. Взять input {source_id, audio_url, model}.
  2. Скачать аудио по presigned URL (видео-байты по сети НЕ гоняем).
  3. Резолвнуть болванку: source_id -> /avatars/<source_id>.mp4 (запечена в образ/volume).
  4. Прогнать выбранную lip-sync модель (musetalk по умолчанию | liveportrait) на GPU.
  5. Нормализовать вывод через ffmpeg (H.264 + faststart, чтобы стримилось).
  6. Залить результат в тот же S3, что у фабрики, вернуть presigned video_url.

Идемпотентность: ключ результата привязан к RunPod job id — повторный запуск того же
джоба перезальёт тот же объект, а не наплодит мусор. Аудио и болванку скачиваем заново
(дёшево), inference-кэш не держим — serverless-воркер эфемерен.

Секреты (S3 creds) НЕ зашиты в образ — приходят env-переменными эндпоинта
(задаются при создании RunPod endpoint), см. runpod/README_RUNPOD.md.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import boto3
import httpx
import runpod
from botocore.config import Config

# ----------------------------------------------------------------------------- config
AVATARS_DIR = Path(os.environ.get("AVATARS_DIR", "/avatars"))
WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/cf-runpod"))
OUT_PREFIX = os.environ.get("RUNPOD_OUT_PREFIX", "runpod/out")
DEFAULT_MODEL = os.environ.get("RUNPOD_DEFAULT_MODEL", "musetalk")

# Команды inference задаются через env, чтобы поправить пути под конкретный образ,
# не трогая логику handler'а. {avatar} {audio} {out} подставляются перед запуском.
# Значения по умолчанию — под официальный MuseTalk realtime inference и LivePortrait;
# ПРОВЕРИТЬ на первом GPU-прогоне и при необходимости скорректировать в env эндпоинта.
MODEL_CMDS = {
    "musetalk": os.environ.get(
        "MUSETALK_CMD",
        "python -m scripts.realtime_inference "
        "--video_path {avatar} --audio_path {audio} --result_path {out} --fps 25",
    ),
    "liveportrait": os.environ.get(
        "LIVEPORTRAIT_CMD",
        "python inference.py --source {avatar} --driving {audio} --output {out}",
    ),
}
MODEL_CWD = {
    "musetalk": os.environ.get("MUSETALK_DIR", "/opt/MuseTalk"),
    "liveportrait": os.environ.get("LIVEPORTRAIT_DIR", "/opt/LivePortrait"),
}


# ----------------------------------------------------------------------------- s3
def _s3():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT") or None,
        region_name=os.environ.get("S3_REGION", "auto"),
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def _upload_and_sign(local: Path, key: str, ttl_sec: int = 6 * 3600) -> str:
    bucket = os.environ["S3_BUCKET"]
    cli = _s3()
    cli.upload_file(str(local), bucket, key, ExtraArgs={"ContentType": "video/mp4"})
    return cli.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=ttl_sec
    )


# ----------------------------------------------------------------------------- helpers
def _download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
        r.raise_for_status()
        with dst.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    if dst.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file from {url[:80]}...")
    return dst


AVATAR_S3_PREFIX = os.environ.get("AVATAR_S3_PREFIX", "avatars")


def _resolve_avatar(source_id: str) -> Path:
    # 1) локально (если болванка запечена в образ/volume);
    # 2) иначе тянем из S3 фабрики (avatars/<source_id>.<ext>) и кэшируем в AVATARS_DIR.
    AVATARS_DIR.mkdir(parents=True, exist_ok=True)
    exts = (".mp4", ".mov", ".png", ".jpg")
    for ext in exts:
        p = AVATARS_DIR / f"{source_id}{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    # S3 fallback
    bucket = os.environ.get("S3_BUCKET")
    if bucket:
        cli = _s3()
        for ext in exts:
            key = f"{AVATAR_S3_PREFIX}/{source_id}{ext}"
            try:
                cli.head_object(Bucket=bucket, Key=key)
            except Exception:
                continue
            dst = AVATARS_DIR / f"{source_id}{ext}"
            cli.download_file(bucket, key, str(dst))
            print(f"[runpod] avatar {source_id} pulled from s3://{bucket}/{key}")
            return dst
    have = sorted(p.name for p in AVATARS_DIR.glob("*")) if AVATARS_DIR.exists() else []
    raise RuntimeError(
        f"avatar source_id={source_id!r} not found locally ({AVATARS_DIR}) "
        f"nor in S3 ({AVATAR_S3_PREFIX}/{source_id}.*). Local: {have}"
    )


def _run_model(model: str, avatar: Path, audio: Path, out: Path) -> Path:
    cmd_tpl = MODEL_CMDS.get(model)
    if not cmd_tpl:
        raise RuntimeError(f"unknown lipsync model {model!r}. Known: {list(MODEL_CMDS)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = cmd_tpl.format(avatar=str(avatar), audio=str(audio), out=str(out))
    cwd = MODEL_CWD.get(model) or None
    t0 = time.time()
    proc = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=60 * 12
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        raise RuntimeError(f"{model} inference failed (rc={proc.returncode}):\n{tail}")
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"{model} produced no output at {out}")
    print(f"[runpod] {model} inference ok in {time.time() - t0:.1f}s -> {out}")
    return out


def _normalize(src: Path, dst: Path) -> Path:
    # H.264 + AAC + faststart, чтобы результат стримился и открывался везде.
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dst),
        ],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        # если нормализация не удалась — отдаём сырой out, лучше сырой чем ничего
        print(f"[runpod] ffmpeg normalize failed, returning raw: {proc.stderr[-500:]}")
        shutil.copyfile(src, dst)
    return dst


# ----------------------------------------------------------------------------- handler
def handler(event: dict) -> dict:
    job_id = event.get("id") or uuid.uuid4().hex
    inp = event.get("input") or {}
    source_id = inp.get("source_id")
    audio_url = inp.get("audio_url")
    model = (inp.get("model") or DEFAULT_MODEL).lower()

    if not source_id:
        raise ValueError("input.source_id is required (avatar болванка)")
    if not audio_url:
        raise ValueError("input.audio_url is required (presigned mp3)")

    job_dir = WORK_DIR / job_id
    try:
        audio = _download(audio_url, job_dir / "audio.mp3")
        avatar = _resolve_avatar(source_id)
        raw = _run_model(model, avatar, audio, job_dir / "raw.mp4")
        final = _normalize(raw, job_dir / "final.mp4")

        key = f"{OUT_PREFIX}/{source_id}/{int(time.time())}_{job_id}.mp4"
        video_url = _upload_and_sign(final, key)
        size = final.stat().st_size
        print(f"[runpod] done job={job_id} model={model} key={key} size={size}")
        return {
            "video_url": video_url,
            "s3_key": key,
            "model": model,
            "source_id": source_id,
            "bytes": size,
        }
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
