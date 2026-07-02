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


# ----------------------------------------------------------------------------- gpu diag
def _nvidia_smi() -> dict:
    out = {}
    for label, args in (
        ("list", ["nvidia-smi", "-L"]),
        ("query", ["nvidia-smi",
                   "--query-gpu=name,compute_cap,driver_version,memory.total",
                   "--format=csv,noheader"]),
    ):
        try:
            pr = subprocess.run(args, capture_output=True, text=True, timeout=30)
            out[label] = (pr.stdout or pr.stderr or "").strip()
        except Exception as e:  # noqa: BLE001
            out[label] = f"<{type(e).__name__}: {e}>"
    return out


def _gpu_diag() -> dict:
    """Собрать всё, что нужно, чтобы понять причину no-kernel-image:
    какую compute capability отдаёт хост vs под какие арх собран torch."""
    d: dict = {"nvidia_smi": _nvidia_smi()}
    try:
        import torch  # тяжёлый импорт — только в diag/ошибке, не на каждом инференсе
        d["torch_version"] = torch.__version__
        d["torch_cuda"] = torch.version.cuda
        d["cuda_available"] = torch.cuda.is_available()
        try:
            d["torch_arch_list"] = torch.cuda.get_arch_list()  # sm_XX, под которые собран torch
        except Exception as e:  # noqa: BLE001
            d["torch_arch_list"] = f"<{type(e).__name__}: {e}>"
        if torch.cuda.is_available():
            try:
                d["device_name"] = torch.cuda.get_device_name(0)
                cap = torch.cuda.get_device_capability(0)
                d["device_capability"] = f"sm_{cap[0]}{cap[1]}"
                d["device_count"] = torch.cuda.device_count()
                # ключевой вывод: знает ли собранный torch арх этой карты
                arch = d.get("torch_arch_list") or []
                d["arch_supported"] = (
                    d["device_capability"] in arch if isinstance(arch, list) else "unknown"
                )
                # реальный тест: аллокация + матмул на GPU провоцирует no-kernel прямо здесь
                try:
                    x = torch.randn(64, 64, device="cuda")
                    _ = (x @ x).sum().item()
                    d["matmul_ok"] = True
                except Exception as e:  # noqa: BLE001
                    d["matmul_ok"] = False
                    d["matmul_error"] = f"{type(e).__name__}: {e}"
            except Exception as e:  # noqa: BLE001
                d["device_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        d["torch_error"] = f"{type(e).__name__}: {e}"
    return d


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


def _to_wav(audio: Path, out_wav: Path) -> Path:
    # MuseTalk/whisper надёжнее работают с wav 16k mono.
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio), "-ar", "16000", "-ac", "1", str(out_wav)],
        capture_output=True, text=True, timeout=120,
    )
    return out_wav if (out_wav.exists() and out_wav.stat().st_size > 0) else audio


def _run_musetalk(avatar: Path, audio: Path, out: Path) -> Path:
    # MuseTalk realtime_inference — config-driven (YAML: avatar_id -> video_path + audio_clips).
    md = Path(MODEL_CWD["musetalk"])
    work = out.parent
    wav = _to_wav(audio, work / "audio.wav")
    result_dir = work / "mt_results"
    avatar_id = avatar.stem  # уникальный per source_id — иначе кеш аватара путается
    cfg = work / "mt_config.yaml"
    cfg.write_text(
        f"{avatar_id}:\n"
        "  preparation: True\n"
        "  bbox_shift: 0\n"
        f'  video_path: "{avatar}"\n'
        "  audio_clips:\n"
        f'    clip0: "{wav}"\n'
    )
    cmd = [
        "python", "-m", "scripts.realtime_inference",
        "--version", "v15",
        "--inference_config", str(cfg),
        "--result_dir", str(result_dir),
        "--fps", "25",
    ]
    t0 = time.time()
    # stdin "n": если аватар уже подготовлен на этом (тёплом) воркере — НЕ пересоздавать,
    # грузить кеш (быстро). На свежем воркере запроса нет — готовит сам. timeout 30 мин
    # под первую подготовку полного DIMA.
    proc = subprocess.run(cmd, cwd=str(md), input="n\n", capture_output=True, text=True, timeout=60 * 30)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1800:]
        raise RuntimeError(f"musetalk inference failed (rc={proc.returncode}):\n{tail}")
    # MuseTalk сохраняет в ./results/<ver>/avatars/<id>/vid_output/*.mp4 (относительно cwd),
    # часто игнорируя --result_dir. Ищем в обоих местах, предпочитая vid_output.
    cands = [
        p for p in list((md / "results").rglob("*.mp4")) + list(result_dir.rglob("*.mp4"))
        if p.exists() and p.stat().st_size > 0 and p.name != "temp.mp4"
    ]
    vid = [p for p in cands if "vid_output" in str(p)]
    chosen = sorted(vid or cands, key=lambda p: p.stat().st_mtime)
    if not chosen:
        raise RuntimeError(
            f"musetalk produced no mp4 (searched {md}/results и {result_dir}). "
            f"stdout tail:\n{(proc.stdout or '')[-800:]}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(chosen[-1], out)
    print(f"[runpod] musetalk ok in {time.time() - t0:.1f}s -> {out} (from {chosen[-1]})")
    return out


def _cap_video_side(src: Path, dst: Path) -> Path:
    # Ограничить длинную сторону аватара (по умолч. 1280) — чтобы длинный/hi-res аватар (полный DIMA
    # 1080x1920, 1600 кадров) не выжирал память и не ронял воркер OOM. Тюнится input.maxside.
    maxside = int(os.environ.get("LATENTSYNC_MAX_SIDE", "1280"))
    try:
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(src)],
            capture_output=True, text=True, timeout=60,
        )
        w, h = (int(x) for x in pr.stdout.strip().split("x")[:2])
    except Exception:
        return src
    if max(w, h) <= maxside:
        return src
    vf = f"scale={maxside}:-2" if w >= h else f"scale=-2:{maxside}"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "18", "-an", str(dst)],
        capture_output=True, text=True, timeout=300,
    )
    return dst if (dst.exists() and dst.stat().st_size > 0) else src


def _run_latentsync(avatar: Path, audio: Path, out: Path) -> Path:
    # LatentSync: video+audio -> out.mp4 напрямую, явный --video_out_path, 512px (резкий рот).
    ld = Path(os.environ.get("LATENTSYNC_DIR", "/opt/LatentSync"))
    work = out.parent
    avatar = _cap_video_side(avatar, work / "avatar_capped.mp4")  # анти-OOM для длинного/hi-res аватара
    wav = _to_wav(audio, work / "audio.wav")
    cfg = os.environ.get("LATENTSYNC_UNET_CFG", "configs/unet/stage2_512.yaml")
    steps = os.environ.get("LATENTSYNC_STEPS", "30")      # 30 по умолчанию (было 20) — чётче/стабильнее
    guidance = os.environ.get("LATENTSYNC_GUIDANCE", "1.5")
    cmd = [
        "python", "-m", "scripts.inference",
        "--unet_config_path", cfg,
        "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
        "--inference_steps", str(steps),
        "--guidance_scale", str(guidance),
        "--video_path", str(avatar),
        "--audio_path", str(wav),
        "--video_out_path", str(out),
    ]
    if os.environ.get("LATENTSYNC_DEEPCACHE", "1") != "0":  # deepcache=0 → медленнее, но чуть качественнее
        cmd.append("--enable_deepcache")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ld), capture_output=True, text=True, timeout=60 * 20)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1800:]
        # при no-kernel-image приложить железо/арх хоста — иначе не понять, на какой карте упало
        smi = _nvidia_smi().get("query", "?")
        raise RuntimeError(f"latentsync failed (rc={proc.returncode}) [gpu: {smi}]:\n{tail}")
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"latentsync produced no output. stdout tail:\n{(proc.stdout or '')[-800:]}")
    print(f"[runpod] latentsync ok in {time.time() - t0:.1f}s -> {out}")
    return out


def _run_model(model: str, avatar: Path, audio: Path, out: Path) -> Path:
    if model == "latentsync":
        return _run_latentsync(avatar, audio, out)
    if model == "musetalk":
        return _run_musetalk(avatar, audio, out)
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

    # diag-режим: не гоняем инференс, отдаём железо/torch хоста, чтобы понять no-kernel-image.
    if model == "diag" or inp.get("diag"):
        diag = _gpu_diag()
        print(f"[runpod] diag job={job_id}: {diag}")
        return {"diag": diag, "job_id": job_id}

    # тюнинг без пересборки: параметры из input -> env (читаются раннерами)
    for k, env in (("steps", "LATENTSYNC_STEPS"), ("guidance", "LATENTSYNC_GUIDANCE"),
                   ("deepcache", "LATENTSYNC_DEEPCACHE"), ("maxside", "LATENTSYNC_MAX_SIDE")):
        if inp.get(k) is not None:
            os.environ[env] = str(inp[k])

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
