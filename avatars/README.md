# Болванки-аватары (source_id → файл)

- `dima.mp4` = `input_videos/DIMA.mp4` (md5 24121349c6c3f3d9785712b3f6272b32), source_id="dima".

Файлы .mp4 сюда кладутся при сборке образа (см. runpod/README_RUNPOD.md, шаг 1),
в git НЕ коммитятся (тяжёлые). Handler резолвит source_id → /avatars/<source_id>.mp4.
