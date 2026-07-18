"""Sinh khung nhép miệng (viseme/độ mở miệng) từ audio — backend side.

Ở MVP frontend tự tính lip-sync từ biên độ audio (Web Audio API). Module này
tính SẴN một chuỗi độ-mở-miệng theo thời gian ở backend để:
  - Frontend (Live2D/3D) chỉ việc phát lại theo mốc thời gian, không cần DSP.
  - Chuẩn bị cho avatar 3D: mỗi frame là một "viseme" điều khiển blendshape miệng.

Hiện dùng envelope biên độ (đủ cho mouth-open 0..1). Nâng cấp lên viseme theo âm vị
(phoneme→viseme) khi làm 3D thật: thay `audio_to_visemes` bằng bộ căn âm vị.
"""
from __future__ import annotations

import io

from core.config import settings


def audio_to_visemes(wav_bytes: bytes, fps: int | None = None) -> list[dict]:
    """Trả về [{t, v}] với t = giây, v = độ mở miệng 0..1, lấy mẫu ở `fps`.

    Chỉ hỗ trợ WAV (đọc bằng soundfile). Audio khác (mp3) trả [] — frontend tự
    tính lip-sync như cũ.
    """
    if not settings.viseme_enabled or not wav_bytes:
        return []
    fps = fps or settings.viseme_fps
    try:
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception:
        return []  # không phải WAV hoặc thiếu lib → bỏ qua, frontend tự lo

    if getattr(data, "ndim", 1) == 2:
        data = data.mean(axis=1)
    n = len(data)
    if n == 0:
        return []

    hop = max(1, int(sr / fps))
    frames: list[dict] = []
    for i in range(0, n, hop):
        chunk = data[i : i + hop]
        if len(chunk) == 0:
            break
        rms = float(np.sqrt(np.mean(chunk**2)))
        v = min(1.0, max(0.0, rms * 3.0))  # chuẩn hóa lên 0..1, nhấn dải giọng nói
        frames.append({"t": round(i / sr, 3), "v": round(v, 3)})
    return frames
