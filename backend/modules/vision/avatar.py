"""Cấu hình avatar (module Vision) — mô tả avatar cho frontend nạp/render.

Backend không render; nó chỉ mô tả avatar (kiểu, model, cách điều khiển miệng) để
frontend (Live2D/3D) dùng chung một nguồn cấu hình. Chuẩn bị sẵn cho 3D: khi
`type == "3d"` frontend sẽ nạp model 3D và điều khiển blendshape miệng theo viseme
do modules/vision/lipsync.py sinh ra.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.config import settings


@dataclass
class AvatarConfig:
    type: str          # "svg" | "live2d" | "3d"
    model: str         # đường dẫn/định danh model (frontend phân giải)
    mouth_param: str   # tham số điều khiển miệng (Live2D: ParamMouthOpenY; 3D: blendshape)
    viseme_fps: int    # tần số khung nhép miệng backend sinh ra

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "model": self.model,
            "mouth_param": self.mouth_param,
            "viseme_fps": self.viseme_fps,
            "viseme_enabled": settings.viseme_enabled,
        }


def get_avatar_config() -> AvatarConfig:
    if settings.avatar_type == "vrm":
        mouth = "aa"  # VRM: expression preset khẩu hình "aa" (độ mở miệng)
    elif settings.avatar_type in {"live2d", "svg"}:
        mouth = "ParamMouthOpenY"
    else:
        mouth = "jawOpen"
    return AvatarConfig(
        type=settings.avatar_type,
        model=settings.avatar_model,
        mouth_param=mouth,
        viseme_fps=settings.viseme_fps,
    )
