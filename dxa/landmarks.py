"""Отметки малого вертела: где он на самом деле — для проверки, правильно ли сервис его находит.

Это анатомия, а не диагноз: метки нарушений (организатора и врача) не меняются, обучение их не использует.
Отметка — точка в долях кадра (x, y ∈ [0; 1], кадр = картинка разметки) или «не виден» (контур гладкий).
Вместе с ней сохраняется точка наибольшего выступа, найденная сервисом, и размер кадра в мм —
расстояние между ними и есть ошибка нахождения малого вертела (validate_lt.py, страница /review).

    <DXA_FEEDBACK>/landmarks.json    {"<study_uid>:<image_uid>": {...}}
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from .feedback import FEEDBACK

STATES = ("point", "not_visible")
_lock = threading.Lock()


def _file(root: Path) -> Path:
    return Path(root) / "landmarks.json"


def load(root: Path = FEEDBACK) -> dict:
    try:
        return json.loads(_file(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save(study_uid: str, image_uid: str, state: str, point=None, service=None, frame_mm=None, path: str = "",
         root: Path = FEEDBACK) -> dict:
    """point, service — [x, y] в долях кадра; frame_mm — [ширина, высота] кадра в мм."""
    if state not in STATES:
        raise ValueError(f"состояние — одно из {STATES}")
    if not study_uid or not image_uid:
        raise ValueError("у снимка нет идентификаторов исследования и изображения")

    def frac(p):
        if p is None:
            return None
        x, y = (float(v) for v in p)
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("точка вне кадра")
        return [round(x, 5), round(y, 5)]

    entry = dict(state=state, point=frac(point) if state == "point" else None, service=frac(service),
                 frame_mm=[float(v) for v in frame_mm] if frame_mm else None, path=str(path),
                 created=time.strftime("%Y-%m-%d %H:%M:%S"))
    if state == "point" and entry["point"] is None:
        raise ValueError("нужна точка")
    entry["error_mm"] = error_mm(entry)
    with _lock:
        data = load(root)
        data[f"{study_uid}:{image_uid}"] = entry
        Path(root).mkdir(parents=True, exist_ok=True)
        _file(root).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return entry


def delete(study_uid: str, image_uid: str, root: Path = FEEDBACK) -> bool:
    with _lock:
        data = load(root)
        if data.pop(f"{study_uid}:{image_uid}", None) is None:
            return False
        _file(root).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def error_mm(e: dict) -> float | None:
    """Расстояние между отметкой и точкой сервиса, мм; None — сравнивать нечего."""
    if e.get("state") != "point" or not e.get("point") or not e.get("service") or not e.get("frame_mm"):
        return None
    (x, y), (sx, sy), (w, h) = e["point"], e["service"], e["frame_mm"]
    return round(float(np.hypot((x - sx) * w, (y - sy) * h)), 1)


def summary(root: Path = FEEDBACK) -> dict:
    """Сводка: сколько отмечено, медианная ошибка, доля в пределах 5 и 10 мм."""
    data = load(root)
    err = np.array([e["error_mm"] for e in data.values() if e.get("error_mm") is not None], float)
    out = dict(total=len(data), points=sum(e["state"] == "point" for e in data.values()),
               not_visible=sum(e["state"] == "not_visible" for e in data.values()), compared=int(len(err)))
    if len(err):
        out.update(median_mm=round(float(np.median(err)), 1), within_5mm=round(float((err <= 5).mean()), 3),
                   within_10mm=round(float((err <= 10).mean()), 3))
    return out
