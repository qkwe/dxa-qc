"""Разметка врача: подтверждение или исправление вердиктов сервиса и обведённые посторонние предметы.

ТЗ п. 2.6: «...с возможностью подтверждения специалистом». Врач в интерфейсе /review подтверждает или
исправляет нарушения по каждому критерию снимка и может обвести пропущенный предмет. Запись хранит
исходный DICOM и метки — это пополнение обучающего набора:

    out/feedback/<id>/image.dcm + entry.json        (каталог задаётся DXA_FEEDBACK)

Дообучение (train.py / evaluate_final.py --feedback ...) берёт набор организатора и все записи; если
врач разметил снимок, который есть в наборе организатора, его метки заменяют исходные.
"""

from __future__ import annotations

import io
import itertools
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pydicom

from .data import LABEL_AXIS, LABEL_FOREIGN, LABEL_POSITIONING, LABEL_ROI, Image, pixel_spacing

FEEDBACK = Path(os.environ.get("DXA_FEEDBACK", "out/feedback"))
CRITERIA = {"spine": [LABEL_POSITIONING, LABEL_AXIS, LABEL_FOREIGN], "femur": [LABEL_POSITIONING, LABEL_ROI]}
# тип нарушения ротации бедра (ТЗ, рис. 5): переротация — контур гладкий, недоротация — малый вертел крупный
ROTATION_TYPES = ("over", "under", "unknown")
# порядок сохранения: часы Windows тикают раз в ~15 мс — две записи подряд получают одно время
_seq = itertools.count()


def save(dicom: bytes, region: str, side: str | None, labels: dict, boxes: list, note: str = "",
         source: dict | None = None, root: Path = FEEDBACK, rotation_type: str | None = None) -> str:
    """Сохранить разметку врача. labels — {метка: 0/1} для всех критериев области; boxes — рамки
    предметов [x0, y0, x1, y1] в пикселях исходного DICOM; rotation_type — тип нарушения укладки бедра
    (over / under / unknown), только при нарушении укладки бедра."""
    if region not in CRITERIA:
        raise ValueError(f"неизвестная область: {region!r}")
    missing = [c for c in CRITERIA[region] if c not in labels]
    extra = [c for c in labels if c not in CRITERIA[region]]
    if missing or extra:
        raise ValueError(f"нужны метки ровно по критериям области {CRITERIA[region]}; нет {missing}, лишние {extra}")
    if any(int(v) not in (0, 1) for v in labels.values()):
        raise ValueError("метки — 0 (норма) или 1 (нарушение)")
    ds = pydicom.dcmread(io.BytesIO(dicom), force=True)          # проверка, что это читаемый DICOM-снимок
    rows, cols = np.asarray(ds.pixel_array).shape[:2]
    clean = []
    for b in boxes or []:
        x0, y0, x1, y1 = (float(v) for v in b)
        x0, x1 = sorted((max(0.0, min(x0, cols)), max(0.0, min(x1, cols))))
        y0, y1 = sorted((max(0.0, min(y0, rows)), max(0.0, min(y1, rows))))
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            clean.append([round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)])
    if clean and region != "spine":
        raise ValueError("обводить посторонние предметы можно на снимке позвоночника")
    if rotation_type is not None:
        if rotation_type not in ROTATION_TYPES:
            raise ValueError(f"тип ротации — один из {ROTATION_TYPES}, получено {rotation_type!r}")
        if region != "femur" or not int(labels.get(LABEL_POSITIONING, 0)):
            raise ValueError("тип ротации указывается только при нарушении укладки бедра")
    if clean and not int(labels.get(LABEL_FOREIGN, 0)):
        labels = {**labels, LABEL_FOREIGN: 1}   # обведён предмет — значит, нарушение «посторонние предметы»
    entry_id = uuid.uuid4().hex[:12]
    d = root / entry_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "image.dcm").write_bytes(dicom)
    entry = dict(id=entry_id, created=time.strftime("%Y-%m-%d %H:%M:%S"), ts=time.time(), seq=next(_seq),
                 study_uid=str(ds.get("StudyInstanceUID", "") or entry_id),
                 image_uid=str(ds.get("SOPInstanceUID", "") or entry_id),
                 region=region, side=side, labels={k: int(v) for k, v in labels.items()},
                 boxes=clean, note=str(note)[:2000], source=source or {})
    if rotation_type is not None:
        entry["rotation_type"] = rotation_type
    (d / "entry.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
    return entry_id


def entries(root: Path = FEEDBACK) -> list[dict]:
    out = []
    for f in sorted(Path(root).glob("*/entry.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda e: (e.get("created", ""), e.get("ts", 0.0), e.get("seq", 0)))  # порядок сохранения


def delete(entry_id: str, root: Path = FEEDBACK) -> bool:
    d = Path(root) / Path(entry_id).name
    if not (d / "entry.json").exists():
        return False
    shutil.rmtree(d)
    return True


def load_images(root: str | Path | None) -> list[Image]:
    """Записи врача как обучающие снимки (тот же формат, что у набора организатора)."""
    if not root or not Path(root).exists():
        return []
    from .pipeline import Item, Service

    images = []
    for e in entries(Path(root)):
        try:
            ds = pydicom.dcmread(Path(root) / e["id"] / "image.dcm", force=True)
            it = Item(path=str(Path(root) / e["id"] / "image.dcm"), ds=ds, array=np.asarray(ds.pixel_array))
            if it.array.ndim == 3:
                it.array = it.array[..., :3].mean(-1) if it.array.shape[-1] in (3, 4) else it.array[0]
            spacing = Service._to_native(it, pixel_spacing(ds, it.array))   # как в сервисе: к пикселю ~0.6 мм
        except Exception:
            continue
        labels = {k: int(v) for k, v in e["labels"].items()}
        images.append(Image(study_uid=e["study_uid"], image_uid=e["image_uid"], path=Path(it.path), array=it.array,
                            region=e["region"], side=e.get("side"), instance=0, copies=1, n_images=1,
                            spacing=spacing, quality=int(any(labels.values())), labels=labels))
    return images


def merge(base: list[Image], extra: list[Image]) -> list[Image]:
    """Набор организатора + разметка врача; снимок, размеченный врачом повторно, берётся с его метками.
    Сопоставление — по SOPInstanceUID: исследование в наборе организатора названо по каталогу, а сервис
    берёт StudyInstanceUID из тега — у одного снимка они разные. Заменённый снимок остаётся в группе своего
    исследования из набора организатора, иначе честная CV развела бы снимки одного пациента по фолдам."""
    extra = list({e.image_uid: e for e in extra}.values())   # снимок размечен повторно — действует последняя запись
    by_image = {i.image_uid: i for i in base}
    for e in extra:
        if e.image_uid in by_image:
            e.study_uid = by_image[e.image_uid].study_uid
    replaced = {e.image_uid for e in extra}
    return [i for i in base if i.image_uid not in replaced] + list(extra)
