"""Дополнительные серии (ТЗ п. 2.6): визуализация нарушения и текстовое описание в DICOM SR.

Для каждого снимка:
  * PNG с разметкой (ось позвоночника, зона гребней, область интереса бедра, головка);
  * DICOM Secondary Capture (RGB) той же разметкой — в том же исследовании, новая серия;
  * DICOM Basic Text SR с перечнем критериев и объяснением решения.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
from PIL import Image as P, ImageDraw, ImageFont
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from . import axis, roi

SC_CLASS = "1.2.840.10008.5.1.4.1.1.7"        # Secondary Capture Image Storage
SR_CLASS = "1.2.840.10008.5.1.4.1.1.88.11"    # Basic Text SR Storage
RED, GREEN, YELLOW, CYAN = (230, 50, 40), (60, 200, 80), (255, 215, 0), (0, 190, 255)
ORANGE = (255, 140, 0)
LT_VISIBLE_MM = 1.0   # выступ малого вертела меньше этого — «контур гладкий», вершину не рисуем


def _font(size):
    for f in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label(d: ImageDraw.ImageDraw, xy, text: str, font, color):
    """Подпись на тёмной подложке — читается и на светлой кости; не выходит за край кадра."""
    x, y = xy
    width, height = d.im.size
    x = min(max(x, 2), width - d.textlength(text, font=font) - 4)
    y = min(max(y, 2), height - font.size - 6)
    d.rectangle([x - 2, y - 1, x + d.textlength(text, font=font) + 2, y + font.size + 3], fill=(0, 0, 0))
    d.text((x, y), text, font=font, fill=color)


def overlay(it, show_peak: bool = False, show_foreign: bool = False) -> P.Image:
    """Снимок с разметкой. В интерфейсе включаются галочками (по умолчанию — меньше надписей):
    show_peak — наибольший выступ медиального контура бедра; show_foreign — рамки вокруг найденных
    посторонних предметов на снимке позвоночника."""
    a = axis.normalize(it.array)
    scale = 3
    img = P.fromarray((a * 255).astype(np.uint8)).convert("RGB").resize((a.shape[1] * scale, a.shape[0] * scale))
    d = ImageDraw.Draw(img)
    crit = it.result.get("criteria", {})
    spacing = it.result.get("spacing", (0.6, 0.6))
    feats = it.result.get("features", {})
    small = _font(max(11, img.width // 70))
    bad = any(flag for flag, _, _ in crit.values())
    notes = []  # пояснения под легендой (жёлтым): почему чего-то нет на картинке
    axis_marks = None
    if it.region == "spine":
        ys, xs = axis.centerline(it.array, spacing)
        if len(ys):
            n = max(int(len(ys) * 0.15), 3)
            top = (np.median(xs[:n]) * scale, np.median(ys[:n]) * scale)
            bot = (np.median(xs[-n:]) * scale, np.median(ys[-n:]) * scale)
            ok = not crit.get("Не выравнена ось позвоночника", (0,))[0]
            lw = max(3, img.width // 220)
            # ось как на рис. 2 ТЗ: через центры верхних и нижних позвонков (медианы крайних 15% линии)
            d.line([bot, top], fill=GREEN if ok else RED, width=lw)
            d.line([bot, (bot[0], top[1])], fill=CYAN, width=max(2, lw // 2))
            axis_marks = (top, bot, GREEN if ok else RED, lw)  # рисуются в конце — поверх легенды
            notes.append("ось — через центры крайних позвонков (кружки), голубая — вертикаль")
            ang = feats.get("axis_chord15_deg")
            if ang is not None and np.isfinite(ang):
                # подпись у середины линии: вверху кадра её закрывает легенда
                mid = ((top[0] + bot[0]) / 2, (top[1] + bot[1]) / 2)
                _label(d, (max(top[0], bot[0]) + 10, mid[1]), f"наклон {ang:.1f}°", small, GREEN if ok else RED)
        h = a.shape[0] * scale
        ok = not crit.get("Некорректная укладка", (0,))[0]
        d.rectangle([2, h * 0.75, a.shape[1] * scale - 3, h - 3], outline=GREEN if ok else RED, width=3)
    else:
        lm = roi.landmarks(it.array, spacing, it.side)
        if lm:
            hx, hy, r = lm["head"]
            ok = not crit.get("Некорректная укладка", (0,))[0]
            color = GREEN if ok else RED
            # ось диафиза и офсет головки — на что система смотрит при оценке ротации
            sc = lm.get("shaft_center")
            if sc is not None:
                d.line([(sc * scale, 0), (sc * scale, a.shape[0] * scale)], fill=CYAN, width=2)
                d.line([(hx * scale, hy * scale), (sc * scale, hy * scale)], fill=CYAN, width=2)
                # подпись — над серединой отрезка «центр головки — ось бедра», со значением
                off = feats.get("rot_offset_mm", np.nan)
                text = "офсет головки" + (f" {abs(off):.0f} мм" if np.isfinite(off) else "")
                mid = (hx + sc) / 2 * scale - d.textlength(text, font=small) / 2
                _label(d, (mid, max(hy * scale - small.size - 6, 0)), text, small, CYAN)
            d.ellipse([(hx - r) * scale, (hy - r) * scale, (hx + r) * scale, (hy + r) * scale],
                      outline=color, width=3)
            # зона седалищной кости (ТЗ: должна быть представлена) — медиальнее и ниже головки
            yz0, yz1 = int(hy + 1.2 * r) * scale, int(hy + 3.0 * r) * scale
            if lm["medial"] == "left":
                zx0, zx1 = 0, int(max(hx - 0.3 * r, 0)) * scale
            else:
                zx0, zx1 = int(min(hx + 0.3 * r, a.shape[1] - 1)) * scale, a.shape[1] * scale
            d.rectangle([zx0, yz0, zx1, yz1], outline=(170, 170, 255), width=2)
            _label(d, (zx0 + 4, yz0 + 3), "седалищная кость", small, (170, 170, 255))
            # ротация (dxa/rotation.py): анализируемый медиальный контур от шейки до диафиза и малый вертел на нём
            # (выступ ниже изгиба шейки — проверено по ручным отметкам). Точку рисуем, только если вертел
            # заметно выступает: у гладкого контура «самая выступающая точка» случайна
            contour = feats.get("rot_lt_contour") or []
            bump = feats.get("rot_lt_true_bump_mm", np.nan)
            if len(contour) > 1:
                d.line([(x * scale, y * scale) for x, y in contour], fill=YELLOW, width=2)
            lx, ly = feats.get("rot_lt_true_x"), feats.get("rot_lt_true_y")
            visible = np.isfinite(bump) and bump >= LT_VISIBLE_MM
            if not show_peak:
                ax = None
            elif visible and lx is not None and ly is not None and np.isfinite(lx) and np.isfinite(ly):
                d.ellipse([(lx - 5) * scale, (ly - 5) * scale, (lx + 5) * scale, (ly + 5) * scale],
                          outline=YELLOW, width=3)
                ax, ay, text = lx, ly, f"малый вертел: выступ {bump:.1f} мм"
            elif len(contour) > 1:
                ax, ay = (lx, ly) if lx is not None and ly is not None and np.isfinite(lx) and np.isfinite(ly)                     else contour[len(contour) // 2]
                text = f"контур гладкий, малый вертел не выступает ({bump:.1f} мм)" if np.isfinite(bump) else "контур не измерен"
            else:
                ax = None
            if ax is not None:
                w = d.textlength(text, font=small)
                tx = ax * scale + 14 if lm["medial"] == "right" else ax * scale - 14 - w
                _label(d, (tx, ay * scale - small.size // 2), text, small, YELLOW)
            if lm.get("reliable", True):
                x0, y0, x1, y1 = [v * scale for v in lm["box"]]
                ok = not crit.get("Некорректная область интереса", (0,))[0]
                d.rectangle([x0, y0, x1, y1], outline=GREEN if ok else RED, width=3)
                yb = lm["lt_bottom"] * scale
                d.line([(x0, yb), (x1, yb)], fill=YELLOW, width=2)
            else:
                # Замечание экспертов 16.10: на коротком скане рамку области интереса не рисуем —
                # она ошибочно сегментирует посторонние структуры. Показываем только причину.
                notes.append(f"рамка области интереса не построена: {lm.get('reliable_reason', '')}")
    # подпись
    lines = [("ЕСТЬ НАРУШЕНИЕ" if bad else "КАЧЕСТВЕННОЕ", RED if bad else GREEN)]
    lines += [(f"{'[НАРУШЕНИЕ]' if flag else '[норма]'} {name}", RED if flag else GREEN) for name, (flag, _, _) in crit.items()]
    lines += [(n, YELLOW) for n in notes]
    # шрифт легенды — по ширине кадра: самая длинная строка должна влезть (узкий снимок бедра)
    size = max(14, img.width // 45)
    f = _font(size)
    while size > 9 and max(d.textlength(t, font=f) for t, _ in lines) > img.width - 20:
        size -= 1
        f = _font(size)
    y = 6
    for text, color in lines:
        d.rectangle([4, y - 2, 12 + d.textlength(text, font=f), y + f.size + 4], fill=(0, 0, 0))
        d.text((8, y), text, font=f, fill=color)
        y += f.size + 8
    # ось и центры крайних позвонков — поверх легенды: верх столба часто под ней
    if axis_marks:
        top, bot, color, lw = axis_marks
        d.line([bot, top], fill=color, width=lw)
        for px, py in (top, bot):
            d.ellipse([px - 2 * lw, py - 2 * lw, px + 2 * lw, py + 2 * lw], outline=color, width=lw)
    # рамки найденных предметов — поверх легенды: косточки белья часто вверху кадра, под ней
    if show_foreign and it.region == "spine":
        for k, box in enumerate(it.result.get("foreign_boxes", [])):
            x0, y0, x1, y1 = [v * scale for v in box["box"]]
            d.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], outline=ORANGE, width=3)
            _label(d, (x0 - 4, y1 + 8), f"предмет {k + 1}" + (f" ({box['where']})" if box["where"] != "вне столба" else ""),
                   small, ORANGE)
    d.text((8, img.height - f.size - 8), "ИИ-контроль качества DXA · не для диагностики", font=_font(max(11, img.width // 70)),
           fill=(200, 200, 200))
    return img


def _base(src, sop_class, modality, series_desc):
    ds = Dataset()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = sop_class, meta.MediaStorageSOPInstanceUID
    for tag in ("PatientID", "PatientName", "PatientBirthDate", "PatientSex", "StudyInstanceUID",
                "StudyID", "StudyDate", "StudyTime", "AccessionNumber", "ReferringPhysicianName"):
        setattr(ds, tag, src.get(tag, "") or "")
    ds.Modality = modality
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 9001 if modality == "OT" else 9002
    ds.InstanceNumber = 1
    ds.SeriesDescription = series_desc
    now = dt.datetime.now()
    ds.ContentDate, ds.ContentTime = now.strftime("%Y%m%d"), now.strftime("%H%M%S")
    ds.Manufacturer = "DXA-QC (хакатон)"
    ds.SpecificCharacterSet = "ISO_IR 192"
    return ds


def secondary_capture(it, img: P.Image) -> Dataset:
    ds = _base(it.ds, SC_CLASS, "OT", "ИИ: контроль качества DXA — разметка")
    arr = np.asarray(img.convert("RGB"))
    ds.ConversionType = "WSD"
    ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "RGB", 0
    ds.Rows, ds.Columns = arr.shape[:2]
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = arr.tobytes()
    return ds


def text_sr(it) -> Dataset:
    ds = _base(it.ds, SR_CLASS, "SR", "ИИ: контроль качества DXA — заключение")
    crit = it.result.get("criteria", {})
    bad = [n for n, (f, _, _) in crit.items() if f]

    def code(value, meaning, scheme="99DXAQC"):
        c = Dataset()
        c.CodeValue, c.CodingSchemeDesignator, c.CodeMeaning = value, scheme, meaning
        return c

    def text_item(meaning, text, rel="CONTAINS"):
        item = Dataset()
        item.RelationshipType, item.ValueType = rel, "TEXT"
        item.ConceptNameCodeSequence = Sequence([code("TXT", meaning)])
        item.TextValue = text[:1024]
        return item

    ds.ValueType = "CONTAINER"
    ds.ContinuityOfContent = "SEPARATE"
    ds.ConceptNameCodeSequence = Sequence([code("QC", "Контроль качества денситометрии")])
    ds.CompletionFlag, ds.VerificationFlag = "COMPLETE", "UNVERIFIED"
    items = [text_item("Изображение", f"SOPInstanceUID {it.image_uid}"),
             text_item("Область", f"{_REGION.get(it.region, it.region)}{' (' + _SIDE.get(it.side, it.side) + ')' if it.side else ''}"
                                  f"; определена: {it.result.get('region_source', '')}"),
             text_item("Итог", ("Есть нарушение: " + "; ".join(bad)) if bad else "Исследование качественное")]
    items += [text_item(name, ("НАРУШЕНИЕ. " if flag else "норма. ") + text) for name, (flag, _, text) in crit.items()]
    if it.result.get("settings_note"):
        items.append(text_item("Настройки", it.result["settings_note"]))
    items.append(text_item("Примечание", "Результат ИИ-сервиса, требует подтверждения специалистом."))
    ds.ContentSequence = Sequence(items)
    ref = Dataset()
    ref.ReferencedSOPClassUID = it.ds.get("SOPClassUID", "")
    ref.ReferencedSOPInstanceUID = it.image_uid
    ds.CurrentRequestedProcedureEvidenceSequence = Sequence([])
    return ds


_REGION = {"spine": "Поясничный отдел позвоночника", "femur": "Проксимальный отдел бедра"}
_SIDE = {"l": "левое", "r": "правое"}


def write_series(it, out_dir: Path) -> dict[str, Path]:
    """PNG + DICOM SC + DICOM SR. Варианты PNG для галочек интерфейса: бедро — с наибольшим выступом
    контура (overlay_peak), позвоночник с найденными предметами — с рамками вокруг них (overlay_fo)."""
    out_dir = Path(out_dir) / _safe(it.study_uid)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = overlay(it)
    stem = _safe(it.image_uid)[-40:]
    img.save(out_dir / f"{stem}.png")
    files = {"overlay": out_dir / f"{stem}.png"}
    if it.region == "femur":
        files["overlay_peak"] = out_dir / f"{stem}_peak.png"
        overlay(it, show_peak=True).save(files["overlay_peak"])
    if it.result.get("foreign_boxes"):
        files["overlay_fo"] = out_dir / f"{stem}_fo.png"
        overlay(it, show_foreign=True).save(files["overlay_fo"])
    secondary_capture(it, img).save_as(out_dir / f"{stem}_SC.dcm", enforce_file_format=True)
    text_sr(it).save_as(out_dir / f"{stem}_SR.dcm", enforce_file_format=True)
    return files


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))
