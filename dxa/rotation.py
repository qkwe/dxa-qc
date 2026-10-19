"""Ротация бедра без обучения: малый вертел и проекционные размеры проксимального отдела.

ТЗ: «отсутствие ротации (оценка малого вертела)»: в норме контур слегка деформирован
малым вертелом; при переротации контур плавный; при недоротации вертел слишком большой.
Дополнительно ротация меняет проекцию шейки: при наружной ротации шейка укорачивается,
головка приближается к оси диафиза (уменьшается офсет).

Снимок приводится к единой ориентации: медиальная сторона справа.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import roi

Spacing = tuple[float, float]


def _femur_runs(mask: np.ndarray, axis_x) -> tuple[np.ndarray, np.ndarray]:
    """Для каждой строки — края отрезка кости, через который проходит ось диафиза.

    Так в контур не попадают седалищная кость и таз, лежащие рядом в той же строке.
    """
    rows = mask.shape[0]
    left = np.full(rows, np.nan)
    right = np.full(rows, np.nan)
    for r in range(rows):
        xs = np.nonzero(mask[r])[0]
        if not len(xs):
            continue
        groups = np.split(xs, np.nonzero(np.diff(xs) > 1)[0] + 1)
        x0 = axis_x(r)
        best = min(groups, key=lambda g: 0 if g[0] <= x0 <= g[-1] else min(abs(g[0] - x0), abs(g[-1] - x0)))
        if best[0] - 3 <= x0 <= best[-1] + 3:
            left[r], right[r] = best[0], best[-1]
    return left, right


def measure(array: np.ndarray, spacing: Spacing, side: str | None, mask: np.ndarray | None = None) -> dict:
    """mask — готовая маска бедренной кости (например, SAM); иначе эвристическая из roi."""
    empty = dict(rot_lt_bump_mm=np.nan, rot_lt_area_mm2=np.nan, rot_offset_mm=np.nan,
                 rot_neck_width_mm=np.nan, rot_head_mm=np.nan, rot_gt_above_head_mm=np.nan,
                 rot_neck_len_mm=np.nan, rot_neck_angle_deg=np.nan)
    lm = roi.landmarks(array, spacing, side)
    if lm is None:
        return empty
    if mask is not None:
        labels, count = ndimage.label(mask)
        if count > 1:  # только самая крупная компонента
            sizes = np.bincount(labels.ravel()); sizes[0] = 0
            mask = labels == int(np.argmax(sizes))
    else:
        mask = lm["mask"]
    hx, hy, radius = lm["head"]
    if lm["medial"] == "left":  # к единой ориентации: медиально справа
        mask = mask[:, ::-1]
        hx = mask.shape[1] - 1 - hx
    sx, sy = spacing
    ys = np.nonzero(mask.any(axis=1))[0]
    y1 = int(ys.max())

    # ось диафиза по центрам нижних 4 см кости
    shaft_rows = np.arange(max(y1 - int(40 / sy), int(hy + radius)), y1 + 1)
    centers = np.array([np.nonzero(mask[r])[0].mean() for r in shaft_rows])
    k, b = np.polyfit(shaft_rows, centers, 1)
    axis_x = lambda r: k * r + b
    left, right = _femur_runs(mask, axis_x)

    # медиальный контур от низа головки до диафиза
    start, stop = int(hy + radius * 0.6), int(shaft_rows[0] + (y1 - shaft_rows[0]) * 0.3)
    band = np.arange(start, stop)
    band = band[~np.isnan(right[band])]
    if len(band) < 15:
        return empty
    # контур ведём от диафиза вверх и обрываем на первом скачке: выше него медиальный край
    # «перепрыгивает» на седалищную кость, которая на проекции касается шейки
    raw = right[band]
    jumps = np.nonzero(np.abs(np.diff(raw)) > max(4.0 / sx, 3))[0]
    if len(jumps):
        cut = int(jumps[-1]) + 1  # последний скачок — ближайший к диафизу
        band, raw = band[cut:], raw[cut:]
    if len(band) < 15:
        return empty
    edge = ndimage.median_filter(raw, 3)
    # выступ малого вертела: контур минус его сглаженная версия (крупный масштаб ~10 мм);
    # края участка не берём — там сглаживание врёт
    trend = ndimage.gaussian_filter1d(edge, sigma=10 / sy, mode="nearest")
    bump = (edge - trend) * sx
    margin = max(int(3 / sy), 2)
    # Верхнюю часть участка (переход шейки к седалищной кости) НЕ исключаем, хотя наибольший выступ
    # иногда приходится туда, а не на сам малый вертел: проверено 19.10 на вложенной CV — исключение
    # верхних 15/25/35/50% участка ухудшает оценку ротации (AUC 0.863 → 0.851/0.840/0.815/0.736).
    # Полезный сигнал даёт форма всего медиального контура от шейки до диафиза, а не только вертел.
    bump[:margin] = 0
    bump[-margin:] = 0
    # и относительно хорды между началом и концом участка (как в прошлой версии)
    chord = np.interp(band, [band[0], band[-1]], [edge[0], edge[-1]])
    over_chord = (edge - chord) * sx

    # шейка: самое узкое место между головкой и вертелом по перпендикуляру ~ по строкам
    widths = (right - left)[int(hy):int(hy + radius * 2.5)]
    neck_width = float(np.nanmin(widths) * sx) if np.isfinite(widths).any() else np.nan

    # большой вертел: верхняя точка кости латеральнее оси диафиза
    lateral_part = mask[:, : int(max(axis_x(hy), 1))]
    gt_rows = np.nonzero(lateral_part.any(axis=1))[0]
    gt_top = float(gt_rows.min()) if len(gt_rows) else np.nan

    offset = (hx - axis_x(hy)) * sx
    # наибольший выступ и сам анализируемый участок контура — в координатах исходного снимка, для разметки
    to_x = (lambda x: float(x)) if lm["medial"] == "right" else (lambda x: float(mask.shape[1] - 1 - x))
    peak = int(np.argmax(bump))
    peak_x = to_x(edge[peak])
    contour = [(to_x(x), float(r)) for x, r in zip(edge[::2], band[::2])]
    # угол шейки: от центра головки к точке оси диафиза на уровне малого вертела
    neck_dx = (hx - axis_x(hy + radius * 2)) * sx
    neck_dy = radius * 2 * sy
    return dict(
        rot_lt_bump_mm=float(bump.max()),
        rot_lt_peak_x=peak_x,
        rot_lt_peak_y=float(band[peak]),
        rot_lt_contour=contour,
        rot_lt_area_mm2=float(np.clip(bump, 0, None).sum() * sy * sx),
        rot_lt_chord_mm=float(over_chord.max()),
        rot_offset_mm=float(offset),
        rot_neck_width_mm=neck_width,
        rot_head_mm=float(2 * radius * sx),
        rot_gt_above_head_mm=float((hy - gt_top) * sy),
        rot_neck_len_mm=float(np.hypot(neck_dx, neck_dy)),
        rot_neck_angle_deg=float(np.degrees(np.arctan2(neck_dx, neck_dy))),
        rot_shaft_deg=float(np.degrees(np.arctan(k * sx / sy))),
    )
