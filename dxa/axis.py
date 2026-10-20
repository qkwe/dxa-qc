"""Ось позвоночника без обучения: средняя линия столба и её геометрия.

По ТЗ ось считается выровненной, если наклон к вертикали не больше 5°.
Средняя линия строится по центрам тел позвонков в каждой строке:
  1. профиль яркости строки сворачивается с «окном» ширины позвонка — максимум
     свёртки даёт центр столба, устойчиво к рёбрам, газу и гребням таза;
  2. центры связываются динамическим программированием, чтобы линия не прыгала;
  3. прямая подгоняется робастно (Huber), выбросы не тянут угол.
Кроме наклона считаются смещение от центра кадра и изгиб (отклонение от прямой):
врачи называют «осью» и то, и другое, поэтому проверяем по данным.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

Spacing = tuple[float, float]  # (мм по X, мм по Y)

AXIS_LIMIT_DEG = 5.0  # ТЗ
VERTEBRA_WIDTH_MM = 45.0  # ширина тела поясничного позвонка у взрослого, 40–50 мм
MAX_STEP_MM = 1.5  # смещение центра между соседними строками


def normalize(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float32)
    lo, hi = np.percentile(values, (1, 99.5))
    return np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1)


def valid_rows(values: np.ndarray) -> np.ndarray:
    """Строки с изображением: у части снимков снизу или сверху чёрная полоса дополнения."""
    return values.max(axis=1) > 0.05


def centerline(array: np.ndarray, spacing: Spacing) -> tuple[np.ndarray, np.ndarray]:
    """Координаты центра столба (x) по строкам (y)."""
    values = ndimage.gaussian_filter(normalize(array), 1.0)
    rows, cols = values.shape
    mm_x, mm_y = spacing
    width = max(int(round(VERTEBRA_WIDTH_MM / mm_x)), 5)
    lo, hi = int(cols * 0.15), int(cols * 0.85)

    # отклик «окна» шириной позвонка минус окружение: ищем яркую полосу нужной ширины
    kernel = np.concatenate([-np.ones(width // 2), np.ones(width), -np.ones(width // 2)])
    kernel /= np.abs(kernel).sum()
    response = ndimage.convolve1d(values, kernel, axis=1, mode="nearest")
    response = ndimage.uniform_filter1d(response, 5, axis=0)[:, lo:hi]

    ok = valid_rows(values)
    ys = np.nonzero(ok)[0]
    # отбрасываем по 5% сверху и снизу: там край кадра и крестец
    margin = max(int(len(ys) * 0.05), 1)
    ys = ys[margin:-margin]
    if len(ys) < 20:
        return np.array([]), np.array([])

    # динамическое программирование по строкам: максимум отклика при ограниченном шаге
    step = max(int(round(MAX_STEP_MM / mm_x * mm_y)), 1)
    score = response[ys[0]].copy()
    back = np.zeros((len(ys), score.size), dtype=np.int32)
    for k in range(1, len(ys)):
        shifted = np.full((2 * step + 1, score.size), -np.inf)
        for d in range(-step, step + 1):
            if d < 0:
                shifted[d + step, :d] = score[-d:]
            elif d > 0:
                shifted[d + step, d:] = score[:-d]
            else:
                shifted[step] = score
        best = np.argmax(shifted, axis=0)
        back[k] = np.arange(score.size) - (best - step)
        score = shifted[best, np.arange(score.size)] + response[ys[k]]
    xs = np.zeros(len(ys), dtype=np.int32)
    xs[-1] = int(np.argmax(score))
    for k in range(len(ys) - 1, 0, -1):
        xs[k - 1] = back[k, xs[k]]
    xs = xs + lo
    # грубый центр цепляется за остистые отростки внутри позвонка, поэтому уточняем:
    # центр = середина между краями столба на полувысоте яркости
    smooth = ndimage.gaussian_filter(normalize(array), 2.0)
    background = np.percentile(smooth[ok], 30)
    centers = np.empty(len(ys))
    for k, (row, x0) in enumerate(zip(ys, xs)):
        profile = smooth[row]
        a, b = max(x0 - width // 3, 0), min(x0 + width // 3 + 1, cols)
        level = background + (np.median(profile[a:b]) - background) * 0.5
        left, right = x0, x0
        limit = int(width * 0.9)
        while left > 0 and x0 - left < limit and profile[left - 1] > level:
            left -= 1
        while right < cols - 1 and right - x0 < limit and profile[right + 1] > level:
            right += 1
        centers[k] = (left + right) / 2
    centers = ndimage.median_filter(centers, 9, mode="nearest")
    return ys.astype(float), centers


def _huber_line(ys: np.ndarray, xs: np.ndarray, delta: float = 3.0) -> tuple[float, float]:
    slope, intercept = np.polyfit(ys, xs, 1)
    for _ in range(20):
        residual = xs - (slope * ys + intercept)
        weights = np.where(np.abs(residual) <= delta, 1.0, delta / np.maximum(np.abs(residual), 1e-6))
        slope, intercept = np.polyfit(ys, xs, 1, w=np.sqrt(weights))
    return float(slope), float(intercept)


def measure(array: np.ndarray, spacing: Spacing) -> dict:
    ys, xs = centerline(array, spacing)
    if len(ys) < 20:
        return dict(axis_deg=np.nan, axis_signed_deg=np.nan)
    mm_x, mm_y = spacing
    rows, cols = array.shape
    slope, intercept = _huber_line(ys, xs)
    signed = float(np.degrees(np.arctan2(slope * mm_x, mm_y)))
    residual = (xs - (slope * ys + intercept)) * mm_x

    # наклон верхней и нижней половин: изгиб и локальный наклон
    half = len(ys) // 2
    s_top, _ = _huber_line(ys[:half], xs[:half])
    s_bot, _ = _huber_line(ys[half:], xs[half:])
    deg_top = float(np.degrees(np.arctan2(s_top * mm_x, mm_y)))
    deg_bot = float(np.degrees(np.arctan2(s_bot * mm_x, mm_y)))

    # как на рис. 2 ТЗ: линия через две точки — центр нижнего и центр верхнего позвонка
    # (не регрессия по всем строкам; при изгибе столба результаты расходятся)
    chord = {}
    for frac in (0.1, 0.15, 0.2):
        n = max(int(len(ys) * frac), 3)
        dx = (np.median(xs[:n]) - np.median(xs[-n:])) * mm_x
        dy = (np.median(ys[:n]) - np.median(ys[-n:])) * mm_y
        chord[frac] = float(np.degrees(np.arctan2(abs(dx), abs(dy))))

    # стрела прогиба: насколько средняя линия столба отходит от хорды между крайними позвонками, мм.
    # Дугообразный изгиб (сколиоз) даёт большую стрелу при любом наклоне хорды; врачи сколиоз
    # не считают нарушением «ось не выровнена» — это анатомия пациента, а не укладка
    n15 = max(int(len(ys) * 0.15), 3)
    y0, x0, y1, x1 = np.median(ys[:n15]), np.median(xs[:n15]), np.median(ys[-n15:]), np.median(xs[-n15:])
    chord_x = x0 + (ys - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else np.full(len(ys), x0)
    sagitta = float(np.max(np.abs(xs - chord_x)) * mm_x)

    # смещение от центра кадра (по средней точке линии) и разброс по горизонтали
    center_mm = (float(np.median(xs)) - cols / 2) * mm_x
    return dict(
        axis_deg=abs(signed),
        axis_signed_deg=signed,
        axis_chord10_deg=chord[0.1],
        axis_chord15_deg=chord[0.15],
        axis_chord20_deg=chord[0.2],
        axis_sagitta_mm=sagitta,
        axis_top_deg=abs(deg_top),
        axis_bottom_deg=abs(deg_bot),
        axis_max_half_deg=max(abs(deg_top), abs(deg_bot)),
        axis_bend_deg=abs(deg_top - deg_bot),
        axis_curve_mm=float(np.max(np.abs(residual))),
        axis_curve_rms_mm=float(np.sqrt(np.mean(residual**2))),
        axis_offset_mm=abs(center_mm),
        axis_span_mm=float((xs.max() - xs.min()) * mm_x),
        axis_top_offset_mm=abs((xs[: len(xs) // 5].mean() - cols / 2) * mm_x),
        axis_bottom_offset_mm=abs((xs[-len(xs) // 5 :].mean() - cols / 2) * mm_x),
    )


def verdict(features: dict) -> tuple[int, str]:
    angle = features["axis_chord15_deg"]  # как на рис. 2 ТЗ: линия через центры крайних позвонков
    return int(angle > AXIS_LIMIT_DEG), f"наклон оси {angle:.1f}° при допуске {AXIS_LIMIT_DEG:g}°"
