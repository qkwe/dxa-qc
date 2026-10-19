"""Позвоночник: укладка и посторонние предметы — измерения без обучения.

ТЗ: «корректная укладка (на нижнем уровне сканирования визуализированы верхние края подвздошных
костей, верхний уровень — половина тела позвонка Th12)»; «отсутствие посторонних предметов,
выраженных артефактов или наложений (металлические артефакты от одежды)».

Укладка: гребни подвздошных костей — яркие «крылья» по бокам от столба у нижнего края кадра;
сверху у Th12 отходят 12-е рёбра. Если кадр обрезан, крыльев нет.
Посторонние предметы: косточки белья и металл — тонкие очень яркие линии/пятна; рёбра тоже
дугообразные, но шире и бледнее, поэтому отбор по ширине и яркости гребня (ridge).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import axis

Spacing = tuple[float, float]


def _column_center(array: np.ndarray, spacing: Spacing) -> tuple[np.ndarray, np.ndarray, float]:
    ys, xs = axis.centerline(array, spacing)
    width = axis.VERTEBRA_WIDTH_MM / spacing[0]
    return ys, xs, width


def _ridges(values: np.ndarray, sigma: float) -> np.ndarray:
    """Яркие тонкие линии: наибольшее по модулю отрицательное собственное число гессиана."""
    g = ndimage.gaussian_filter(values, sigma)
    gyy = ndimage.gaussian_filter(values, sigma, order=(2, 0))
    gxx = ndimage.gaussian_filter(values, sigma, order=(0, 2))
    gxy = ndimage.gaussian_filter(values, sigma, order=(1, 1))
    tmp = np.sqrt((gxx - gyy) ** 2 + 4 * gxy**2)
    lam = (gxx + gyy - tmp) / 2  # наименьшее собственное число: сильно отрицательно на ярком гребне
    del g
    return np.clip(-lam, 0, None) * sigma**2


def measure(array: np.ndarray, spacing: Spacing) -> dict:
    return _analyse(array, spacing)[0]


def foreign_boxes(array: np.ndarray, spacing: Spacing, thresholds: dict, limit: int = 6) -> list[dict]:
    """Где детектор посторонних предметов увидел сигнал: рамки (x0, y0, x1, y1) в пикселях снимка.

    Те же карты и пороги, что в решении: вне столба (r1), поверх таза и поверх позвоночника (r1 по окну 5).
    Пиксели выше порога решения склеиваются в объекты. Чтобы не обводить шум (отдельные пиксели
    чуть выше порога), показываются: самый сильный объект всегда (детектор сработал — нужно показать
    где), остальные — если выше порога в 1.5 раза (косточки белья с обеих сторон бывают разной
    яркости — сравнивать с самым сильным нельзя). На обучающем наборе предметы дают отклик в 2–8 раз
    выше порога, шум — в 1.1–1.4 раза. Зоны на кости — только если сработал детектор этой зоны.
    Только для разметки, на решение не влияет. thresholds: {"lateral", "pelvis", "column"}."""
    maps = _analyse(array, spacing)[1]
    if not maps:
        return []
    detectors = [("вне столба", maps["r1"], maps["area"], thresholds["lateral"])]
    for where, zone, key in (("поверх таза", maps["pelvis_zone"], "pelvis"), ("поверх позвоночника", maps["inside_zone"], "column")):
        thr = thresholds.get(key, float("inf"))
        if zone.any() and maps["ext"][zone].max() >= thr:
            detectors.append((where, maps["ext"], zone, thr))
    out = []
    for where, strength, zone, thr in detectors:
        hits = (strength >= thr) & zone
        if not hits.any():
            continue
        # рамка — на весь предмет, а не только на самую яркую часть: объект «дорастает» от ядра (выше
        # порога) по соседним пикселям выше половины порога (гистерезис, как в детекторе границ Канни)
        grown = ndimage.binary_dilation((strength >= 0.5 * thr) & zone, iterations=1)
        lab_g, _ = ndimage.label(grown)
        keep = np.isin(lab_g, np.unique(lab_g[hits]))
        keep &= lab_g > 0
        lab, _ = ndimage.label(ndimage.binary_dilation(keep, iterations=2))
        found = []
        for k, sl in enumerate(ndimage.find_objects(lab), start=1):
            comp = (lab[sl] == k) & hits[sl]
            if comp.sum() >= 3:  # одиночные пиксели — шум
                y, x = sl
                found.append(dict(box=(x.start, y.start, x.stop, y.stop), where=where,
                                  strength=float(strength[sl][comp].max()), ratio=float(strength[sl][comp].max() / thr)))
        found.sort(key=lambda f: -f["strength"])
        if found:
            out += [found[0]] + [f for f in found[1:] if f["ratio"] >= 1.5]
    return sorted(out, key=lambda f: -f["ratio"])[:limit]


def _analyse(array: np.ndarray, spacing: Spacing) -> tuple[dict, dict]:
    values = axis.normalize(array)
    rows, cols = values.shape
    sx, sy = spacing
    valid = axis.valid_rows(values)
    vr = np.nonzero(valid)[0]
    if len(vr) < 20:
        return {}, {}
    top, bottom = int(vr[0]), int(vr[-1])
    ys, xs, width = _column_center(array, spacing)
    center = float(np.median(xs)) if len(xs) else cols / 2

    # кость по порогу между мягкими тканями и плотной костью
    soft, dense = np.percentile(values[valid], (50, 98))
    bone = values > soft + (dense - soft) * 0.25
    xx = np.arange(cols)[None, :]
    lateral = np.abs(xx - center) > width * 0.9  # вне позвоночного столба

    def band(y0, y1):
        m = np.zeros_like(bone)
        m[max(y0, 0) : max(y1, 0)] = True
        return m

    h = bottom - top
    low = band(bottom - int(h * 0.25), bottom + 1) & lateral
    high = band(top, top + int(h * 0.2)) & lateral
    # гребни: доля кости по бокам внизу и насколько высоко они поднимаются от нижнего края
    iliac_frac = float(bone[low].mean()) if low.any() else 0.0
    lat_rows = np.nonzero((bone & lateral)[top : bottom + 1].mean(axis=1) > 0.08)[0] + top
    low_rows = lat_rows[lat_rows > top + h * 0.5]
    iliac_height_mm = float((bottom - low_rows.min()) * sy) if len(low_rows) else 0.0
    ribs_frac = float(bone[high].mean()) if high.any() else 0.0

    # посторонние предметы: тонкие очень яркие гребни вне столба.
    # Поле дополнения (нули по краям и прямоугольные вырезы) даёт резкий перепад, который
    # детектор принимает за линию, — заполняем его ближайшими значениями и не смотрим у границы.
    pad = array == 0
    pad = ndimage.binary_opening(pad, np.ones((3, 3)))
    filled = values
    if pad.any():
        idx = ndimage.distance_transform_edt(pad, return_distances=False, return_indices=True)
        filled = values[tuple(idx)]
    inside = ~ndimage.binary_dilation(pad, iterations=5)
    inside[:4], inside[-4:], inside[:, :4], inside[:, -4:] = False, False, False, False
    r1 = _ridges(filled, 1.0) * inside
    r3 = _ridges(filled, 3.0)
    thin = (r1 > 0.12) & (r1 > 1.5 * r3) & (values > np.percentile(values[valid], 97)) & lateral & valid[:, None]
    thin = ndimage.binary_opening(thin, np.ones((1, 1)))
    lab, n = ndimage.label(thin, np.ones((3, 3)))
    sizes = ndimage.sum(thin, lab, range(1, n + 1)) if n else np.array([])
    long_objs = int((sizes >= 12).sum())
    # компактные сверхъяркие пятна (скобы, металл): ярче 99.8 перцентиля кости
    very = (values >= 0.995) & lateral & valid[:, None]
    lab2, n2 = ndimage.label(very)
    spots = int((ndimage.sum(very, lab2, range(1, n2 + 1)) >= 4).sum()) if n2 else 0

    # зона поиска предметов: вне столба, внутри снятого поля (без нулей дополнения — раньше они
    # попадали в перцентиль и размывали его по-разному на каждом снимке) и без таза:
    # края гребней подвздошных костей давали большинство ложных тревог
    area = lateral & valid[:, None] & inside
    # таз ищем только вне столба: иначе позвоночник склеивает с тазом рёбра и всё, что к ним
    # прилегает (в т.ч. косточки белья вверху), и они исключаются вместе с тазом
    pelvis_mask = ndimage.binary_closing(bone, np.ones((5, 5))) & lateral
    lab3, _ = ndimage.label(pelvis_mask)
    ids = [k for k in np.unique(lab3[max(bottom - 3, 0) : bottom + 1]) if k]
    big = [k for k in ids if (lab3 == k).sum() > bone.size * 0.005]  # крупные кости у нижнего края — таз
    if big:
        area &= ~ndimage.binary_dilation(np.isin(lab3, big), iterations=4)
    # и без нижней четверти кадра сбоку — зоны гребней подвздошных костей (та же зона `low`, что для
    # укладки): их края остаются главным источником ложных тревог, когда таз не отделился выше.
    # Проверено 19.10 вложенной CV: предметы F1 0.757 → 0.889 (чувств. 0.82 → 0.94, специф. 0.93 → 0.96),
    # одинаково при исключении 15–30% высоты. Цена: предмет сбоку на уровне таза не ищется (над столбом —
    # ищет второй детектор).
    area &= ~low
    fo_score = float(np.percentile(r1[area], 99.5)) if area.any() else 0.0

    # Второй, независимый детектор — объекты ПОВЕРХ позвоночного столба (замечание экспертов 16.10:
    # раньше зона поиска жёстко исключала столб, поэтому предмет, наложенный на позвоночник,
    # пропускался). Внутри столба кость сама яркая, поэтому берём «тонкость» гребня: (r1 - r3) —
    # отклик тонкой линии минус отклик на крупном масштабе. Кость (позвонок) широкая → r1≈r3 → ≈0;
    # металл/скоба — тонкая яркая структура → r1 >> r3. Проверено: синтетический предмет над столбом
    # даёт 0.06–0.07 против 0.01 у нормы; на обучающих данных ложных срабатываний нет.
    thin_ridge = np.clip(r1 - r3, 0, None)
    inside_zone = (~lateral) & valid[:, None] & inside
    fo_over_spine_score = float(np.percentile(thin_ridge[inside_zone], 99.5)) if inside_zone.any() else 0.0

    # Предметы НА КОСТИ (замечание экспертов: исключать их нельзя) — отдельно по зонам, каждая со своим
    # порогом: поверх таза (нижняя четверть кадра сбоку) и поверх позвоночника. Признак — отклик тонкой
    # яркой линии, усреднённый по окну 5 пикселей (протяжённый предмет сильнее отдельных ярких точек кости),
    # максимум по зоне. Проверено 20.10 на стенде из настоящих предметов, вклеенных в нормальные снимки
    # (с перенормировкой, как у аппарата): поверх таза находится ~92% (было 8%), поверх позвоночника 7–29%
    # (было 5%), ложных тревог на нормах не больше прежнего.
    ext = ndimage.uniform_filter(r1, 5)
    pelvis_zone = low & valid[:, None] & inside
    fo_pelvis = float(ext[pelvis_zone].max()) if pelvis_zone.any() else 0.0
    fo_column = float(ext[inside_zone].max()) if inside_zone.any() else 0.0

    maps = dict(r1=r1, area=area, thin_ridge=thin_ridge, inside_zone=inside_zone, ext=ext, pelvis_zone=pelvis_zone)
    return dict(
        sp_iliac_frac=iliac_frac,
        sp_iliac_height_mm=iliac_height_mm,
        sp_ribs_frac=ribs_frac,
        sp_field_height_mm=float(h * sy),
        sp_center_offset_mm=float((center - cols / 2) * sx),
        fo_thin_pixels=float(thin.sum()),
        fo_thin_objects=float(long_objs),
        fo_spots=float(spots),
        fo_ridge_p99=fo_score,
        fo_over_spine_p99=fo_over_spine_score,
        fo_pelvis_max=fo_pelvis,
        fo_column_max=fo_column,
    ), maps
