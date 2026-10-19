"""Модель контроля качества: решение по каждому критерию ТЗ.

| Область | Критерий (метка организатора) | Как решается |
|---|---|---|
| позвоночник | Не выравнена ось позвоночника | правило ТЗ: наклон > 5° (линия через крайние позвонки) |
| позвоночник | Некорректная укладка | правило ТЗ: не видны гребни подвздошных костей |
| позвоночник | Присутствуют посторонние предметы | сила тонких ярких линий вне столба, порог по обучению |
| бедро | Некорректная область интереса | правило ТЗ: < 3 см ниже малого вертела до края кадра |
| бедро | Некорректная укладка (ротация) | среднее двух оценок: малый вертел (U-образно), офсет головки |

Обучаемые параметры (fit): медиана выступа малого вертела у нормы, эталонные распределения
для перевода оценок в процентили, пороги для двух оценок, калибровка вероятностей.
Нейросетевых признаков нет: замороженный SigLIP в честной CV не улучшал ротацию бедра
(AUC 0.852 с ним против 0.863 без него), а тянул за собой torch и 1.6 ГБ весов.
Пороги подбираются по F1 на out-of-fold предсказаниях ОБУЧАЮЩЕЙ части.

Для обучаемых порогов (посторонние предметы, ротация бедра) запоминаются три режима
(dxa/settings.py): balanced — максимум F1 (умолчание), sensitive — найти не меньше 90%
нарушений обучающей выборки, specific — не больше 5% ложных тревог. Пороги ТЗ и включение
критериев задаются в Settings; без настроек поведение прежнее.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .data import LABEL_AXIS, LABEL_FOREIGN, LABEL_POSITIONING, LABEL_ROI
from .settings import MODE_TITLES, SENSITIVE_TARGET, SPECIFIC_TARGET, Settings

_DEFAULT = Settings()
# умолчания (ТЗ); сами значения и пояснения — в dxa/settings.py
AXIS_LIMIT_DEG = _DEFAULT.axis_limit_deg
ILIAC_MIN_FRAC = _DEFAULT.iliac_min_frac
ROI_BOTTOM_CM = _DEFAULT.roi_bottom_cm
FIELD_MIN_CM = _DEFAULT.field_min_cm
ISCHIUM_MIN_FRAC = _DEFAULT.ischium_min_frac

# ключ критерия в настройках (Settings.criteria) по области и метке организатора
CRITERION_KEY = {
    ("spine", LABEL_AXIS): "spine_axis", ("spine", LABEL_POSITIONING): "spine_positioning",
    ("spine", LABEL_FOREIGN): "spine_foreign", ("femur", LABEL_POSITIONING): "femur_positioning",
    ("femur", LABEL_ROI): "femur_roi",
}

CRITERIA = {
    "spine": [LABEL_POSITIONING, LABEL_AXIS, LABEL_FOREIGN],
    "femur": [LABEL_POSITIONING, LABEL_ROI],
}


def _nan(x, fill):
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), x, fill)


def _cdf(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    ref = np.sort(ref)
    return np.searchsorted(ref, x, side="right") / max(len(ref), 1)


def _best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Порог с максимальным F1; кандидаты — квантили оценки (без подгонки под отдельные точки)."""
    cands = np.unique(np.quantile(score, np.linspace(0.5, 0.98, 49)))
    f1 = [f1_score(y, score >= t, zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1))])


def _tz(st: Settings, name: str) -> str:
    """«ТЗ » перед значением порога, пока он не изменён пользователем."""
    return "" if st.note(name) else "ТЗ "


def _mode_thresholds(y: np.ndarray, score: np.ndarray, balanced: float) -> dict[str, float]:
    """Пороги режимов. sensitive не строже balanced, specific не мягче — режимы упорядочены."""
    y, score = np.asarray(y, int), np.asarray(score, float)
    pos, neg = np.sort(score[y == 1]), np.sort(score[y == 0])
    sens = spec = balanced
    if len(pos):  # самый строгий порог, при котором находится не меньше SENSITIVE_TARGET нарушений
        k = int(np.ceil(SENSITIVE_TARGET * len(pos)))
        sens = min(balanced, float(pos[len(pos) - k]))
    if len(neg):  # самый мягкий порог, при котором ложных тревог не больше 1 − SPECIFIC_TARGET
        k = int(np.ceil(SPECIFIC_TARGET * len(neg)))
        spec = max(balanced, float(np.nextafter(neg[k - 1], np.inf)))
    return {"sensitive": sens, "balanced": balanced, "specific": spec}


class QualityModel:
    def fit(self, feats: list[dict], regions: list[str], labels: list[dict]):
        regions = np.array(regions)
        # --- позвоночник: посторонние предметы — один порог по обучению
        sp = np.nonzero(regions == "spine")[0]
        y_fo = np.array([labels[i].get(LABEL_FOREIGN, 0) for i in sp])
        ridge = _nan([feats[i].get("fo_ridge_p99", np.nan) for i in sp], 0.0)
        self.fo_threshold = _best_threshold(y_fo, ridge)
        self.fo_thresholds = _mode_thresholds(y_fo, ridge, self.fo_threshold)
        # Второй детектор — объекты ПОВЕРХ столба (замечание экспертов 16.10). Порог консервативный:
        # максимум по нормам обучающей части (ложных тревог на обучении нет), срабатывает только на
        # сильном сигнале — чтобы не потерять предмет, наложенный на позвоночник, не вредя специфичности.
        over = _nan([feats[i].get("fo_over_spine_p99", np.nan) for i in sp], 0.0)
        neg_in = over[y_fo == 0]
        self.fo_over_spine_threshold = float(neg_in.max() * 1.02) if len(neg_in) else float("inf")
        # предметы на кости — свои пороги по зонам: максимум по нормам обучающей части × 1.02
        # (край гребня подвздошной кости и структура позвонков не дают ложных тревог на обучении)
        self.fo_bone_thresholds = {}
        for zone in ("pelvis", "column"):
            vals = np.array([feats[i].get(f"fo_{zone}_max", np.nan) for i in sp], float)[y_fo == 0]
            vals = vals[np.isfinite(vals)]
            # нет признака у норм (старые измерения) или он нулевой — зона не проверяется, а не «всё — предмет»
            self.fo_bone_thresholds[zone] = float(vals.max() * 1.02) if len(vals) and vals.max() > 0 else float("inf")

        # --- бедро: укладка/ротация — две оценки
        fe = np.nonzero(regions == "femur")[0]
        y = np.array([labels[i].get(LABEL_POSITIONING, 0) for i in fe])
        area = np.array([feats[i].get("rot_lt_area_mm2", np.nan) for i in fe], float)
        off = np.array([feats[i].get("rot_offset_mm", np.nan) for i in fe], float)
        self.area_fill, self.off_fill = float(np.nanmedian(area)), float(np.nanmedian(off))
        area, off = _nan(area, self.area_fill), _nan(off, self.off_fill)
        # площадь выступа — величина «во сколько раз»: гладкий контур 1 мм² и норма 6 мм² отличаются
        # так же сильно, как крупный вертел 36 мм²; в линейной шкале переротация недооценивалась
        area = np.log1p(np.clip(area, 0, None))
        self.area_norm = float(np.median(area[y == 0]))
        self.ref_u = np.abs(area - self.area_norm)
        self.ref_off = -off
        score = self._femur_score(area, off)
        self.femur_threshold = _best_threshold(y, score)
        self.femur_thresholds = _mode_thresholds(y, score, self.femur_threshold)

        # --- калибровка: оценка каждого критерия -> вероятность нарушения (для quality_prob)
        self.calib = {}
        spine_out = [self.predict(feats[i], "spine") for i in sp]
        for L in (LABEL_AXIS, LABEL_POSITIONING, LABEL_FOREIGN):
            idx = [k for k, i in enumerate(sp) if L in labels[i]]
            self._fit_calib(("spine", L), [spine_out[k][L][1] for k in idx], [labels[sp[k]][L] for k in idx])
        roi_idx = [i for i in fe if LABEL_ROI in labels[i]]
        self._fit_calib(("femur", LABEL_ROI), [self.predict(feats[i], "femur")[LABEL_ROI][1] for i in roi_idx],
                        [labels[i][LABEL_ROI] for i in roi_idx])
        self._fit_calib(("femur", LABEL_POSITIONING), score, y)
        return self

    def _fit_calib(self, key, score, y):
        s = np.asarray(score, float)
        y = np.asarray(y, int)
        mu, sd = float(s.mean()), float(s.std() + 1e-9)
        lr = LogisticRegression(C=1.0).fit(((s - mu) / sd)[:, None], y) if 0 < y.sum() < len(y) else None
        self.calib[key] = (mu, sd, lr, float(y.mean()))

    def probabilities(self, out: dict, region: str) -> tuple[dict, float]:
        """Вероятность нарушения по каждому критерию и итоговая quality_prob = P(хотя бы одно нарушение)."""
        probs = {}
        for L, (_, score, _) in out.items():
            mu, sd, lr, base = self.calib.get((region, L), (0.0, 1.0, None, 0.0))
            probs[L] = float(lr.predict_proba([[(score - mu) / sd]])[0, 1]) if lr is not None else base
        return probs, float(1 - np.prod([1 - p for p in probs.values()]))

    def _femur_score(self, area, off):
        u = np.abs(area - self.area_norm)
        return (_cdf(self.ref_u, u) + _cdf(self.ref_off, -off)) / 2

    def foreign_thresholds(self, mode: str = "balanced") -> dict:
        """Пороги детектора посторонних предметов по зонам — для решения и для рамок на картинке."""
        bone = getattr(self, "fo_bone_thresholds", {})
        return dict(lateral=self.threshold("fo", mode), pelvis=bone.get("pelvis", float("inf")),
                    column=bone.get("column", float("inf")))

    def threshold(self, name: str, mode: str = "balanced") -> float:
        """Обучаемый порог ("fo" или "femur") для режима; у модели без режимов — порог по F1."""
        table = getattr(self, f"{name}_thresholds", None)
        return float(table[mode]) if table else float(getattr(self, f"{name}_threshold"))

    def predict(self, feat: dict, region: str, settings: Settings | None = None) -> dict:
        """{метка: (флаг 0/1, оценка, объяснение)} по включённым критериям области."""
        st = settings or _DEFAULT
        mode_note = "" if st.mode == "balanced" else f", режим «{MODE_TITLES[st.mode]}»"
        out = {}
        if region == "spine":
            lim, iliac = st.axis_limit_deg, st.iliac_min_frac
            ang = feat.get("axis_chord15_deg", np.nan)
            out[LABEL_AXIS] = (int(np.isfinite(ang) and ang > lim), float(_nan([ang], 0)[0]),
                               f"наклон оси {ang:.1f}° (допуск {_tz(st, 'axis_limit_deg')}{lim:g}°){st.note('axis_limit_deg')}")
            fr = feat.get("sp_iliac_frac", np.nan)
            out[LABEL_POSITIONING] = (int(np.isfinite(fr) and fr < iliac), float(-_nan([fr], 0)[0]),
                                      ("гребни подвздошных костей не видны у нижнего края" if np.isfinite(fr) and fr < iliac
                                       else "гребни подвздошных костей видны") + st.note("iliac_min_frac"))
            r = float(_nan([feat.get("fo_ridge_p99", np.nan)], 0)[0])
            thr_all = self.foreign_thresholds(st.mode)
            thr = thr_all["lateral"]
            zones = [("вне столба", r, thr)]
            for key, title in (("pelvis", "поверх таза"), ("column", "поверх позвоночника")):
                zones.append((title, float(_nan([feat.get(f"fo_{key}_max", np.nan)], 0)[0]), thr_all[key]))
            found = [title for title, val, t in zones if val >= t]
            # оценка для вероятности — самая сильная из зон в единицах порога «вне столба»
            score = max([r] + [thr * val / t for _, val, t in zones[1:] if np.isfinite(t) and t > 0])
            out[LABEL_FOREIGN] = (int(bool(found)), score,
                                  "; ".join(f"{title}: {val:.3f} (порог {t:.3f}{mode_note if title == 'вне столба' else ''})"
                                            for title, val, t in zones)
                                  + (f" — обнаружено {', '.join(found)}" if found else ""))
        else:
            b = feat.get("roi_margin_bottom_cm", np.nan)
            h = feat.get("field_height_cm", np.nan)
            bottom_min, field_min = st.roi_bottom_cm, st.field_min_cm
            rel = feat.get("femur_roi_reliable", 1.0)
            short_field = bool(np.isfinite(h) and h < field_min)
            # при ненадёжных ориентирах (короткий скан) решение принимаем ТОЛЬКО по длине поля:
            # отступ до малого вертела не используем (замечание экспертов 16.10)
            short_bottom = bool(rel and np.isfinite(b) and b < bottom_min)
            # оценка — насколько не хватает до нормы ТЗ (больше — хуже); считается по умолчаниям ТЗ,
            # чтобы вероятность нарушения не зависела от пользовательских порогов
            deficit = max(ROI_BOTTOM_CM - float(_nan([b], 0)[0]), FIELD_MIN_CM - float(_nan([h], FIELD_MIN_CM)[0]))
            out[LABEL_ROI] = (int(short_bottom or short_field), deficit,
                              f"ниже малого вертела {b:.1f} см (норма {_tz(st, 'roi_bottom_cm')}≥ {bottom_min:g}){st.note('roi_bottom_cm')}; длина скана "
                              f"{h:.1f} см (минимум {field_min:g}){st.note('field_min_cm')}; сверху {feat.get('roi_margin_top_cm', np.nan):.1f}, "
                              f"медиально {feat.get('roi_margin_medial_cm', np.nan):.1f} см"
                              + ("; скан слишком короткий — бедро не помещается" if short_field else "")
                              + ("; ориентиры ненадёжны — решение только по длине скана" if not rel else ""))
            area_mm2 = float(_nan([feat.get("rot_lt_area_mm2", np.nan)], self.area_fill)[0])
            area = float(np.log1p(max(area_mm2, 0.0)))
            off = float(_nan([feat.get("rot_offset_mm", np.nan)], self.off_fill)[0])
            s = float(self._femur_score(np.array([area]), np.array([off]))[0])
            kind = "переротация (контур гладкий)" if area < self.area_norm else "недоротация (малый вертел крупный)"
            isch = feat.get("femur_ischium_frac", np.nan)
            no_ischium = bool(np.isfinite(isch) and isch < st.ischium_min_frac)
            fthr = self.threshold("femur", st.mode)
            rotated = s >= fthr
            out[LABEL_POSITIONING] = (int(rotated or no_ischium), s,
                                      f"малый вертел {area_mm2:.0f} мм² (норма ~{np.expm1(self.area_norm):.0f}), офсет головки {off:.0f} мм; "
                                      f"оценка ротации {s:.2f} (порог {fthr:.2f}{mode_note})"
                                      + (f"; вероятно {kind}" if rotated else "")
                                      + ("; седалищная кость не попала в кадр" if no_ischium else "")
                                      + st.note("ischium_min_frac"))
        return {L: v for L, v in out.items() if st.criteria.get(CRITERION_KEY[(region, L)], True)}
