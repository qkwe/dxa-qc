"""Модель контроля качества: решение по каждому критерию ТЗ.

| Область | Критерий (метка организатора) | Как решается |
|---|---|---|
| позвоночник | Не выравнена ось позвоночника | правило ТЗ: наклон > 5° (линия через крайние позвонки) |
| позвоночник | Некорректная укладка | правило ТЗ: не видны гребни подвздошных костей |
| позвоночник | Присутствуют посторонние предметы | сила тонких ярких линий вне столба, порог по обучению |
| бедро | Некорректная область интереса | правило ТЗ: < 3 см ниже малого вертела до края кадра |
| бедро | Некорректная укладка (ротация) | среднее двух оценок: выступ медиального контура — шейка и малый вертел (U-образно), офсет головки |

Обучаемые параметры (fit): медиана выступа малого вертела у нормы, эталонные распределения
для перевода оценок в процентили, пороги для двух оценок, калибровка вероятностей (изотоническая, по OOF-оценкам).
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


def _v(x, spec: str = ".1f", unit: str = "") -> str:
    """Число для заключения; не измерено — так и пишем (а не «nan» или подставленную медиану)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "не измерено"
    return f"{x:{spec}}{unit}" if np.isfinite(x) else "не измерено"


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
    def fit(self, feats: list[dict], regions: list[str], labels: list[dict], groups=None):
        """Пороги и правила — по всей обучающей части; калибровка вероятностей — по out-of-fold оценкам
        (внутренние фолды по исследованиям: groups = StudyInstanceUID), иначе калибратор видит оценки
        модели на тех же снимках, по которым подобраны её пороги."""
        self._fit_rules(feats, regions, labels)
        self._fit_calibration(feats, regions, labels, groups)
        return self

    def _fit_rules(self, feats: list[dict], regions: list[str], labels: list[dict]):
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

        # --- ось: наклон при дугообразном изгибе (сколиоз) — не нарушение укладки. Порог стрелы прогиба —
        # по F1 правила «наклон > допуска и изгиб < порога» на обучающей части; не помогает — проверка выключена
        ang = np.array([feats[i].get("axis_chord15_deg", np.nan) for i in sp], float)
        sag = np.array([feats[i].get("axis_sagitta_mm", np.nan) for i in sp], float)
        y_ax = np.array([labels[i].get(LABEL_AXIS, 0) for i in sp])
        tilted = np.nan_to_num(ang) > AXIS_LIMIT_DEG
        self.scoliosis_sagitta_mm = float("inf")
        if np.isfinite(sag).sum() >= 10 and tilted.any():
            base = f1_score(y_ax, tilted, zero_division=0)
            cands = np.unique(np.quantile(sag[np.isfinite(sag)], np.linspace(0.5, 0.98, 25)))
            f1s = [f1_score(y_ax, tilted & ~(np.nan_to_num(sag) >= t), zero_division=0) for t in cands]
            if max(f1s) > base:
                self.scoliosis_sagitta_mm = float(cands[int(np.argmax(f1s))])

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

        return self

    def _scores(self, feats, regions, labels, idx) -> dict:
        """{(область, метка): (оценки, метки)} — оценки модели по снимкам idx с разметкой критерия."""
        out = {}
        for i in idx:
            for L, (_, s, _) in self.predict(feats[i], regions[i]).items():
                if L in labels[i]:
                    sc, ys = out.setdefault((regions[i], L), ([], []))
                    sc.append(s)
                    ys.append(int(labels[i][L]))
        return out

    def _fit_calibration(self, feats, regions, labels, groups=None, folds: int = 5):
        """Оценка критерия → вероятность нарушения (для quality_prob): изотоническая регрессия по OOF-оценкам.
        Изотоника (монотонная ступенчатая функция) не навязывает логистическую форму: в честной CV
        quality_prob ROC-AUC 0.863 против 0.840 у логистической калибровки (та же CV, те же фолды)."""
        from sklearn.isotonic import IsotonicRegression
        from sklearn.model_selection import GroupKFold

        regions = list(regions)
        n = len(feats)
        groups = np.asarray(groups if groups is not None else np.arange(n))
        pool: dict = {}
        k = min(folds, len(np.unique(groups)))
        if k >= 2:
            for tr, te in GroupKFold(k).split(np.zeros(n), groups=groups):
                inner = QualityModel()._fit_rules([feats[i] for i in tr], [regions[i] for i in tr], [labels[i] for i in tr])
                for key, (s, y) in inner._scores(feats, regions, labels, te).items():
                    pool.setdefault(key, ([], []))
                    pool[key][0].extend(s)
                    pool[key][1].extend(y)
        else:  # слишком мало исследований для внутренних фолдов — оценки самой модели
            pool = self._scores(feats, regions, labels, range(n))
        self.calib = {}
        for key, (s, y) in pool.items():
            s, y = np.asarray(s, float), np.asarray(y, int)
            iso = (IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(s, y)
                   if 0 < y.sum() < len(y) else None)
            self.calib[key] = ("iso", iso, float(y.mean()))

    def probabilities(self, out: dict, region: str) -> tuple[dict, float]:
        """Вероятность нарушения по каждому критерию и итоговая quality_prob = P(хотя бы одно нарушение)."""
        probs = {}
        for L, (_, score, _) in out.items():
            c = self.calib.get((region, L))
            if c is None:
                probs[L] = 0.0
            elif c[0] == "iso":
                probs[L] = float(c[1].predict([score])[0]) if c[1] is not None else c[2]
            else:  # модель, сохранённая прежней версией: (mu, sd, логистическая регрессия, доля)
                mu, sd, lr, base = c
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
            sag = feat.get("axis_sagitta_mm", np.nan)
            sthr = getattr(self, "scoliosis_sagitta_mm", float("inf"))
            scol = bool(np.isfinite(ang) and ang > lim and np.isfinite(sag) and sag >= sthr)
            scol_note = (f"; столб изогнут дугой (стрела прогиба {sag:.0f} мм, порог {sthr:.0f}) — похоже на сколиоз, "
                         + ("а не на неправильную укладку: нарушением не считается, проверьте" if st.axis_scoliosis_check
                            else "но проверка сколиоза выключена пользователем — нарушение по наклону")) if scol else ""
            if scol and st.axis_scoliosis_check:
                out[LABEL_AXIS] = (0, float(min(ang, lim)),
                                   f"наклон оси {_v(ang, '.1f', '°')} (допуск {_tz(st, 'axis_limit_deg')}{lim:g}°){st.note('axis_limit_deg')}"
                                   + scol_note)
            else:
                out[LABEL_AXIS] = (int(np.isfinite(ang) and ang > lim), float(_nan([ang], 0)[0]),
                                   f"наклон оси {_v(ang, '.1f', '°')} (допуск {_tz(st, 'axis_limit_deg')}{lim:g}°){st.note('axis_limit_deg')}"
                                   + scol_note)
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
                              f"ниже малого вертела {_v(b, '.1f', ' см')} (норма {_tz(st, 'roi_bottom_cm')}≥ {bottom_min:g}){st.note('roi_bottom_cm')}; длина скана "
                              f"{_v(h, '.1f', ' см')} (минимум {field_min:g}){st.note('field_min_cm')}; сверху {_v(feat.get('roi_margin_top_cm'), '.1f', ' см')}, "
                              f"медиально {_v(feat.get('roi_margin_medial_cm'), '.1f', ' см')}"
                              + ("; скан слишком короткий — бедро не помещается" if short_field else "")
                              + ("; ориентиры ненадёжны — решение только по длине скана" if not rel else ""))
            raw_area, raw_off = feat.get("rot_lt_area_mm2", np.nan), feat.get("rot_offset_mm", np.nan)
            area_mm2 = float(_nan([raw_area], self.area_fill)[0])
            area = float(np.log1p(max(area_mm2, 0.0)))
            off = float(_nan([feat.get("rot_offset_mm", np.nan)], self.off_fill)[0])
            s = float(self._femur_score(np.array([area]), np.array([off]))[0])
            kind = "переротация (контур гладкий)" if area < self.area_norm else "недоротация (малый вертел крупный)"
            isch = feat.get("femur_ischium_frac", np.nan)
            no_ischium = bool(np.isfinite(isch) and isch < st.ischium_min_frac)
            fthr = self.threshold("femur", st.mode)
            rotated = s >= fthr
            out[LABEL_POSITIONING] = (int(rotated or no_ischium), s,
                                      f"выступ медиального контура (шейка и малый вертел) {_v(raw_area, '.0f', ' мм²')} "
                                      f"(норма ~{np.expm1(self.area_norm):.0f}), "
                                      + (f"малый вертел выступает на {feat['rot_lt_true_bump_mm']:.1f} мм, "
                                         if np.isfinite(feat.get("rot_lt_true_bump_mm", np.nan)) else "")
                                      + f"офсет головки {_v(raw_off, '.0f', ' мм')}; "
                                      + ("ротация оценена по типичным значениям (контур не измерен); "
                                         if not (np.isfinite(_nan([raw_area], np.nan)[0]) and np.isfinite(_nan([raw_off], np.nan)[0])) else "")
                                      + f"оценка ротации {s:.2f} (порог {fthr:.2f}{mode_note})"
                                      + (f"; вероятно {kind}" if rotated else "")
                                      + ("; седалищная кость не попала в кадр" if no_ischium else "")
                                      + st.note("ischium_min_frac"))
        return {L: v for L, v in out.items() if st.criteria.get(CRITERION_KEY[(region, L)], True)}
