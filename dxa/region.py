"""Область снимка (позвоночник / бедро) по анатомии — без зависимости от размера снимка.

Нужна, когда в тегах и имени файла подсказки нет, а снимок не «родного» размера аппарата
организатора (другой аппарат, масштабированная копия): правило ширины тогда угадывает.
Признаки инвариантны к масштабу: снимок приводится к ширине 96 пикселей, затем
  * симметрия слева-направо — позвоночник почти симметричен, бедро нет;
  * яркая полоса в центре и её устойчивость по высоте — столб позвонков идёт по центру кадра;
  * перекос яркости между половинами — у бедра головка и таз с одной стороны.
Логистическая регрессия на 252 снимках: 248 из 252 при кросс-валидации по исследованиям,
все ошибки неуверенные (вероятность 0.46–0.65); устойчиво к увеличению ×3, уменьшению ×0.5,
обрезке и рамкам.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WIDTH = 96
# ширина снимков аппарата организатора (позвоночник 300, бедро 248–280): в этом диапазоне
# правило ширины точнее анатомии (252 из 252), вне его — анатомия
NATIVE_WIDTH_PX = (240, 320)
CONFIDENT = 0.9   # анатомия перекрывает правило ширины, только если уверена


def features(array: np.ndarray) -> np.ndarray:
    a = np.asarray(array, np.float32)
    lo, hi = np.percentile(a, (1, 99))
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    h, w = a.shape
    img = Image.fromarray((a * 255).astype(np.uint8)).resize((WIDTH, max(int(round(h * WIDTH / w)), 8)), Image.BILINEAR)
    c = np.asarray(img, np.float32) / 255
    w = c.shape[1]
    sym = np.corrcoef(c.ravel(), c[:, ::-1].ravel())[0, 1]
    col = c.mean(0)
    col = (col - col.mean()) / (col.std() + 1e-6)
    center = col[w // 3: 2 * w // 3].mean() - np.r_[col[: w // 4], col[-w // 4:]].mean()
    col_sym = np.corrcoef(col, col[::-1])[0, 1]
    bands = np.array_split(c, 6, axis=0)
    peaks = np.array([np.argmax(np.convolve(b.mean(0), np.ones(9) / 9, "same")) / w for b in bands])
    lr = abs(c[:, : w // 2].mean() - c[:, w // 2:].mean())
    f = np.array([sym, center, col_sym, np.mean(np.abs(peaks - 0.5)), peaks.std(), lr], float)
    return np.nan_to_num(f)


def fit(arrays: list[np.ndarray], is_spine: list[int]):
    X = np.array([features(a) for a in arrays])
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(X, np.asarray(is_spine, int))


def spine_probability(model, array: np.ndarray) -> float:
    return float(model.predict_proba(features(array)[None])[0, 1])
