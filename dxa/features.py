"""Все признаки одного снимка: измерения по критериям ТЗ (без обучения)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import axis, roi, rotation, spine


def code_version() -> str:
    """Отпечаток кода измерителей: кэш измерений, посчитанный другим кодом, использовать нельзя."""
    import hashlib

    h = hashlib.sha256()
    root = Path(__file__).parent
    for name in ("axis.py", "spine.py", "roi.py", "rotation.py", "features.py", "data.py"):
        h.update((root / name).read_bytes())
    return h.hexdigest()[:16]


def geometry(array: np.ndarray, spacing, region: str, side: str | None) -> dict:
    """Измерения без обучения; ошибки отдельных измерений не роняют обработку."""
    if region == "spine":
        feats = axis.measure(array, spacing)
        feats.update(spine.measure(array, spacing))
        return feats
    feats = roi.measure(array, spacing, side)
    feats.update(rotation.measure(array, spacing, side))
    return feats
