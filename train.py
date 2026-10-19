"""Обучение итоговой модели на наборе организатора (НД_для_обучения).

    python train.py --data C:\dxa --out models/quality_model.joblib

Шаги: чтение DICOM и меток → измерения по критериям ТЗ → модель качества (пороги и калибровка
по обучающим данным). Для честных метрик см. evaluate_final.py (тот же код модели, CV по исследованиям).
"""

import argparse
import datetime as dt
import json
import pickle
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from dxa import features, region
from dxa.data import load_dataset
from dxa.model import QualityModel


def main():
    try:  # cp1251-консоль на русской Windows не должна ронять печать
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="распакованный НД_для_обучения (с xlsx разметкой)")
    ap.add_argument("--out", default="models/quality_model.joblib")
    ap.add_argument("--cache", default="out/train_geo.pkl", help="кэш измерений (ускоряет повторное обучение)")
    ap.add_argument("--refresh-geo", action="store_true", help="пересчитать измерения, не брать из кэша")
    args = ap.parse_args()

    images = load_dataset(args.data)
    print(f"снимков {len(images)}", flush=True)
    # кэш годится, только если посчитан этим же кодом измерителей (иначе в нём нет новых признаков
    # или они посчитаны по-старому — модель молча обучится на устаревших измерениях)
    version = features.code_version()
    cache = {}
    if Path(args.cache).exists() and not args.refresh_geo:
        saved = pickle.load(open(args.cache, "rb"))
        if isinstance(saved, dict) and saved.get("__version__") == version:
            cache = saved
        else:
            print("кэш измерений от другой версии кода — пересчитываю", flush=True)
    cache["__version__"] = version
    geo = []
    for n, im in enumerate(images):
        key = f"{im.study_uid}:{im.image_uid}"
        if key not in cache:
            cache[key] = features.geometry(im.array, im.spacing, im.region, im.side)
            if n % 20 == 0:
                print(f"  {n}/{len(images)}", flush=True)
        geo.append(cache[key])
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cache, open(args.cache, "wb"))

    lab = [i for i, im in enumerate(images) if im.quality is not None]
    model = QualityModel().fit([geo[i] for i in lab], [images[i].region for i in lab], [images[i].labels for i in lab])
    # область по анатомии — запасной способ для снимков не «родного» размера (dxa/region.py)
    arrays = [im.array for im in images]
    is_spine = np.array([im.region == "spine" for im in images], int)
    groups = np.array([im.study_uid for im in images])
    X = np.array([region.features(a) for a in arrays])
    oof = np.zeros(len(images))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, is_spine, groups):
        oof[te] = region.fit([arrays[i] for i in tr], is_spine[tr]).predict_proba(X[te])[:, 1]
    region_model = region.fit(arrays, is_spine)
    region_cv = f"{int(((oof >= 0.5) == is_spine).sum())} из {len(images)}"
    print("область по анатомии, кросс-валидация по исследованиям:", region_cv, flush=True)

    meta = dict(trained=dt.datetime.now().isoformat(timespec="seconds"), n_images=len(images), n_labeled=len(lab),
                fo_threshold=model.fo_threshold, fo_over_spine_threshold=model.fo_over_spine_threshold,
                femur_threshold=model.femur_threshold, area_norm=model.area_norm,
                fo_thresholds=model.fo_thresholds, femur_thresholds=model.femur_thresholds,
                region_anatomy_cv=region_cv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(quality=model, region=region_model, meta=meta), args.out)
    json.dump(meta, open(Path(args.out).with_suffix(".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("сохранено", args.out, meta)


if __name__ == "__main__":
    main()
