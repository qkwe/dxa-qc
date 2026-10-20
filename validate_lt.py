"""Точность нахождения малого вертела — по отметкам из инструмента врача (/review → «Отметить малый вертел»).

    python validate_lt.py --data <НД_для_обучения> [--landmarks out/feedback/landmarks.json]

Отметка — анатомия, а не диагноз: где вершина малого вертела или «не виден» (контур гладкий). Скрипт сравнивает
её с точкой наибольшего выступа медиального контура, которую находит сервис (dxa/rotation.py), и отвечает:
  1. как далеко точка сервиса от настоящего малого вертела (медиана, доля в пределах 5 / 10 мм);
  2. связаны ли ошибки решения по ротации (OOF честной CV, out/oof_final.pkl) с тем, что вертел найден не там;
  3. согласуется ли «не виден» с разметкой нарушений (переротация — контур гладкий, ТЗ рис. 5).
Итог — out/lt_validation.json.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from dxa.data import LABEL_POSITIONING, load_dataset

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", required=True)
ap.add_argument("--landmarks", default="out/feedback/landmarks.json")
ap.add_argument("--oof", default="out/oof_final.pkl")
args = ap.parse_args()

marks = json.loads(Path(args.landmarks).read_text(encoding="utf-8"))
# сопоставление по SOPInstanceUID: исследование в наборе организатора названо по каталогу, в сервисе — по тегу
ims = {i.image_uid: i for i in load_dataset(args.data) if i.region == "femur"}
oof = pickle.load(open(args.oof, "rb")) if Path(args.oof).exists() else None
pred = {}
if oof:
    for k, key in enumerate(oof["keys"]):
        pred[key.split(":", 1)[1]] = oof["flags"][LABEL_POSITIONING][k] >= 0.5

rows = []
for key, e in marks.items():
    uid = key.split(":", 1)[1]
    im = ims.get(uid)
    label = im.labels.get(LABEL_POSITIONING) if im is not None else None
    state = "faint" if e.get("faint") else e.get("state")
    rows.append(dict(key=key, state=state, err=e.get("error_mm"), label=label, pred=pred.get(uid),
                     rotation=e.get("rotation"), area=None))
print(f"отметок {len(rows)}, из них снимков набора организатора {sum(r['label'] is not None for r in rows)}")

err = np.array([r["err"] for r in rows if r["err"] is not None], float)
report = {"отметок": len(rows), "не виден": sum(r["state"] == "not_visible" for r in rows)}
if len(err):
    report["ошибка нахождения, мм"] = dict(n=int(len(err)), median=round(float(np.median(err)), 1),
                                           p90=round(float(np.percentile(err, 90)), 1),
                                           within_5mm=round(float((err <= 5).mean()), 3),
                                           within_10mm=round(float((err <= 10).mean()), 3))
    print("1. точка сервиса от отмеченного малого вертела:", report["ошибка нахождения, мм"])

lab = [r for r in rows if r["label"] is not None and r["pred"] is not None and r["err"] is not None]
if lab:
    right = np.array([r["err"] for r in lab if r["pred"] == r["label"]])
    wrong = np.array([r["err"] for r in lab if r["pred"] != r["label"]])
    far = lambda a: round(float((a > 10).mean()), 3) if len(a) else None
    report["ошибка решения vs нахождение"] = dict(
        верных=len(right), неверных=len(wrong),
        медиана_мм_верные=round(float(np.median(right)), 1) if len(right) else None,
        медиана_мм_неверные=round(float(np.median(wrong)), 1) if len(wrong) else None,
        дальше_10мм_верные=far(right), дальше_10мм_неверные=far(wrong))
    print("2. решения по ротации, верные и неверные:", report["ошибка решения vs нахождение"])

nv = [r for r in rows if r["label"] is not None]
if nv:
    names = {"point": "виден чётко", "faint": "виден слабо", "not_visible": "не виден"}
    report["видимость вертела и разметка нарушений"] = {
        names[s]: dict(бёдер=len(v), доля_нарушений=round(float(np.mean(v)), 3) if v else None)
        for s, v in ((s, [r["label"] for r in nv if r["state"] == s]) for s in names)}
    print("3.", report["видимость вертела и разметка нарушений"])

# 4. тип ротации (рис. 5 ТЗ): согласие с разметкой нарушений и с догадкой сервиса (площадь выступа ниже / выше нормы)
rot = [r for r in rows if r["rotation"] in ("none", "over", "under")]
if rot:
    from dxa import features
    model = None
    try:
        import joblib
        model = joblib.load("models/quality_model.joblib")["quality"]
    except Exception:
        pass
    for r in rot:
        im = ims.get(r["key"].split(":", 1)[1])
        if im is not None:
            r["area"] = features.geometry(im.array, im.spacing, im.region, im.side).get("rot_lt_area_mm2")
    lab = [r for r in rot if r["label"] is not None]
    agree = [(r["rotation"] != "none") == bool(r["label"]) for r in lab]
    out = {"отмечено": len(rot), "совпадает с разметкой нарушений": round(float(np.mean(agree)), 3) if agree else None}
    if model is not None:
        norm = float(np.expm1(model.area_norm))
        typed = [r for r in rot if r["rotation"] in ("over", "under") and r["area"] is not None and np.isfinite(r["area"])]
        guess = ["over" if r["area"] < norm else "under" for r in typed]
        out["догадка сервиса о типе верна"] = round(float(np.mean([g == r["rotation"] for g, r in zip(guess, typed)])), 3) if typed else None
        out["норма площади, мм²"] = round(norm, 1)
    report["тип ротации"] = out
    print("4. тип ротации:", out)

Path("out").mkdir(exist_ok=True)
Path("out/lt_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print("→ out/lt_validation.json")
