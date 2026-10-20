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
    rows.append(dict(key=key, state=e["state"], err=e.get("error_mm"), label=label, pred=pred.get(uid)))
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
    t = {s: [r["label"] for r in nv if r["state"] == s] for s in ("not_visible", "point")}
    report["«не виден» и разметка нарушений"] = {
        "доля нарушений, вертел не виден": round(float(np.mean(t["not_visible"])), 3) if t["not_visible"] else None,
        "доля нарушений, вертел виден": round(float(np.mean(t["point"])), 3) if t["point"] else None,
        "не виден": len(t["not_visible"]), "виден": len(t["point"])}
    print("3.", report["«не виден» и разметка нарушений"])

Path("out").mkdir(exist_ok=True)
Path("out/lt_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print("→ out/lt_validation.json")
