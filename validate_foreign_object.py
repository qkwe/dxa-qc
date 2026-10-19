"""Проверка детектора инородных тел ПОВЕРХ позвоночного столба (замечание экспертов 16.10).

    python validate_foreign_object.py --data <НД_для_обучения>

На качественный снимок позвоночника накладывается синтетический инородный предмет (тонкая
яркая линия поверх столба — как косточка белья/скоба) и проверяется, что система его замечает.
Раньше зона поиска жёстко исключала столб, поэтому такой предмет пропускался.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import joblib

from dxa import features, spine
from dxa.data import LABEL_FOREIGN, load_dataset
from dxa.model import QualityModel

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="models/quality_model.joblib")
args = ap.parse_args()

model: QualityModel = joblib.load(args.model)["quality"]
images = [i for i in load_dataset(args.data)
          if i.region == "spine" and i.quality == 0 and i.labels.get(LABEL_FOREIGN, 0) == 0]
if not images:
    print("нет качественных снимков позвоночника в наборе")
    sys.exit(1)

im = images[0]
base = im.array.copy()
rows, cols = base.shape


def verdict(arr):
    f = features.geometry(arr, im.spacing, "spine", None)
    flag, score, text = model.predict(f, "spine")[LABEL_FOREIGN]
    return flag, f["fo_ridge_p99"], f["fo_over_spine_p99"], text


flag0, lat0, over0, _ = verdict(base)
print(f"оригинал (норма):          вердикт={flag0}  вне столба={lat0:.4f}  поверх столба={over0:.4f}")
print(f"  пороги: {model.foreign_thresholds()}")

ok = flag0 == 0
for dy, tag in ((0.5, "центр (над позвонком)"), (0.25, "верхняя треть"), (0.75, "нижняя треть")):
    arr = base.copy()
    y0 = int(rows * dy)
    arr[y0, cols // 2 - 50:cols // 2 + 50] = 255
    arr[y0 + 1, cols // 2 - 50:cols // 2 + 50] = 255
    flag, lat, over, _ = verdict(arr)
    print(f"  + предмет {tag:24s} вердикт={flag}  вне столба={lat:.4f}  поверх столба={over:.4f}")
    ok = ok and flag == 1

print("\nИТОГ:", "детектор объектов поверх столба работает" if ok else "ПРОВАЛ")
sys.exit(0 if ok else 2)
