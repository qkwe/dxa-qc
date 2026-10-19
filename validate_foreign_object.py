"""Стенд детектора посторонних предметов: настоящие предметы, перенесённые на нормальные снимки.

    python validate_foreign_object.py --data /path/to/НД_для_обучения

Зачем: в разметке организатора предметы почти все в мягких тканях (косточки белья), а эксперты
требуют находить и предметы, наложенные на кость. Реальных примеров на кости нет, поэтому:
  1. с размеченных снимков с предметами вырезаются сами предметы (контраст над локальным фоном);
  2. они переносятся на нормальные снимки в три зоны — мягкие ткани, поверх таза, поверх позвоночника;
     металл добавляет ослабление к фону, затем снимок нормируется как у аппарата (максимум прежний);
  3. детектор обучается на одной половине норм (и всех бедер), проверяется на другой (и наоборот):
     доля найденных перенесённых предметов по зонам и доля ложных тревог на чистых нормах.
Это оценка, а не клиническая метрика: предметы настоящие, фон — чужие снимки. Результат:
out/foreign_stand.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from dxa import axis, features, spine
from dxa.data import LABEL_FOREIGN, load_dataset
from dxa.model import QualityModel

ZONES = ("мягкие ткани", "поверх таза", "поверх позвоночника")


def object_patches(ims, model) -> list[np.ndarray]:
    """Контраст настоящих предметов (над фоном, «открытие» диском ~5 пикселей) с размеченных снимков."""
    out = []
    r = 5
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (yy ** 2 + xx ** 2) <= r * r
    for im in ims:
        if not im.labels.get(LABEL_FOREIGN):
            continue
        raw = im.array.astype(np.float32)
        bg = ndimage.grey_opening(raw, footprint=disk)
        for b in spine.foreign_boxes(im.array, im.spacing, model.foreign_thresholds()):
            if b["ratio"] < 2.0 or b["where"] != "вне столба":
                continue
            x0, y0, x1, y1 = b["box"]
            y0, x0, y1, x1 = max(y0 - 4, 0), max(x0 - 4, 0), min(y1 + 4, raw.shape[0]), min(x1 + 4, raw.shape[1])
            c = np.clip(raw[y0:y1, x0:x1] - bg[y0:y1, x0:x1], 0, None)
            c[c < 0.25 * c.max()] = 0
            if c.max() > 20 and (c > 0).sum() >= 8:
                out.append(c)
    return out


def compact_patch(rng, contrasts) -> np.ndarray:
    """Компактный предмет (застёжка, кнопка): эллипс/прямоугольник 4–10 пикселей с контрастом как у настоящих."""
    h, w = rng.integers(4, 11), rng.integers(4, 11)
    yy, xx = np.mgrid[:h, :w]
    m = ((yy - h / 2) ** 2 / (h / 2) ** 2 + (xx - w / 2) ** 2 / (w / 2) ** 2) <= 1 if rng.random() < 0.5 else np.ones((h, w), bool)
    return m * float(rng.choice(contrasts))


def place(raw, contrast, zone, spacing, rng):
    v = axis.normalize(raw)
    rows, cols = raw.shape
    valid = np.nonzero(axis.valid_rows(v))[0]
    top, bottom = valid[0], valid[-1]
    ys, xs = axis.centerline(raw, spacing)
    center = float(np.median(xs)) if len(xs) else cols / 2
    width = axis.VERTEBRA_WIDTH_MM / spacing[0]
    h, w = contrast.shape
    H = bottom - top
    for _ in range(50):
        if zone == "мягкие ткани":
            y = rng.integers(top + 5, top + int(0.6 * H))
            x = int(center + rng.choice([-1, 1]) * rng.uniform(1.2, 2.5) * width)
        elif zone == "поверх позвоночника":
            y = rng.integers(top + int(0.2 * H), top + int(0.75 * H))
            x = int(center + rng.uniform(-0.3, 0.3) * width)
        else:
            lo, hi = bottom - int(0.22 * H), bottom - h - 2
            if lo >= hi:
                return None
            y = rng.integers(lo, hi)
            x = int(center + rng.choice([-1, 1]) * rng.uniform(1.1, 2.4) * width)
        y0, x0 = int(y - h // 2), int(x - w // 2)
        if not (2 <= y0 and y0 + h < rows - 2 and 2 <= x0 and x0 + w < cols - 2 and (raw[y0:y0 + h, x0:x0 + w] > 0).all()):
            continue
        if zone == "поверх таза" and v[y0:y0 + h, x0:x0 + w].mean() <= np.percentile(v[top:bottom], 70):
            continue   # таз — именно на кости
        out = raw.astype(np.float32).copy()
        out[y0:y0 + h, x0:x0 + w] += contrast
        out *= float(raw.max()) / float(out.max())   # аппарат нормирует снимок: максимум прежний
        return np.clip(np.rint(out), 0, 255).astype(raw.dtype)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="распакованный НД_для_обучения")
    ap.add_argument("--per-zone", type=int, default=80, help="перенесённых предметов на зону в каждой половине")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ims = [i for i in load_dataset(args.data) if i.quality is not None]
    feats = [features.geometry(i.array, i.spacing, i.region, i.side) for i in ims]
    spine_idx = [k for k, i in enumerate(ims) if i.region == "spine" and LABEL_FOREIGN in i.labels]
    normals = [k for k in spine_idx if not ims[k].labels[LABEL_FOREIGN]]
    other = [k for k in range(len(ims)) if k not in normals]
    full = QualityModel().fit(feats, [i.region for i in ims], [i.labels for i in ims])
    patches = object_patches([ims[k] for k in spine_idx], full)
    contrasts = [p.max() for p in patches]
    print(f"настоящих предметов: {len(patches)}; нормальных снимков позвоночника: {len(normals)}", flush=True)

    res = {z: [] for z in ZONES}
    fp = []
    for half in (0, 1):
        train = [k for n, k in enumerate(normals) if n % 2 != half] + other
        test = [k for n, k in enumerate(normals) if n % 2 == half]
        m = QualityModel().fit([feats[k] for k in train], [ims[k].region for k in train], [ims[k].labels for k in train])
        fp += [m.predict(feats[k], "spine")[LABEL_FOREIGN][0] for k in test]
        rng = np.random.default_rng(100 + half)
        for zone in ZONES:
            n = 0
            while n < args.per_zone:
                im = ims[test[rng.integers(len(test))]]
                c = patches[rng.integers(len(patches))] if rng.random() < 0.67 else compact_patch(rng, contrasts)
                if rng.random() < 0.5:
                    c = c[:, ::-1]
                arr = place(im.array, c, zone, im.spacing, rng)
                if arr is None:
                    continue
                f = features.geometry(arr, im.spacing, "spine", None)
                res[zone].append(int(m.predict(f, "spine")[LABEL_FOREIGN][0]))
                n += 1
    report = dict(objects=len(patches), normals=len(normals),
                  false_alarms_on_normals=float(np.mean(fp)),
                  found={z: dict(rate=float(np.mean(v)), n=len(v)) for z, v in res.items()})
    print(f"ложные тревоги на чистых нормах (проверочные половины): {report['false_alarms_on_normals']:.0%}")
    for z, r in report["found"].items():
        print(f"найдено предметов {z:22s} {r['rate']:.0%} (n={r['n']})")
    Path("out").mkdir(exist_ok=True)
    json.dump(report, open("out/foreign_stand.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
