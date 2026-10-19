"""Проверка всего решения одной командой — для организаторов и экспертов.

    python verify.py --data /path/to/НД_для_обучения [--test /path/to/Для_теста] [--quick]
    ./run.sh check /path/to/НД_для_обучения [/path/to/Для_теста]        # то же в контейнере

С нуля, без кэшей и без участия команды:
  1. модель обучается заново и сверяется с поставленной (models/quality_model.json) — воспроизводимость;
  2. честные метрики: вложенная кросс-валидация по исследованиям, 95% ДИ (evaluate_final.py);
  3. поверка измерителя на снимках, повёрнутых/обрезанных на известную величину (validate_measures.py);
  4. стенд посторонних предметов на кости (validate_foreign_object.py); --quick — пропустить 3–4;
  5. пакетная обработка (тестовый фрагмент или 10 исследований обучающего набора) и проверка выдачи
     по ТЗ п. 2.5 и «Разъяснениям V2»: колонки, допустимые значения, разделитель «;», quality_prob в [0;1],
     время ≤ 3 мин на исследование, отсутствие необработанных ошибок.
Итог — out/verification.md (таблица «проверка — результат») и код возврата 0, если всё пройдено.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
REGIONS = {"Поясничный отдел позвоночника", "Проксимальный отдел бедра"}
VIOLATIONS = {"Поясничный отдел позвоночника": {"Некорректная укладка", "Не выравнена ось позвоночника",
                                                "Присутствуют посторонние предметы"},
              "Проксимальный отдел бедра": {"Некорректная укладка", "Некорректная область интереса"}}
COLUMNS = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
           "violation_type", "processing_status", "time_of_processing", "quality_prob"]


def run(args, log):
    t0 = time.time()
    r = subprocess.run([PY, "-W", "ignore", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    log.write(f"\n$ python {' '.join(args)}\n{r.stdout}{r.stderr}\n")
    return r, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="распакованный НД_для_обучения (DICOM + xlsx разметка)")
    ap.add_argument("--test", default=None, help="каталог/zip для пакетной проверки (по умолчанию — часть обучающего набора)")
    ap.add_argument("--quick", action="store_true", help="без поверки измерителя и стенда предметов")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    checks = []
    log = open(out / "verification.log", "w", encoding="utf-8")

    def check(name, ok, detail):
        checks.append((name, bool(ok), detail))
        print(f"[{'OK' if ok else 'НЕТ'}] {name}: {detail}", flush=True)

    # 1. воспроизводимость обучения
    shipped = json.loads((ROOT / "models" / "quality_model.json").read_text(encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp())
    r, sec = run(["train.py", "--data", args.data, "--out", str(tmp / "m.joblib"), "--cache", str(tmp / "geo.pkl")], log)
    if r.returncode == 0:
        fresh = json.loads((tmp / "m.json").read_text(encoding="utf-8"))
        keys = [k for k in shipped if k not in ("trained",)]
        diff = [k for k in keys if shipped.get(k) != fresh.get(k)]
        check("Модель воспроизводится обучением с нуля", not diff,
              "все параметры совпали с поставленной моделью" if not diff else f"отличаются: {diff}")
    else:
        check("Модель воспроизводится обучением с нуля", False, "обучение завершилось ошибкой — см. verification.log")
    shutil.rmtree(tmp, ignore_errors=True)

    # 2. честные метрики
    r, sec = run(["evaluate_final.py", "--data", args.data], log)
    ok = r.returncode == 0 and (out / "metrics_final.json").exists()
    if ok:
        m = json.loads((out / "metrics_final.json").read_text(encoding="utf-8"))
        allv = m["все снимки: качество (есть нарушение)"]
        check("Метрики (вложенная CV по исследованиям, 95% ДИ)", True,
              f"«есть нарушение» ROC-AUC {allv['auc']:.3f} [{allv['ci']['auc'][0]:.2f}; {allv['ci']['auc'][1]:.2f}], "
              f"F1 {allv['f1']:.3f} [{allv['ci']['f1'][0]:.2f}; {allv['ci']['f1'][1]:.2f}]; "
              f"quality_prob ROC-AUC {m['quality_prob ROC-AUC: все снимки']['auc']:.3f}; "
              f"macro-F1 {m['macro-F1 по типам нарушений']:.3f} → out/metrics_final.json")
        run(["make_metrics_table.py"], log)
    else:
        check("Метрики (вложенная CV по исследованиям, 95% ДИ)", False, "ошибка — см. verification.log")

    # 3–4. поверка измерителя и стенд предметов
    if not args.quick:
        r, sec = run(["validate_measures.py", "--data", args.data], log)
        if r.returncode == 0:
            v = json.loads((out / "validation.json").read_text(encoding="utf-8"))
            ax = next(x for k, x in v.items() if k.startswith("ось"))
            check("Поверка угла оси (повороты на известный угол)", ax["mean"] <= 3,
                  f"средняя ошибка {ax['mean']:.2f}° при допуске ТЗ 3° (n={ax['n']})")
            for k, x in v.items():
                if k.startswith("отступ"):
                    check(f"Поверка: {k}", x["mean"] <= 0.05, f"средняя ошибка {x['mean']:.1%} при допуске 5% (n={x['n']})")
        else:
            check("Поверка измерителя", False, "ошибка — см. verification.log")
        r, sec = run(["validate_foreign_object.py", "--data", args.data], log)
        if r.returncode == 0:
            s = json.loads((out / "foreign_stand.json").read_text(encoding="utf-8"))
            check("Стенд: посторонние предметы на кости (оценка)", True,
                  ", ".join(f"{z} {v['rate']:.0%}" for z, v in s["found"].items())
                  + f"; ложные тревоги на нормах {s['false_alarms_on_normals']:.0%}")
        else:
            check("Стенд посторонних предметов", False, "ошибка — см. verification.log")

    # 5. пакетная обработка и формат выдачи
    src = args.test
    tmp_in = None
    if not src:   # 10 исследований обучающего набора
        tmp_in = Path(tempfile.mkdtemp())
        studies = sorted(p for p in Path(args.data).rglob("*") if p.is_dir() and any(f.suffix.lower() == ".dcm" for f in p.iterdir()))[:30]
        for k, d in enumerate(studies):
            shutil.copytree(d, tmp_in / f"s{k:02d}")
        src = str(tmp_in)
    res = out / "verification_run"
    shutil.rmtree(res, ignore_errors=True)
    r, sec = run(["-m", "dxa.predict", "--input", src, "--output", str(res / "results.xlsx"), "--series", str(res / "series")], log)
    check("Пакетная обработка завершилась без необработанных исключений", r.returncode == 0 and "Traceback" not in r.stderr,
          f"код {r.returncode}, {sec:.0f} с")
    if (res / "results.csv").exists():
        import pandas as pd

        df = pd.read_csv(res / "results.csv", dtype=str, keep_default_na=False)
        miss = [c for c in COLUMNS if c not in df.columns]
        check("Колонки по ТЗ п. 2.5 + quality_prob (Разъяснения V2)", not miss, "все есть" if not miss else f"нет: {miss}")
        ok_rows = df[df["processing_status"] == "Success"]
        bad_region = set(ok_rows["anatomical_region"]) - REGIONS
        check("anatomical_region — только значения из Разъяснений V2", not bad_region,
              f"значения: {sorted(set(ok_rows['anatomical_region']))}")
        bad_v = []
        for _, row in ok_rows.iterrows():
            for v in filter(None, row["violation_type"].split(";")):
                if v not in VIOLATIONS.get(row["anatomical_region"], set()) or v != v.strip():
                    bad_v.append(v)
        check("violation_type — значения из списка, разделитель «;»", not bad_v,
              "все значения допустимые" if not bad_v else f"недопустимые: {sorted(set(bad_v))[:5]}")
        cls_ok = set(ok_rows["quality_class"]) <= {"0", "1"} and all((v == "") == (c == "0") for v, c in
                                                                    zip(ok_rows["violation_type"], ok_rows["quality_class"]))
        check("quality_class 0/1 и согласован с violation_type", cls_ok, "да" if cls_ok else "есть расхождения")
        probs = pd.to_numeric(ok_rows["quality_prob"], errors="coerce")
        check("quality_prob в [0;1]", probs.notna().all() and probs.between(0, 1).all(),
              f"от {probs.min():.3f} до {probs.max():.3f}")
        st = set(df["processing_status"])
        check("processing_status — Success/Failure", st <= {"Success", "Failure"}, f"{sorted(st)}")
        t = pd.to_numeric(df["time_of_processing"], errors="coerce")
        per_study = t.groupby(df["study_uid"]).sum().max()
        check("Время на исследование ≤ 3 мин (ТЗ 2.7)", per_study <= 180, f"максимум {per_study:.2f} с на исследование")
        n_sr = len(list((res / "series").rglob("*_SR.dcm"))) if (res / "series").exists() else 0
        check("Доп. серии: разметка (PNG, DICOM SC) и заключение DICOM SR", n_sr > 0, f"SR: {n_sr}")
    if tmp_in:
        shutil.rmtree(tmp_in, ignore_errors=True)

    passed = all(ok for _, ok, _ in checks)
    lines = ["# Проверка решения DXA-QC", "", f"Дата: {time.strftime('%Y-%m-%d %H:%M')}. Данные: `{args.data}`.", "",
             "| Проверка | Результат | Подробности |", "|---|---|---|"]
    lines += [f"| {n} | {'✅' if ok else '❌'} | {d} |" for n, ok, d in checks]
    lines += ["", f"**Итог: {'все проверки пройдены' if passed else 'есть непройденные проверки'}.** "
              "Подробный журнал — out/verification.log, таблица метрик — out/metrics_table.md."]
    (out / "verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + ("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ" if passed else "ЕСТЬ НЕПРОЙДЕННЫЕ ПРОВЕРКИ") + " → out/verification.md")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
