"""Пакетная обработка из командной строки.

    python -m dxa.predict --input /data/input --output /data/output/results.xlsx [--series /data/output/series]

--input  — каталог (любой вложенности) или zip с DICOM-файлами;
--output — .xlsx или .csv (второй формат пишется рядом автоматически);
--series — каталог для доп. серий (PNG, DICOM SC, DICOM SR); рядом создаётся series.zip;
--config — JSON с настройками (см. config.example.json), --mode — режим чувствительности.
Код возврата 0, даже если отдельные файлы не обработались: ошибки — в колонке error.
Код 2 — нет входа или неверные настройки.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from .pipeline import Service
from .settings import MODES, PARAMS, Settings

COLUMNS = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class", "quality_prob",
           "violation_type", "processing_status", "time_of_processing",
           "side", "projection", "details", "duplicate_of", "error", "settings"]


def save_table(rows: list[dict], output: Path, settings: Settings | None = None):
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df:
            df[c] = ""
    df = df[COLUMNS + [c for c in df.columns if c not in COLUMNS]]
    output.parent.mkdir(parents=True, exist_ok=True)
    xlsx, csv = output.with_suffix(".xlsx"), output.with_suffix(".csv")
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="results")
        ws = w.sheets["results"]
        for col, width in zip("ABCDEFGHIJKLMNO", (40, 30, 30, 30, 8, 8, 44, 10, 10, 9, 8, 90, 20, 30, 40)):
            ws.column_dimensions[col].width = width
        if settings is not None:  # с какими настройками получен результат — для воспроизводимости
            settings_frame(settings).to_excel(w, index=False, sheet_name="settings")
    if settings is not None:
        output.with_name("settings.json").write_text(json.dumps(settings.to_dict(), ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    return df


def settings_frame(st: Settings) -> pd.DataFrame:
    default = Settings()
    rows = [("mode", "Режим обучаемых порогов", st.mode, default.mode, "")]
    for k, (label, unit, *_ , tz) in PARAMS.items():
        rows.append((k, label + (f", {unit}" if unit else ""), getattr(st, k), getattr(default, k), "ТЗ" if tz else ""))
    rows.append(("pixel_spacing_mm", "Размер пикселя, мм", st.pixel_spacing_mm or "из тега", "из тега", ""))
    rows.append(("axis_scoliosis_check", "Наклон оси при дугообразном изгибе (сколиоз) — не нарушение укладки",
                 st.axis_scoliosis_check, default.axis_scoliosis_check, ""))
    rows += [(f"criteria.{k}", "Критерий включён", v, True, "") for k, v in st.criteria.items()]
    df = pd.DataFrame(rows, columns=["parameter", "description", "value", "default", "source"])
    df["changed"] = [str(v) != str(d) for v, d in zip(df["value"], df["default"])]
    return df


def main(argv=None):
    # на русской Windows консоль в cp1251 — символы вроде «→» роняют печать с UnicodeEncodeError;
    # переключаем вывод в utf-8 с заменой непечатаемых символов, чтобы процесс не падал в конце
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="DXA-QC: контроль качества денситометрии")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--series", default=None)
    ap.add_argument("--model", default="models/quality_model.joblib")
    ap.add_argument("--config", default=None, help="JSON с настройками (пороги, режим, критерии)")
    ap.add_argument("--mode", choices=MODES, default=None, help="режим чувствительности (перекрывает config)")
    args = ap.parse_args(argv)

    if not Path(args.input).exists():
        print(f"нет входа: {args.input}", file=sys.stderr)
        return 2
    try:
        settings = Settings.load(args.config)
        if args.mode:
            settings = Settings.from_dict({**settings.to_dict(), "mode": args.mode})
    except (ValueError, TypeError, OSError) as e:
        print(f"ошибка в настройках: {e}", file=sys.stderr)
        return 2

    t0 = time.time()
    service = Service(args.model, settings)
    print(f"модель загружена за {time.time() - t0:.1f} с", flush=True)
    rows = service.process(args.input, args.series,
                           progress=lambda i, n: print(f"  исследование {i}/{n}", flush=True))
    df = save_table(rows, Path(args.output), settings)
    if args.series and Path(args.series).exists():
        shutil.make_archive(str(Path(args.series)), "zip", args.series)
    ok = (df["processing_status"] == "Success").sum()
    print(f"готово: {len(df)} файлов, успешно {ok}, с нарушением {int((df['quality_class'] == 1).sum())}, "
          f"время {time.time() - t0:.1f} с → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
