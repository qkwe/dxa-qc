"""Инструмент врача: разметка (подтверждение/исправление вердиктов, обведённые предметы) и дообучение.

Отдельный инструмент поверх сервиса: страница /review и API /api/feedback, /api/retrain, /api/models.

Дообучение безопасно по устройству:
  * новая модель — только кандидат (models/candidates/), сервис продолжает работать на прежней;
  * для кандидата считаются честные метрики той же вложенной CV на наборе «организатор + разметка врачей»,
    рядом — метрики действующей модели и её согласие с врачами на размеченных снимках;
  * применить кандидата — явное действие пользователя; прежняя модель уходит в models/history/,
    откат — одной кнопкой.
Набор организатора для дообучения: переменная DXA_TRAIN_DATA (в контейнере — смонтированный каталог).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from . import feedback, landmarks

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("DXA_MODELS_DIR", "models"))
TRAIN_DATA = os.environ.get("DXA_TRAIN_DATA")
router = APIRouter()
_retrain: dict = {"status": "idle"}
_lock = threading.Lock()


def _api():
    from . import api   # круговой импорт: api подключает этот модуль
    return api


@router.get("/review", response_class=HTMLResponse)
def review_page():
    return (Path(__file__).parent / "static" / "review.html").read_text(encoding="utf-8")


def _source_bytes(job: dict, row: dict) -> bytes:
    """Исходный DICOM строки задания: файл из загрузки, член zip-архива или файл каталога сервера."""
    from .pipeline import _iter_sources

    src = job.get("src")
    if not src:
        raise HTTPException(409, "у задания не сохранён путь к входным данным (задание создано старой версией)")
    for name, load in _iter_sources(src):
        if name == row["path_to_study"]:
            return load()
    raise HTTPException(404, "исходный файл задания не найден (удалён?)")


@router.get("/api/jobs/{job_id}/plain/{k}.png")
def plain_image(job_id: str, k: int):
    """Снимок без легенды и разметки — для обводки предметов (легенда закрывает верх кадра). Тот же кадр,
    что под картинкой разметки (тот же пересчёт к ~0.6 мм, ×3), поэтому рамки совпадают."""
    import io

    import pydicom
    from fastapi.responses import Response
    from PIL import Image as PImage

    from . import axis
    from .pipeline import Item, Service, _decode
    from .settings import Settings

    job = _api().job_status_dict(job_id)
    rows = job.get("rows") or []
    if not 0 <= k < len(rows) or rows[k].get("processing_status") != "Success":
        raise HTTPException(404, "нет такого обработанного снимка")
    ds = pydicom.dcmread(io.BytesIO(_source_bytes(job, rows[k])), force=True)
    it = Item(path=rows[k]["path_to_study"], ds=ds, array=_decode(ds))
    Service._to_native(it, Service._spacing(it, Settings.from_dict(job.get("settings"))))
    a = axis.normalize(it.array)
    img = PImage.fromarray((a * 255).astype("uint8")).resize((a.shape[1] * 3, a.shape[0] * 3))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@router.post("/api/feedback")
def add_feedback(body: dict):
    """{job, row, labels: {метка: 0/1}, boxes: [[x0,y0,x1,y1]] в пикселях картинки разметки,
    overlay_size: [ширина, высота] этой картинки, note, rotation_type: over/under/unknown — только бедро}."""
    api = _api()
    job = api.job_status_dict(str(body.get("job", "")))
    rows = job.get("rows") or []
    k = body.get("row")
    if not isinstance(k, int) or not 0 <= k < len(rows):
        raise HTTPException(422, "нет такой строки задания")
    row = rows[k]
    if row.get("processing_status") != "Success":
        raise HTTPException(422, "снимок не обработан — размечать нечего")
    region = {"Поясничный отдел позвоночника": "spine", "Проксимальный отдел бедра": "femur"}.get(row["anatomical_region"])
    side = {"левое": "l", "правое": "r"}.get(row.get("side") or "")
    data = _source_bytes(job, row)
    # рамки нарисованы на картинке разметки (снимок ×3, возможно пересчитанный к 0.6 мм) — в пиксели DICOM
    boxes = body.get("boxes") or []
    if boxes:
        import io

        import numpy as np
        import pydicom

        arr = np.asarray(pydicom.dcmread(io.BytesIO(data), force=True).pixel_array)
        rows_px, cols_px = arr.shape[:2] if arr.ndim == 2 or arr.shape[-1] in (3, 4) else arr.shape[1:3]
        w, h = (body.get("overlay_size") or [0, 0])[:2]
        if not (w and h):
            raise HTTPException(422, "нужен overlay_size — размер картинки, на которой нарисованы рамки")
        sx, sy = cols_px / float(w), rows_px / float(h)
        boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]
    try:
        entry_id = feedback.save(data, region, side, body.get("labels") or {}, boxes, body.get("note", ""),
                                 source=dict(job=job["id"], row=k, path=row["path_to_study"],
                                             service_verdict=row.get("violation_type", "")),
                                 rotation_type=body.get("rotation_type") or None)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"id": entry_id, "total": len(feedback.entries())}


def _job_row(body: dict) -> tuple[dict, int, dict]:
    job = _api().job_status_dict(str(body.get("job", "")))
    rows = job.get("rows") or []
    k = body.get("row")
    if not isinstance(k, int) or not 0 <= k < len(rows):
        raise HTTPException(422, "нет такой строки задания")
    return job, k, rows[k]


@router.post("/api/landmarks")
def set_landmark(body: dict):
    """Отметка малого вертела: {job, row, state: point | not_visible | clear, point: [x, y] в долях кадра}.
    Точка сервиса и размер кадра берутся из строки задания — сравнение честное, клиент их не задаёт."""
    _, _, row = _job_row(body)
    if row.get("anatomical_region") != "Проксимальный отдел бедра" or row.get("processing_status") != "Success":
        raise HTTPException(422, "малый вертел отмечается на обработанном снимке бедра")
    state = body.get("state")
    if state == "clear":
        landmarks.delete(row["study_uid"], row["image_uid"])
        return {"summary": landmarks.summary()}
    service, frame = None, None
    if row.get("overlay_lt_peak"):
        fx, fy, w_mm, h_mm = json.loads(row["overlay_lt_peak"])
        service, frame = [fx, fy], [w_mm, h_mm]
    try:
        e = landmarks.save(row["study_uid"], row["image_uid"], state, body.get("point"), service, frame,
                           path=row.get("path_to_study", ""))
    except (ValueError, TypeError) as err:
        raise HTTPException(422, str(err))
    return {"entry": e, "summary": landmarks.summary()}


@router.get("/api/landmarks")
def list_landmarks():
    return {"items": landmarks.load(), "summary": landmarks.summary()}


@router.get("/api/feedback")
def list_feedback():
    items = feedback.entries()
    return {"total": len(items), "items": items}


@router.delete("/api/feedback/{entry_id}")
def delete_feedback(entry_id: str):
    if not feedback.delete(entry_id):
        raise HTTPException(404, "нет такой записи")
    return {"deleted": entry_id, "total": len(feedback.entries())}


# ---------------------------------------------------------------- дообучение
def _agreement(model_path: Path) -> dict:
    """Согласие модели с врачами на размеченных ими снимках (по каждому критерию и в целом)."""
    import joblib

    from . import features
    from .data import LABEL_FOREIGN

    images = feedback.load_images(feedback.FEEDBACK)
    if not images:
        return {}
    model = joblib.load(model_path)["quality"]
    agree, total, crit = 0, 0, {}
    found, boxed = 0, 0
    for im in images:
        out = model.predict(features.geometry(im.array, im.spacing, im.region, im.side), im.region)
        for label, (flag, _, _) in out.items():
            if label in im.labels:
                ok = int(flag) == int(im.labels[label])
                agree += ok; total += 1
                c = crit.setdefault(f"{im.region}: {label}", [0, 0]); c[0] += ok; c[1] += 1
        if im.region == "spine" and im.labels.get(LABEL_FOREIGN):
            boxed += 1; found += int(out.get(LABEL_FOREIGN, (0,))[0])
    return dict(images=len(images), agreement=agree / total if total else None,
                by_criterion={k: v[0] / v[1] for k, v in crit.items()},
                foreign_found=f"{found} из {boxed}" if boxed else None)


def _run_retrain(stamp: str):
    cand = MODELS / "candidates" / stamp
    cand.mkdir(parents=True, exist_ok=True)
    log = open(cand / "log.txt", "w", encoding="utf-8")
    steps = [("Обучение на наборе организатора и разметке врачей",
              ["train.py", "--data", TRAIN_DATA, "--feedback", str(feedback.FEEDBACK), "--out", str(cand / "quality_model.joblib"),
               "--cache", str(MODELS / "candidates" / "train_geo.pkl")]),
             ("Честная оценка кандидата (вложенная CV по исследованиям)",
              ["evaluate_final.py", "--data", TRAIN_DATA, "--feedback", str(feedback.FEEDBACK), "--out", str(cand / "metrics_final.json")])]
    try:
        for title, args in steps:
            with _lock:
                if _retrain.get("cancel"):
                    raise RuntimeError("остановлено пользователем")
                _retrain["step"] = title
                proc = subprocess.Popen([sys.executable, "-W", "ignore", *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                _retrain["proc"] = proc
            if proc.wait() != 0:
                if _retrain.get("cancel"):
                    raise RuntimeError("остановлено пользователем")
                raise RuntimeError(f"шаг «{title}» завершился ошибкой — см. {cand / 'log.txt'}")
        _retrain["step"] = "Согласие с врачами на размеченных снимках"
        agree = dict(current=_agreement(MODELS / "quality_model.joblib"), candidate=_agreement(cand / "quality_model.joblib"))
        (cand / "agreement.json").write_text(json.dumps(agree, ensure_ascii=False, indent=1), encoding="utf-8")
        _retrain.update(status="done", candidate=stamp, finished=time.strftime("%H:%M:%S"))
    except Exception as e:
        _retrain.update(status="cancelled" if "остановлено" in str(e) else "error", error=str(e))
        shutil.rmtree(cand, ignore_errors=True) if _retrain["status"] == "cancelled" else None
    finally:
        _retrain.pop("proc", None)
        log.close()


@router.post("/api/retrain")
def start_retrain():
    if not TRAIN_DATA or not Path(TRAIN_DATA).exists():
        raise HTTPException(409, "для дообучения нужен набор организатора: задайте DXA_TRAIN_DATA "
                                 "(в контейнере — смонтируйте каталог НД_для_обучения)")
    if not feedback.entries():
        raise HTTPException(409, "разметки врача пока нет — сначала разметьте хотя бы один снимок")
    with _lock:
        if _retrain.get("status") == "running":
            raise HTTPException(409, "дообучение уже идёт")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        _retrain.clear()
        _retrain.update(status="running", started=time.strftime("%H:%M:%S"), stamp=stamp, step="подготовка")
    threading.Thread(target=_run_retrain, args=(stamp,), daemon=True).start()
    return {"status": "running", "stamp": stamp}


@router.post("/api/retrain/cancel")
def cancel_retrain():
    with _lock:
        if _retrain.get("status") != "running":
            return {"status": _retrain.get("status")}
        _retrain["cancel"] = True
        proc = _retrain.get("proc")
        if proc and proc.poll() is None:
            proc.terminate()
    return {"status": "cancelling"}


def _metrics_summary(path: Path) -> dict | None:
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    out = {k: dict(auc=v.get("auc"), f1=v.get("f1"), ci=v.get("ci"), n=v.get("n"), positives=v.get("positives"))
           for k, v in m.items() if isinstance(v, dict) and "f1" in v and "ci" in v}
    q = m.get("quality_prob ROC-AUC: все снимки")
    if q:
        out["quality_prob ROC-AUC"] = dict(auc=q["auc"], ci=q["ci"])
    out["macro-F1"] = m.get("macro-F1 по типам нарушений")
    out["images"] = m.get("снимков в оценке")
    return out


@router.get("/api/retrain")
def retrain_status():
    st = {k: v for k, v in _retrain.items() if k not in ("proc", "cancel")}
    st["train_data"] = bool(TRAIN_DATA and Path(TRAIN_DATA).exists())
    st["feedback"] = len(feedback.entries())
    st["current"] = _metrics_summary(_api().METRICS)
    if st.get("status") == "done" and st.get("candidate"):
        cand = MODELS / "candidates" / st["candidate"]
        st["candidate_metrics"] = _metrics_summary(cand / "metrics_final.json")
        try:
            st["agreement"] = json.loads((cand / "agreement.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st["agreement"] = None
    return st


# ---------------------------------------------------------------- версии модели
def _backup_current(reason: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    h = MODELS / "history" / stamp
    h.mkdir(parents=True, exist_ok=True)
    for f in (MODELS / "quality_model.joblib", MODELS / "quality_model.json"):
        if f.exists():
            shutil.copy2(f, h / f.name)
    if _api().METRICS.exists():
        shutil.copy2(_api().METRICS, h / "metrics_final.json")
    (h / "reason.txt").write_text(reason, encoding="utf-8")
    return stamp


def _activate(src: Path):
    shutil.copy2(src / "quality_model.joblib", MODELS / "quality_model.joblib")
    if (src / "quality_model.json").exists():
        shutil.copy2(src / "quality_model.json", MODELS / "quality_model.json")
    if (src / "metrics_final.json").exists():
        _api().METRICS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "metrics_final.json", _api().METRICS)
    _api().reload_service()


@router.post("/api/models/accept")
def accept_candidate(body: dict):
    stamp = Path(str(body.get("candidate", ""))).name
    cand = MODELS / "candidates" / stamp
    with _lock:
        # во время дообучения статус и файлы кандидата меняются: применение затёрло бы статус (и «Остановить»
        # перестало бы работать), а недообученный кандидат без честной оценки попал бы в работу
        if _retrain.get("status") == "running":
            raise HTTPException(409, "идёт дообучение — дождитесь окончания или остановите его")
        if not stamp or not (cand / "quality_model.joblib").exists():
            raise HTTPException(404, "нет такого кандидата")
        if not (cand / "agreement.json").exists():
            raise HTTPException(409, "кандидат не прошёл оценку до конца — применять его нельзя")
        backup = _backup_current(f"заменена кандидатом {stamp}")
        _activate(cand)
        _retrain.update(status="accepted", accepted=stamp)
    return {"active": stamp, "backup": backup}


@router.post("/api/models/rollback")
def rollback():
    with _lock:
        if _retrain.get("status") == "running":
            raise HTTPException(409, "идёт дообучение — дождитесь окончания или остановите его")
        hist = sorted((MODELS / "history").glob("*/quality_model.joblib")) if (MODELS / "history").exists() else []
        if not hist:
            raise HTTPException(404, "нет сохранённых версий — откатывать некуда")
        last = hist[-1].parent
        _activate(last)
        shutil.rmtree(last)          # возвращённая версия снова действующая
        _retrain.update(status="rolled_back")
    return {"restored": last.name}


@router.get("/api/models")
def models_info():
    def meta(p: Path):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    hist = sorted((MODELS / "history").glob("*")) if (MODELS / "history").exists() else []
    return dict(active=meta(MODELS / "quality_model.json"),
                history=[dict(stamp=h.name, reason=(h / "reason.txt").read_text(encoding="utf-8") if (h / "reason.txt").exists() else "")
                         for h in hist])
