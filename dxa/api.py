"""HTTP API и веб-интерфейс (ТЗ п. 3.2 «API для пакетной обработки», п. 2.6 «веб-интерфейс»).

    uvicorn dxa.api:app --host 0.0.0.0 --port 8000

GET  /                         — веб-интерфейс
GET  /health                   — проверка готовности
GET  /api/settings             — настройки по умолчанию, пределы и честные метрики режимов
GET  /metrics, /api/metrics    — страница и данные метрик: CV с 95% ДИ, режимы, поверка, стенд предметов
GET  /review                   — инструмент врача: подтверждение/исправление вердиктов, обведение предметов,
                                 дообучение (dxa/review.py: /api/feedback, /api/retrain, /api/models)
POST /api/jobs                 — загрузка zip / DICOM-файлов (form-data: files, необязательно settings — JSON)
POST /api/jobs/path            — обработка каталога на сервере: {"path": "/data/input", "settings": {...}}
GET  /api/jobs/{id}            — статус и строки результата
POST /api/jobs/{id}/cancel     — остановить обработку (уже обработанное сохраняется в отчёт)
GET  /api/jobs/{id}/results.xlsx | results.csv | series.zip | settings.json
GET  /api/jobs/{id}/overlay/{path} — PNG с разметкой

Переменные окружения: DXA_MODEL — модель; DXA_CONFIG — JSON настроек по умолчанию для сервиса;
DXA_JOBS — каталог заданий; DXA_INPUT_ROOTS — каталоги, которые разрешено обрабатывать
через /api/jobs/path (через «:»; если не задано — любые).
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .pipeline import Cancelled, Service
from .predict import save_table
from .settings import Settings, describe

JOBS = Path(os.environ.get("DXA_JOBS", "out/jobs"))
MODEL = os.environ.get("DXA_MODEL", "models/quality_model.joblib")
CONFIG = os.environ.get("DXA_CONFIG")
INPUT_ROOTS = [Path(p).resolve() for p in os.environ.get("DXA_INPUT_ROOTS", "").split(os.pathsep) if p]
METRICS = Path(os.environ.get("DXA_METRICS", "out/metrics_final.json"))
app = FastAPI(title="DXA-QC", version="1.1")
_service: Service | None = None
_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def service() -> Service:
    global _service
    with _lock:
        if _service is None:
            _service = Service(MODEL, Settings.load(CONFIG))
    return _service


def reload_service():
    """После принятия новой модели (инструмент врача): следующее задание загрузит её заново."""
    global _service
    with _lock:
        _service = None


def job_status_dict(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if not job:
        saved = JOBS / Path(job_id).name / "job.json"
        if not saved.exists():
            raise HTTPException(404, "нет такого задания")
        job = json.loads(saved.read_text(encoding="utf-8"))
    return job


def _settings(raw) -> Settings:
    """Настройки задания поверх настроек сервиса; ошибка в настройках — 422 с понятным текстом."""
    try:
        if isinstance(raw, str):
            raw = json.loads(raw) if raw.strip() else {}
        if not isinstance(raw, dict):
            raise ValueError("настройки должны быть JSON-объектом")
        base = service().settings.to_dict()
        base.update(raw)
        return Settings.from_dict(base)
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"ошибка в настройках: {e}")


def _save_job(job_id: str):
    """Состояние задания на диск: ссылка ?job=<id> работает и после перезапуска сервиса."""
    job = _jobs[job_id]
    (JOBS / job_id).mkdir(parents=True, exist_ok=True)
    (JOBS / job_id / "job.json").write_text(json.dumps({k: v for k, v in job.items() if k != "trace"},
                                                       ensure_ascii=False, default=str), encoding="utf-8")


def _run(job_id: str, src: Path, settings: Settings):
    job = _jobs[job_id]
    out = JOBS / job_id
    try:
        t0 = time.time()
        rows = service().process(src, out / "series", progress=lambda i, n: job.update(progress=[i, n]),
                                 settings=settings, should_stop=lambda: bool(job.get("cancel")),
                                 on_read=lambda n: job.update(read=n))
        save_table(rows, out / "results.xlsx", settings)
        if (out / "series").exists():
            shutil.make_archive(str(out / "series"), "zip", out / "series")
        job.update(status="done", rows=rows, seconds=round(time.time() - t0, 1))
    except Cancelled as c:
        # остановлено пользователем: отчёт по уже обработанным исследованиям
        rows = getattr(c, "rows", [])
        if rows:
            save_table(rows, out / "results.xlsx", settings)
            if (out / "series").exists():
                shutil.make_archive(str(out / "series"), "zip", out / "series")
        job.update(status="cancelled", rows=rows, seconds=round(time.time() - t0, 1))
    except Exception as e:
        job.update(status="error", error=f"{e}", trace=traceback.format_exc(limit=5))
    job.pop("cancel", None)
    _save_job(job_id)


def _start(src: Path, job_id: str, settings: Settings) -> dict:
    _jobs[job_id] = dict(id=job_id, status="running", progress=[0, 0], created=time.time(),
                         settings=settings.to_dict(), src=str(src))
    threading.Thread(target=_run, args=(job_id, src, settings), daemon=True).start()
    return {"id": job_id, "status": "running"}


def _upload_path(root: Path, filename: str | None) -> Path:
    """Путь для загруженного файла. Подкаталоги из имени сохраняются (загрузка папки), «..» и
    абсолютные пути отбрасываются; одноимённые файлы не затирают друг друга (IM0001 в разных
    исследованиях — обычное дело для выгрузок PACS)."""
    parts = [p for p in PurePosixPath((filename or "file.dcm").replace("\\", "/")).parts
             if p not in ("", ".", "..", "/") and ":" not in p]
    path = root.joinpath(*(parts or ["file.dcm"]))
    stem, suffix, n = path.stem, path.suffix, 1
    while path.exists():
        path = path.with_name(f"{stem}_{n}{suffix}")
        n += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/metrics", response_class=HTMLResponse)
def metrics_page():
    return (Path(__file__).parent / "static" / "metrics.html").read_text(encoding="utf-8")


@app.get("/api/metrics")
def metrics():
    """Честные метрики (out/metrics_final.json), поверка измерителя и стенд посторонних предметов."""
    out = {}
    for key, name in (("metrics", METRICS.name), ("validation", "validation.json"), ("foreign_stand", "foreign_stand.json")):
        try:
            out[key] = json.loads((METRICS.parent / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out[key] = None
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "model_loaded": _service is not None}


@app.get("/api/settings")
def get_settings():
    """Умолчания и пределы для интерфейса + честные (кросс-валидация) метрики каждого режима."""
    info = describe()
    info["service"] = service().settings.to_dict()
    try:
        info["modes_metrics"] = json.loads(METRICS.read_text(encoding="utf-8")).get("режимы")
    except (OSError, ValueError):
        info["modes_metrics"] = None
    return info


@app.post("/api/jobs")
async def create_job(files: list[UploadFile] = File(...), settings: str = Form("")):
    st = _settings(settings)
    job_id = uuid.uuid4().hex[:12]
    src = JOBS / job_id / "input"
    src.mkdir(parents=True, exist_ok=True)
    for f in files:
        # на диск по частям: большой архив не держим в памяти целиком
        with open(_upload_path(src, f.filename), "wb") as out:
            shutil.copyfileobj(f.file, out, length=1 << 20)
    return _start(src, job_id, st)


@app.post("/api/jobs/path")
def create_job_path(body: dict):
    path = Path(body.get("path", "")).resolve()
    if INPUT_ROOTS and not any(path == r or r in path.parents for r in INPUT_ROOTS):
        raise HTTPException(403, f"каталог вне разрешённых (DXA_INPUT_ROOTS): {path}")
    if not path.exists():
        raise HTTPException(404, f"нет такого пути: {path}")
    st = _settings(body.get("settings") or {})
    return _start(path, uuid.uuid4().hex[:12], st)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Остановить обработку: текущий файл/исследование дорабатывается, остальное — нет."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "нет такого задания")
    if job["status"] == "running":
        job["cancel"] = True
        job["status_note"] = "останавливается..."
    return {"id": job_id, "status": job["status"]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = job_status_dict(job_id)
    return JSONResponse({k: v for k, v in job.items() if k not in ("trace", "cancel", "src")})


@app.get("/api/jobs/{job_id}/{name}")
def job_file(job_id: str, name: str):
    if name not in ("results.xlsx", "results.csv", "series.zip", "settings.json"):
        raise HTTPException(404)
    path = JOBS / Path(job_id).name / name
    if not path.exists():
        raise HTTPException(404, "файл ещё не готов")
    return FileResponse(path, filename=name)


@app.get("/api/jobs/{job_id}/overlay/{rel:path}")
def overlay(job_id: str, rel: str):
    base = (JOBS / Path(job_id).name / "series").resolve()
    path = (base / rel).resolve()
    if base not in path.parents or not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


# инструмент врача (отдельный модуль): разметка, дообучение, версии модели
from . import review  # noqa: E402

app.include_router(review.router)
