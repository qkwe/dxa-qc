# DXA-QC — контроль качества денситометрии. Работает полностью локально, без нейросетей и GPU.
#
#   docker build -t dxa-qc .
#
# Перед сборкой в каталоге должна быть models/quality_model.joblib (см. README: python train.py ...).
FROM python:3.10.16-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DXA_JOBS=/data/jobs \
    DXA_FEEDBACK=/data/feedback \
    DXA_INPUT_ROOTS=/data

# шрифт с кириллицей для подписей на доп. сериях; curl — для healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip==25.0.1 && pip install -r requirements.txt

COPY models ./models
COPY dxa ./dxa
COPY train.py evaluate_final.py validate_measures.py validate_foreign_object.py make_metrics_table.py verify.py ./
# честные метрики режимов чувствительности — показываются в настройках веб-интерфейса
COPY out/metrics_final.json out/validation.json out/foreign_stand.json ./out/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "dxa.api:app", "--host", "0.0.0.0", "--port", "8000"]
