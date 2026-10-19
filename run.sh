#!/usr/bin/env bash
# Сборка и запуск DXA-QC в контейнере (Linux / UNIX-подобные системы).
#
#   ./run.sh build                                    # собрать образ
#   ./run.sh predict <вход> <выход> [настройки.json]  # пакетная обработка → <выход>/results.xlsx|csv, series.zip
#   ./run.sh api [настройки.json]                     # веб-интерфейс и API на http://localhost:8000 (порт: PORT=...)
# Настройки (режим чувствительности, пороги, критерии) — см. config.example.json; без них — как в ТЗ.
set -euo pipefail

IMAGE="${IMAGE:-dxa-qc}"
PORT="${PORT:-8000}"   # порт веб-интерфейса и API: PORT=8080 ./run.sh api
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

check_assets() {
    [[ -f "${DIR}/models/quality_model.joblib" ]] || { echo "нет models/quality_model.joblib — см. README (train.py)"; exit 1; }
}

abspath() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }

check_config() {
    [[ -f "$1" ]] || { echo "нет файла настроек: $1"; exit 1; }
}

predict() {
    local input="$1" output="$2" config="${3:-}"
    [[ -e "${input}" ]] || { echo "нет входа: ${input}"; exit 1; }
    mkdir -p "${output}"
    input="$(abspath "${input}")"
    output="$(cd "${output}" && pwd)"
    local args=(-v "${input}:/data/input:ro" -v "${output}:/data/output")
    local cmd=(python -m dxa.predict --input /data/input --output /data/output/results.xlsx --series /data/output/series)
    if [[ -n "${config}" ]]; then
        check_config "${config}"
        args+=(-v "$(abspath "${config}"):/data/config.json:ro")
        cmd+=(--config /data/config.json)
    fi
    docker run --rm "${args[@]}" "${IMAGE}" "${cmd[@]}"
}

api() {
    local config="${1:-}"
    local args=(-p "${PORT}:8000")
    if [[ -n "${config}" ]]; then
        check_config "${config}"
        args+=(-v "$(abspath "${config}"):/data/config.json:ro" -e DXA_CONFIG=/data/config.json)
    fi
    docker run --rm "${args[@]}" "${IMAGE}"
}

case "${1:-}" in
    build)   check_assets; docker build -t "${IMAGE}" "${DIR}" ;;
    predict) predict "${2:?вход}" "${3:?выход}" "${4:-}" ;;
    api)     api "${2:-}" ;;
    *) echo "Использование: $0 {build|predict <вход> <выход> [настройки.json]|api [настройки.json]}"; exit 1 ;;
esac
