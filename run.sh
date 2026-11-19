#!/usr/bin/env bash
# Запуск HTR-сервера (после ./install.sh в окружении conda):
#   ./run.sh [host] [port]        — по умолчанию 127.0.0.1 18245
# Окружение conda активировать не обязательно: берётся $HTR_PYTHON, иначе python активного окружения, иначе ~/.conda/envs/htr.
# CUDA_VISIBLE_DEVICES — какую видеокарту взять, HTR_BATCH — строк за проход (8; строки сортируются по ширине).
set -euo pipefail
cd "$(dirname "$0")"

PY="${HTR_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
PY="${PY:-$HOME/.conda/envs/htr/bin/python}"

# Строки разной длины дробят кэш видеопамяти PyTorch; расширяемые сегменты держат его компактным.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HTR_MODEL_DIR="$PWD/models/htr" HTR_DET_MODEL="$PWD/models/rfdetr/checkpoint_best_total.pth"
exec "$PY" -m uvicorn server:app --host "${1:-127.0.0.1}" --port "${2:-18245}"
