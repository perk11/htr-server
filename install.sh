#!/usr/bin/env bash
# Установка в АКТИВНОЕ окружение conda (Python 3.12):
#   conda create -n htr python=3.12 -y && conda activate htr && ./install.sh
# Ставит torch с CUDA 13, nafrec (только ради детектора строк RF-DETR) и скачивает модели (~2,2 ГБ) в ./models.
set -euo pipefail
cd "$(dirname "$0")"

[ -n "${CONDA_PREFIX:-}" ] || { echo "Сначала: conda activate <окружение>"; exit 1; }
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || { echo "Нужен Python 3.12"; exit 1; }

python -m pip install pip==26.2.2
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1 torchvision==0.29.1
# nafrec[cpu] — onnxruntime нужен только для импорта пакета; детектор работает на torch и видеокарте
python -m pip install --uploaded-prior-to 2026-11-02 \
  "nafrec[cpu]==0.3.0" rfdetr==1.11.2 supervision==0.31.3 shapely==2.2.0 opencv-python-headless==5.0.0.93 \
  transformers==5.22.1 huggingface_hub==2.4.0 fastapi==0.162.0 uvicorn==0.54.0 httpx==0.28.1 \
  pillow==13.0.0 numpy==2.5.4

hf download Kansallisarkisto/cyrillic-htr-model --local-dir models/htr
hf download Kansallisarkisto/rfdetr-textline-textregion-detection-2xl --local-dir models/rfdetr
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
