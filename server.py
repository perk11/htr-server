"""HTR-сервер: строки находит RF-DETR из nafrec (Kansallisarkisto), читает буквально TrOCR Kansallisarkisto/cyrillic-htr-model.

POST /v1/htr  {"image": "<data URL | base64 | http(s) URL>",
               "crop": [x0, y0, x1, y1],  # доли 0..1, необязательно
               "mode": "page",            # page — найти строки детектором; line — картинка уже одна строка
               "rotate": 0,               # градусы против часовой (для наклонных помет)
               "n_best": 3, "num_beams": 4}
→ {"lines": [{"region": 0, "box": [x0, y0, x1, y1], "det_conf": 0.9, "text": "...", "alts": [...]}], "sec": 1.2}
Координаты box — в пикселях изображения после crop/rotate.
GET /health
"""
import base64, io, os, tempfile, time

import httpx
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps
from pydantic import BaseModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from nafrec.image_processing import crop_lines
from nafrec.seg_inference import load_rfdetr_model, predict_polygons
from nafrec.utils import get_default_region, get_line_regions, order_regions_lines

HTR_DIR = os.environ["HTR_MODEL_DIR"]
DET_PATH = os.environ["HTR_DET_MODEL"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
BATCH = int(os.environ.get("HTR_BATCH", "4"))

proc = TrOCRProcessor.from_pretrained(HTR_DIR, subfolder="processor")
model = VisionEncoderDecoderModel.from_pretrained(HTR_DIR, torch_dtype=DTYPE).to(DEVICE).eval()
det = load_rfdetr_model(DET_PATH, device=DEVICE)
app = FastAPI()


class Req(BaseModel):
    image: str
    crop: list[float] | None = None
    mode: str = "page"
    rotate: float = 0
    n_best: int = 3
    num_beams: int = 4


def load(src: str) -> Image.Image:
    if src.startswith(("http://", "https://")):
        r = httpx.get(src, timeout=60, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.content
    else:
        data = base64.b64decode(src.split(",", 1)[1] if src.startswith("data:") else src)
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


def detect(im: Image.Image):
    """Полигоны строк в порядке чтения: [(region, polygon, conf)]. Логика упорядочивания — как в nafrec.predict_single_image."""
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        im.save(f.name)
        lines, confs, mms, regs, reg_confs, reg_mms, shape = predict_polygons(det, f.name)
    if not lines:
        return []
    regions = [{"coords": p, "id": str(i), "max_min": mm, "name": "paragraph", "img_shape": shape, "conf": c}
               for i, (p, c, mm) in enumerate(zip(regs, reg_confs, reg_mms))] or get_default_region(image_shape=shape)
    connected = get_line_regions(lines={"coords": lines, "max_min": mms, "confs": confs}, regions=regions)
    out = []
    for ri, reg in enumerate(order_regions_lines(lines=connected, regions=regions)):
        out += [(ri, p, float(c)) for p, c in zip(reg["lines"], reg["line_confs"])]
    return out


@torch.inference_mode()
def recognize(crops: list[Image.Image], n_best: int, num_beams: int) -> list[list[str]]:
    n_best = max(1, min(n_best, num_beams))
    res = []
    for i in range(0, len(crops), BATCH):
        pv = proc(crops[i:i + BATCH], return_tensors="pt").pixel_values.to(DEVICE, DTYPE)
        seq = model.generate(pv, num_beams=num_beams, num_return_sequences=n_best, max_new_tokens=96)
        txt = proc.batch_decode(seq, skip_special_tokens=True)
        res += [txt[j:j + n_best] for j in range(0, len(txt), n_best)]
    return res


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "htr": HTR_DIR, "detector": DET_PATH}


@app.post("/v1/htr")
def htr(r: Req):
    t = time.time()
    try:
        im = load(r.image)
    except Exception as e:
        raise HTTPException(400, f"не удалось загрузить изображение: {e}")
    if r.crop:
        x0, y0, x1, y1 = r.crop
        im = im.crop((round(x0 * im.width), round(y0 * im.height), round(x1 * im.width), round(y1 * im.height)))
    if r.rotate:
        im = im.rotate(r.rotate, expand=True, fillcolor="white", resample=Image.Resampling.BICUBIC)

    if r.mode == "line":
        found = [(0, [[0, 0], [im.width, 0], [im.width, im.height], [0, im.height]], 1.0)]
        crops = [im]
    else:
        found = detect(im)
        crops = [Image.fromarray(c) for c in crop_lines([p for _, p, _ in found], np.asarray(im))]
    texts = recognize(crops, r.n_best, r.num_beams) if crops else []

    lines = []
    for (reg, poly, conf), t_ in zip(found, texts):
        xs = [int(p[0]) for p in poly]; ys = [int(p[1]) for p in poly]
        lines.append({"region": reg, "box": [min(xs), min(ys), max(xs), max(ys)], "det_conf": round(conf, 3),
                      "text": t_[0], "alts": t_[1:]})
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {"lines": lines, "sec": round(time.time() - t, 2)}
