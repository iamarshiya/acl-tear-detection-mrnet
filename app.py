"""KneeScan AI — FastAPI backend.

Run:  uvicorn app:app --port 8000      then open http://localhost:8000
"""
import os

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from kneescan.inference import Engine, load_volume
from kneescan.model_def import VIEWS

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.environ.get("KNEESCAN_DEPLOY", os.path.join(HERE, "deploy"))

app = FastAPI(title="KneeScan AI", version="1.0")
engine = Engine(DEPLOY)
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/info")
def info():
    avail = set(engine.available_models())
    models = []
    for name, m in engine.cfg["models"].items():
        models.append({
            "name": name,
            "available": name in avail,
            "threshold": m.get("threshold"),
            "params_m": m.get("params_m"),
            "train_time_s": m.get("train_time_s"),
            "peak_vram_mb": m.get("peak_vram_mb"),
            "test_metrics": m.get("test_metrics", {}),
        })
    return {"models": models, "samples": engine.cfg.get("samples", []),
            "device": str(engine.dev), "versions": engine.cfg.get("versions", {})}


def _run(model, volumes):
    if model not in engine.available_models():
        raise HTTPException(400, f"Model '{model}' has no checkpoints in deploy/ckpt")
    try:
        return engine.predict(model, volumes)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/predict")
async def predict(model: str = Form(...),
                  axial: UploadFile | None = File(None),
                  coronal: UploadFile | None = File(None),
                  sagittal: UploadFile | None = File(None)):
    volumes = {}
    for v, f in zip(VIEWS, (axial, coronal, sagittal)):
        if f is not None and f.filename:
            try:
                volumes[v] = load_volume(await f.read())
            except Exception as e:
                raise HTTPException(400, f"{v}: could not read .npy volume ({e})")
    if not volumes:
        raise HTTPException(400, "Upload at least one plane (.npy)")
    return _run(model, volumes)


@app.post("/api/predict_sample")
def predict_sample(model: str = Form(...), sample_id: str = Form(...)):
    ids = {s["id"]: s for s in engine.cfg.get("samples", [])}
    if sample_id not in ids:
        raise HTTPException(404, "Unknown sample")
    folder = os.path.join(DEPLOY, "samples", sample_id)
    volumes = {v: load_volume(os.path.join(folder, f"{v}.npy")) for v in VIEWS
               if os.path.exists(os.path.join(folder, f"{v}.npy"))}
    out = _run(model, volumes)
    out["ground_truth"] = ids[sample_id]["label"]
    return out
