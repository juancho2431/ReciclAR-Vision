"""FastAPI service exposing the waste classification model."""
from __future__ import annotations

import io
import os
from functools import lru_cache
from typing import List

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch

from src.models.inference import Predictor


MODEL_PATH_ENV = "RECICLAR_MODEL_PATH"
CLASS_NAMES_ENV = "RECICLAR_CLASS_NAMES"
DEFAULT_CLASSES = ["paper", "cardboard", "plastic", "glass", "metal", "other"]


app = FastAPI(title="ReciclAR Vision API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_image(data: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # pragma: no cover - PIL raises many errors
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    array = np.array(image)
    predictor = get_predictor()
    transformed = predictor.transforms(image=array)
    return transformed["image"]


def _read_class_names() -> List[str]:
    env_value = os.getenv(CLASS_NAMES_ENV)
    if env_value:
        return [name.strip() for name in env_value.split(",") if name.strip()]
    return DEFAULT_CLASSES


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    model_path = os.getenv(MODEL_PATH_ENV)
    if not model_path:
        raise RuntimeError(
            "Environment variable RECICLAR_MODEL_PATH not set. Provide the path to the trained model."
        )
    return Predictor(model_path)


@app.post("/predict")
async def predict(file: UploadFile = File(...), predictor: Predictor = Depends(get_predictor)):
    data = await file.read()
    image_tensor = load_image(data)
    probs = predictor.predict(image_tensor)
    class_names = predictor.class_names or _read_class_names()
    top_idx = int(torch.argmax(probs).item())
    return {
        "predicted_class": class_names[top_idx] if top_idx < len(class_names) else str(top_idx),
        "confidence": float(probs[top_idx].item()),
        "probabilities": {
            class_names[i] if i < len(class_names) else str(i): float(prob)
            for i, prob in enumerate(probs.tolist())
        },
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
