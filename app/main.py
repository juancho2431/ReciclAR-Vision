"""Streamlit demo for ReciclAR Vision."""
from __future__ import annotations

import io
import os
from typing import List

import numpy as np
import streamlit as st
from PIL import Image
import torch

from src.models.explainability import GradCAM
from src.models.inference import Predictor


st.set_page_config(page_title="ReciclAR Vision", layout="wide")

MODEL_PATH_ENV = "RECICLAR_MODEL_PATH"
CLASS_NAMES_ENV = "RECICLAR_CLASS_NAMES"
DEFAULT_CLASSES = ["paper", "cardboard", "plastic", "glass", "metal", "other"]
TARGET_LAYER_ENV = "RECICLAR_TARGET_LAYER"


@st.cache_resource
def load_predictor() -> Predictor:
    model_path = os.getenv(MODEL_PATH_ENV)
    if not model_path:
        st.error("Variable de entorno RECICLAR_MODEL_PATH no configurada.")
        st.stop()
    predictor = Predictor(model_path)
    return predictor


def read_image(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:  # pragma: no cover - Streamlit handles display
        st.error(f"No se pudo leer la imagen: {exc}")
        st.stop()
    return image


def render_prediction(probabilities: torch.Tensor, class_names: List[str]) -> None:
    st.subheader("Resultados")
    confidences = probabilities.tolist()
    top_idx = int(np.argmax(confidences))
    st.metric("Categoría predicha", class_names[top_idx], f"{confidences[top_idx]*100:.2f}%")
    st.bar_chart({"confianza": confidences}, x=class_names)


def overlay_heatmap(image: Image.Image, heatmap: torch.Tensor) -> Image.Image:
    heatmap_np = heatmap.numpy()
    heatmap_resized = Image.fromarray((heatmap_np * 255).astype(np.uint8)).resize(image.size)
    heatmap_resized = heatmap_resized.convert("RGBA")
    base = image.convert("RGBA")
    return Image.blend(base, heatmap_resized, alpha=0.5)


def main() -> None:
    st.title("ReciclAR Vision")
    st.write("Sube una imagen de un residuo para recibir recomendaciones de reciclaje.")
    predictor = load_predictor()
    class_names = predictor.class_names or [name.strip() for name in os.getenv(CLASS_NAMES_ENV, "").split(",") if name.strip()] or DEFAULT_CLASSES

    uploaded = st.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg", "bmp", "webp"])
    if not uploaded:
        st.info("Esperando una imagen…")
        return
    image = read_image(uploaded.read())
    array = np.array(image)
    transformed = predictor.transforms(image=array)
    tensor = transformed["image"]
    probs = predictor.predict(tensor)
    render_prediction(probs, class_names)

    target_layer = os.getenv(TARGET_LAYER_ENV, "features.6")
    try:
        grad_cam = GradCAM(predictor.model, target_layer=target_layer)
        result = grad_cam(tensor)
        blended = overlay_heatmap(image, result.heatmap)
        st.image(blended, caption="Mapa Grad-CAM", use_column_width=True)
    except Exception as exc:
        st.warning(f"No se pudo generar Grad-CAM: {exc}")


if __name__ == "__main__":
    main()
