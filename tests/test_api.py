from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
import torch

import api.main as api_module
from api.main import app, MODEL_PATH_ENV


class DummyPredictor:
    def __init__(self) -> None:
        self.class_names = ["paper", "plastic"]
        self.transforms = lambda image: {"image": torch.from_numpy(image).permute(2, 0, 1).float() / 255.0}  # type: ignore[arg-type]

    def predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor([0.2, 0.8])


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(MODEL_PATH_ENV, str(tmp_path / "model.pt"))
    app.dependency_overrides[api_module.get_predictor] = lambda: DummyPredictor()

    def fake_load_image(_: bytes) -> torch.Tensor:
        return torch.zeros((3, 32, 32), dtype=torch.float32)

    monkeypatch.setattr(api_module, "load_image", fake_load_image)
    client = TestClient(app)
    buffer = io.BytesIO()
    Image.fromarray((np.zeros((32, 32, 3), dtype=np.uint8))).save(buffer, format="PNG")
    buffer.seek(0)
    response = client.post(
        "/predict",
        files={"file": ("test.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["predicted_class"] == "plastic"
    app.dependency_overrides.clear()
