"""Grad-CAM utilities to visualise predictions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class GradCAMResult:
    heatmap: torch.Tensor
    logits: torch.Tensor
    probs: torch.Tensor


class GradCAM:
    """Simple Grad-CAM implementation using hooks."""

    def __init__(self, model: nn.Module, target_layer: str) -> None:
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        layer = dict(self.model.named_modules()).get(self.target_layer)
        if layer is None:
            raise ValueError(f"Layer '{self.target_layer}' not found in model.")

        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def __call__(self, image: torch.Tensor, class_idx: Optional[int] = None) -> GradCAMResult:
        image = image.unsqueeze(0)
        image.requires_grad_(True)
        logits = self.model(image)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture data.")
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        return GradCAMResult(heatmap=cam.cpu(), logits=logits.squeeze(0).detach(), probs=probs.detach())
