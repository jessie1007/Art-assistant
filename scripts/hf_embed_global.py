from __future__ import annotations
from functools import lru_cache
from typing import List, Dict, Iterable
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

@lru_cache(maxsize=1)
def _load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Using device: {device}")
    if device == "cuda":
        print(f"[info] GPU: {torch.cuda.get_device_name(0)}")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    proc  = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, proc, device

def embed_image(img: Image.Image) -> np.ndarray:
    """Embed one PIL image -> (D,) float32, L2-normalized."""
    model, proc, device = _load_models()
    inputs = proc(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        feats = model.get_image_features(**inputs)  # [1, D]
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats[0].detach().cpu().numpy().astype("float32")

def embed_images(imgs: List[Image.Image]) -> np.ndarray:
    """Embed a batch of PIL images -> (N,D) float32, L2-normalized."""
    if not imgs:
        return np.zeros((0, 512), dtype="float32")
    model, proc, device = _load_models()
    inputs = proc(images=imgs, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        feats = model.get_image_features(**inputs)  # [N, D]
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.detach().cpu().numpy().astype("float32")
