from typing import List, Tuple
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class ZeroShotCLIP:
    def __init__(self, device="cpu", model_name="openai/clip-vit-base-patch32"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def predict_topk(self, pil_image: Image.Image, labels: List[str], k=3) -> List[Tuple[str, float]]:
        prompts = [f"a painting in the style of {lbl}" for lbl in labels]
        batch = self.proc(text=prompts, images=pil_image, return_tensors="pt", padding=True).to(self.device)
        out = self.model(**batch)
        probs = out.logits_per_image.softmax(dim=-1).squeeze(0)
        topk = torch.topk(probs, k=min(k, len(labels)))
        return [(labels[i], float(probs[i])) for i in topk.indices]
