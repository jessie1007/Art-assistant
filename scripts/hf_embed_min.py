# Minimal: load CLIP, embed up to 20 images from a parquet file, save embeddings.npy
import argparse, os, json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="data/corpus/samples20.parquet")
    parser.add_argument("--out-dir", default="data/embeddings_samples")
    args = parser.parse_args()

    # 1) load file list
    # read mega data of those file images
    df = pd.read_parquet(args.parquet)
    paths = [p for p in df["local_path"].tolist() if isinstance(p, str) and os.path.exists(p)]
    paths = paths[:20]  # cap at 20

    # 2) load model + processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 3) load images and preprocess (one small batch)
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = proc(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 4) forward pass → normalized embeddings
    with torch.inference_mode():
        feats = model.get_image_features(**inputs)           # [N, D]
        feats = torch.nn.functional.normalize(feats, dim=-1) # cosine-ready
    mat = feats.cpu().numpy().astype("float32")              # (N, 512)

    # 5) save
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", mat)
    with open(out / "artwork_index.jsonl", "w") as f:
        for p in paths: f.write(json.dumps({"local_path": p}) + "\n")

    print(f"[done] embeddings shape = {mat.shape} -> {out/'embeddings.npy'}")

if __name__ == "__main__":
    main()
