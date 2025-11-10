# Minimal: load CLIP, embed up to 20 images from a parquet file, save embeddings.npy
from hf_embed_global import embed_images
import argparse, os, json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

#from clip_core import embed_images  # <<< new import

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="data/corpus/samples20.parquet")
    parser.add_argument("--out-dir", default="data/embeddings_samples")
    parser.add_argument("--limit", type=int, default=20, help="Max number of images to embed")
    args = parser.parse_args()

    # 1) load file list
    # read mega data of those file images
    df = pd.read_parquet(args.parquet)
    paths = [p for p in df["local_path"].tolist() if isinstance(p, str) and os.path.exists(p)]
    paths = paths[: args.limit]

    # 3) load images and preprocess (one small batch)
    images = [Image.open(p).convert("RGB") for p in paths]
    mat = embed_images(images)  # (N,D) float32, L2-normalized

    # 4) save
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", mat)
    with open(out / "artwork_index.jsonl", "w") as f:
        for p in paths: f.write(json.dumps({"local_path": p}) + "\n")

    print(f"[done] embeddings shape = {mat.shape} -> {out/'embeddings.npy'}")

if __name__ == "__main__":
    main()
