import argparse, os, json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch

# Use hf_embed_global for CLIP embeddings
from scripts.hf_embed_global import embed_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="data/corpus/met_oil_paintings.parquet")
    parser.add_argument("--out-dir", default="data/embeddings_samples")
    parser.add_argument("--limit", type=int, default=20, help="Max number of images to embed")
    args = parser.parse_args()

    # 1) Load parquet file
    df = pd.read_parquet(args.parquet)
    
    # 2) Get local_path entries that exist (images should be in Drive)
    paths = []
    metadata_rows = []
    
    for idx, row in df.iterrows():
        local_path = row.get("local_path")
        if local_path and isinstance(local_path, str) and os.path.exists(local_path):
            paths.append(local_path)
            metadata_rows.append(row.to_dict())
        
        if len(paths) >= args.limit:
            break
    
    if not paths:
        print("ERROR: No valid images found. Check if local_path exists in parquet and files are in Drive.")
        return
    
    print(f"Loading {len(paths)} images from local paths...")
    
    # 3) Load images from local paths (in Google Drive)
    images = []
    valid_metadata = []
    
    for path, meta in zip(paths, metadata_rows):
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_metadata.append(meta)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue
    
    if not images:
        print("ERROR: No images could be loaded.")
        return
    
    # 4) Embed images
    mat = embed_images(images)  # (N,D) float32, L2-normalized

    # 5) Save embeddings and metadata
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", mat)
    
    # Save full metadata
    with open(out / "artwork_index.jsonl", "w") as f:
        for meta in valid_metadata:
            f.write(json.dumps(meta) + "\n")

    print(f"[done] embeddings shape = {mat.shape} -> {out/'embeddings.npy'}")
    print(f"[done] metadata saved to {out/'artwork_index.jsonl'}")

if __name__ == "__main__":
    main()
