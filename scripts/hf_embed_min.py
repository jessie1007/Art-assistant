import argparse, os, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch
import hashlib
# Add project root to path for Colab compatibility
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



# Resolve IMG_DIR relative to project root (works in Colab and local)
ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "images"

def _safe_name(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

def expected_local_path_from_url(url: str) -> str:
    return str((IMG_DIR / (_safe_name(url) + ".jpg")).as_posix())


# Use hf_embed_global for CLIP embeddings
try:
    from scripts.hf_embed_global import embed_images
except ImportError:
    # Fallback for Colab if scripts. doesn't work
    from hf_embed_global import embed_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="data/corpus/met_oil_paintings.parquet")
    parser.add_argument("--out-dir", default="data/embeddings_samples")
    parser.add_argument("--limit", type=int, default=20, help="Max number of images to embed")
    args = parser.parse_args()

    # 1) Load parquet file
    df = pd.read_parquet(args.parquet)
    
    # 2) Collect valid image paths (prefer existing local_path, fall back to derived path)
    paths = []
    metadata_rows = []
    
    for idx, row in df.iterrows():
        local_path = row.get("local_path")
        image_url = row.get("image_url")

        # Skip pandas NA / missing values safely
        if local_path is not None and pd.isna(local_path):
            local_path = None

        found_path = None
        
        # 1) Prefer an existing local_path on disk (resolve relative paths)
        if isinstance(local_path, str) and local_path.strip():
            # If relative path, resolve relative to ROOT
            if not os.path.isabs(local_path):
                resolved_path = ROOT / local_path
            else:
                resolved_path = Path(local_path)
            
            if resolved_path.exists():
                found_path = str(resolved_path)

        # 2) Otherwise, derive expected path from image_url and use it if it exists
        if not found_path and isinstance(image_url, str) and image_url.strip():
            derived = expected_local_path_from_url(image_url)
            if os.path.exists(derived):
                found_path = derived
        
        if found_path:
            meta = row.to_dict()
            meta["local_path"] = found_path  # Store absolute or resolved path
            paths.append(found_path)
            metadata_rows.append(meta)

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
