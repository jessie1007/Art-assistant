# Complete Colab Pipeline: Download Images & Generate All Artifacts

Copy and paste this into Google Colab cells to download images and generate all required files.

## Setup: Mount Drive & Install Dependencies

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Navigate to project directory (or clone if needed)
import os
from pathlib import Path

# Option A: If you already have the repo in Drive
PROJECT_DIR = Path("/content/drive/MyDrive/Art-assistant")
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT_DIR)

# Option B: Clone from GitHub (if you want fresh copy)
# !git clone https://github.com/jessie1007/Art-assistant.git /content/drive/MyDrive/Art-assistant
# os.chdir("/content/drive/MyDrive/Art-assistant")

print(f"Working directory: {os.getcwd()}")
```

```python
# Cell 3: Install dependencies
!pip install -q torch torchvision transformers pillow pandas numpy faiss-cpu requests tqdm pyarrow open-clip-torch
```

## Step 1: Download Images from Met Museum

```python
# Cell 4: Download images (adjust --limit as needed)
# This will download images and create met_oil_paintings.parquet

!python scripts/download_met_images.py \
  --limit 5000 \
  --thumb \
  --download-missing \
  --longest-side 512 \
  --delay 0.1 \
  --min-year 1500 \
  --max-year 2000

# Check results
import pandas as pd
from pathlib import Path

parquet_path = Path("data/corpus/met_oil_paintings.parquet")
if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
    print(f"✅ Downloaded {len(df)} artworks")
    print(f"   Images saved to: data/images/")
    print(f"   Parquet file: {parquet_path}")
else:
    print("❌ Parquet file not found")
```

**Parameters explained:**
- `--limit 5000`: Download up to 5000 artworks (adjust as needed)
- `--thumb`: Use thumbnail images (faster, smaller)
- `--download-missing`: Download images that don't exist yet
- `--longest-side 512`: Resize images (512px max dimension)
- `--delay 0.1`: Wait 0.1 seconds between requests (be nice to API)
- `--min-year 1500 --max-year 2000`: Filter by date range

## Step 2: Generate Embeddings

```python
# Cell 5: Generate CLIP embeddings for all downloaded images
# This creates embeddings.npy and artwork_index.jsonl

import sys
from pathlib import Path

# Add project to path
ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import embedding function
try:
    from scripts.hf_embed_global import embed_images
except ImportError:
    print("⚠️  scripts/hf_embed_global.py not found. Using fallback...")
    # Fallback: install and use transformers directly
    from transformers import CLIPProcessor, CLIPModel
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def embed_images(images):
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=-1)
        return features.cpu().numpy().astype("float32")
```

```python
# Cell 6: Load images and generate embeddings
import pandas as pd
import numpy as np
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Load parquet
df = pd.read_parquet("data/corpus/met_oil_paintings.parquet")
print(f"Loading {len(df)} images...")

# Collect valid images
images = []
metadata = []
failed = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
    local_path = row.get("local_path")
    if local_path and Path(local_path).exists():
        try:
            img = Image.open(local_path).convert("RGB")
            images.append(img)
            metadata.append(row.to_dict())
        except Exception as e:
            failed += 1
            continue
    else:
        failed += 1

print(f"✅ Loaded {len(images)} images ({failed} failed)")

if len(images) == 0:
    print("❌ No images found! Check data/images/ directory")
else:
    # Generate embeddings in batches (to avoid memory issues)
    print("Generating embeddings...")
    batch_size = 32
    all_embeddings = []
    
    for i in tqdm(range(0, len(images), batch_size), desc="Embedding batches"):
        batch = images[i:i+batch_size]
        batch_emb = embed_images(batch)
        all_embeddings.append(batch_emb)
    
    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    print(f"✅ Embeddings shape: {embeddings.shape}")
    
    # Save embeddings
    out_dir = Path("data/embeddings_samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "embeddings.npy", embeddings)
    print(f"✅ Saved embeddings: {out_dir / 'embeddings.npy'}")
    
    # Save metadata as JSONL (one JSON object per line)
    with open(out_dir / "artwork_index.jsonl", "w", encoding="utf-8") as f:
        for meta in metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    
    print(f"✅ Saved metadata: {out_dir / 'artwork_index.jsonl'} ({len(metadata)} entries)")
```

## Step 3: Build FAISS Index

```python
# Cell 7: Build FAISS index from embeddings
import numpy as np
import faiss
import json
from pathlib import Path

# Paths
emb_path = Path("data/embeddings_samples/embeddings.npy")
meta_path = Path("data/embeddings_samples/artwork_index.jsonl")
out_dir = Path("data/index_samples")
out_dir.mkdir(parents=True, exist_ok=True)

# Load embeddings
print("Loading embeddings...")
x = np.load(emb_path).astype("float32")
print(f"✅ Loaded embeddings: {x.shape}")

# Normalize for cosine similarity
faiss.normalize_L2(x)

# Build FAISS index
print("Building FAISS index...")
index = faiss.IndexFlatIP(x.shape[1])  # Inner product = cosine (after L2 normalization)
index.add(x)
print(f"✅ Index built: {index.ntotal} vectors")

# Save index
index_path = out_dir / "index.faiss"
faiss.write_index(index, str(index_path))
print(f"✅ Saved index: {index_path}")

# Load and save metadata as .npy
print("Loading metadata...")
meta_dict = {}
with open(meta_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        meta_dict[i] = json.loads(line)

meta_npy_path = out_dir / "meta.npy"
np.save(meta_npy_path, meta_dict, allow_pickle=True)
print(f"✅ Saved metadata: {meta_npy_path} ({len(meta_dict)} entries)")

# Verify alignment
if len(meta_dict) == index.ntotal:
    print("✅ Metadata and index are aligned!")
else:
    print(f"⚠️  Warning: Metadata ({len(meta_dict)}) != Index ({index.ntotal})")
```

## Step 4: Verify All Files

```python
# Cell 8: Verify all files are created and check sizes
from pathlib import Path
import os

files_to_check = [
    "data/corpus/met_oil_paintings.parquet",
    "data/embeddings_samples/embeddings.npy",
    "data/embeddings_samples/artwork_index.jsonl",
    "data/index_samples/index.faiss",
    "data/index_samples/meta.npy",
]

print("=== File Verification ===\n")
for file_path in files_to_check:
    path = Path(file_path)
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"✅ {file_path}")
        print(f"   Size: {size_mb:.2f} MB")
        
        # Special checks
        if file_path.endswith(".npy"):
            import numpy as np
            arr = np.load(path, allow_pickle=True)
            if isinstance(arr, np.ndarray):
                print(f"   Shape: {arr.shape}")
            else:
                print(f"   Type: dict with {len(arr)} entries")
        elif file_path.endswith(".jsonl"):
            with open(path) as f:
                lines = sum(1 for _ in f)
            print(f"   Entries: {lines}")
        elif file_path.endswith(".faiss"):
            import faiss
            idx = faiss.read_index(str(path))
            print(f"   Vectors: {idx.ntotal}")
    else:
        print(f"❌ {file_path} - NOT FOUND")
    print()
```

## Step 5: Download Files to Local (Optional)

```python
# Cell 9: Download all files to your local machine
from google.colab import files
from pathlib import Path

files_to_download = [
    "data/corpus/met_oil_paintings.parquet",
    "data/embeddings_samples/embeddings.npy",
    "data/embeddings_samples/artwork_index.jsonl",
    "data/index_samples/index.faiss",
    "data/index_samples/meta.npy",
]

print("Downloading files...")
for file_path in files_to_download:
    path = Path(file_path)
    if path.exists():
        print(f"📥 Downloading {path.name}...")
        files.download(str(path))
    else:
        print(f"⚠️  {path} not found, skipping")
```

## Complete All-in-One Script (Alternative)

If you prefer to run everything at once:

```python
# Complete pipeline in one cell (may take a while)
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import faiss
from PIL import Image
from tqdm import tqdm

# Setup
from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = Path("/content/drive/MyDrive/Art-assistant")
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT_DIR)

# Install deps
!pip install -q torch torchvision transformers pillow pandas numpy faiss-cpu requests tqdm pyarrow open-clip-torch

# Step 1: Download images
print("=" * 50)
print("STEP 1: Downloading images...")
print("=" * 50)
!python scripts/download_met_images.py --limit 5000 --thumb --download-missing --longest-side 512 --delay 0.1

# Step 2: Generate embeddings
print("\n" + "=" * 50)
print("STEP 2: Generating embeddings...")
print("=" * 50)

# Import embedding function
ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hf_embed_global import embed_images

# Load and embed
df = pd.read_parquet("data/corpus/met_oil_paintings.parquet")
images = []
metadata = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    local_path = row.get("local_path")
    if local_path and Path(local_path).exists():
        try:
            img = Image.open(local_path).convert("RGB")
            images.append(img)
            metadata.append(row.to_dict())
        except:
            continue

print(f"Loaded {len(images)} images")

# Embed in batches
batch_size = 32
all_embeddings = []
for i in tqdm(range(0, len(images), batch_size)):
    batch = images[i:i+batch_size]
    batch_emb = embed_images(batch)
    all_embeddings.append(batch_emb)

embeddings = np.vstack(all_embeddings)

# Save embeddings
out_dir = Path("data/embeddings_samples")
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "embeddings.npy", embeddings)

with open(out_dir / "artwork_index.jsonl", "w") as f:
    for meta in metadata:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

print(f"✅ Saved embeddings: {embeddings.shape}")

# Step 3: Build FAISS index
print("\n" + "=" * 50)
print("STEP 3: Building FAISS index...")
print("=" * 50)

x = embeddings.astype("float32")
faiss.normalize_L2(x)

index = faiss.IndexFlatIP(x.shape[1])
index.add(x)

out_dir = Path("data/index_samples")
out_dir.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(out_dir / "index.faiss"))

meta_dict = {}
with open("data/embeddings_samples/artwork_index.jsonl", "r") as f:
    for i, line in enumerate(f):
        meta_dict[i] = json.loads(line)

np.save(out_dir / "meta.npy", meta_dict, allow_pickle=True)

print(f"✅ Index built: {index.ntotal} vectors")
print("\n" + "=" * 50)
print("✅ ALL DONE!")
print("=" * 50)
```

## Tips

1. **Adjust `--limit`**: Start with 1000 to test, then increase to 5000-10000
2. **Monitor progress**: Each step prints progress bars
3. **Check file sizes**: After completion, verify files are large (MBs, not KBs)
4. **Memory**: If you get OOM errors, reduce batch_size in embedding step
5. **Time**: Full pipeline for 5000 images takes ~30-60 minutes

## Expected File Sizes (for 5000 images)

- `met_oil_paintings.parquet`: ~5-10 MB
- `embeddings.npy`: ~10-20 MB (5000 × 512 × 4 bytes)
- `artwork_index.jsonl`: ~5-10 MB
- `index.faiss`: ~10-20 MB
- `meta.npy`: ~5-10 MB

If your files are much smaller, you didn't process all images!

