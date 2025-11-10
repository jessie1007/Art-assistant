
from pathlib import Path
import os, numpy as np, faiss
import pathlib
import os, json, numpy as np, faiss

# NEW: resolve paths relative to the repo root (parent of /scripts)
ROOT    = Path(__file__).resolve().parents[1]    # <--- key change
EMB     = ROOT / "data/embeddings_samples/embeddings.npy"         # <--- was "data/..."
META_IN = ROOT / "data/embeddings_samples/artwork_index.jsonl"    # <--- was "data/..."
OUT     = ROOT / "data/index_samples"                              # <--- was "data/..."
OUT.mkdir(parents=True, exist_ok=True)

# sanity checks (helpful messages instead of silent failures)
if not EMB.exists():
    raise FileNotFoundError(f"Missing embeddings: {EMB}")
if not META_IN.exists():
    raise FileNotFoundError(f"Missing JSONL: {META_IN}")

# 1) load embeddings
x = np.load(EMB).astype("float32")
faiss.normalize_L2(x)  # cosine via inner product

# 2) build FAISS index
index = faiss.IndexFlatIP(x.shape[1])
index.add(x)
faiss.write_index(index, str(OUT / "index.faiss"))
print("[done] index size:", index.ntotal)

# 3) also save metadata (row index -> info from jsonl)

meta = {}
with open(META_IN) as f:
    for i, line in enumerate(f):
        meta[i] = json.loads(line)

np.save(OUT / "meta.npy", meta, allow_pickle=True)
print("[done] saved metadata:", OUT / "meta.npy")