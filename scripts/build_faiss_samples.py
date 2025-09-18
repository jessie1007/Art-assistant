import os, numpy as np, faiss
import pathlib

EMB = "data/embeddings_samples/embeddings.npy"
OUT = pathlib.Path("data/index_samples")
OUT.mkdir(parents=True, exist_ok=True)

x = np.load(EMB).astype("float32")
faiss.normalize_L2(x)  # cosine via inner product
index = faiss.IndexFlatIP(x.shape[1])
index.add(x)
faiss.write_index(index, str(OUT / "index.faiss"))
print("[done] index size:", index.ntotal)
