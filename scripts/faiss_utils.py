from pathlib import Path
from typing import Union, Dict, Any
import numpy as np
import faiss

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "data/index_samples/index.faiss"
DEFAULT_META  = ROOT / "data/index_samples/meta.npy"

def load_index(path: Union[Path, str] = DEFAULT_INDEX):
    """Load a FAISS index from disk."""
    return faiss.read_index(str(path))

# Keep backward-compatible name if your app imports this:
def load_faiss_index(path: Union[Path, str] = DEFAULT_INDEX):
    return load_index(path)

def load_meta(path: Union[Path, str] = DEFAULT_META) -> Dict[int, dict]:
    """Load id->metadata dict saved with np.save(..., allow_pickle=True)."""
    return np.load(str(path), allow_pickle=True).item()

# Keep backward-compatible helper:
def id_to_meta(idx: int, meta_path: Union[Path, str] = DEFAULT_META) -> dict:
    meta = load_meta(meta_path)
    return meta.get(idx, {})

def search_index(index, q_vec: np.ndarray, topk: int = 5):
    """Search top-k nearest by inner product/cosine. q_vec is 1D float32."""
    q = np.asarray(q_vec, dtype="float32")[None, :]
    scores, ids = index.search(q, topk)
    return ids[0].tolist(), scores[0].tolist()
