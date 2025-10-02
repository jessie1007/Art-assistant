
# tests/test_core_min.py
import os, sys, pathlib
import numpy as np
from PIL import Image

# --- make project root importable ---
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- import from core/ or scripts/ (whichever you have) ---
def _import_any():
    try:
        from core.clip_core import embed_image
        from core.rerank_tools import hue_hist, texture_vec, combine_scores
        from core.feedback_tools import make_feedback
        return embed_image, hue_hist, texture_vec, combine_scores, make_feedback
    except Exception:
        from scripts.clip_core import embed_image
        from scripts.rerank_tools import hue_hist, texture_vec, combine_scores
        from scripts.feedback_tools import make_feedback
        return embed_image, hue_hist, texture_vec, combine_scores, make_feedback

embed_image, hue_hist, texture_vec, combine_scores, make_feedback = _import_any()

# ------------------- Tests -------------------

def test_clip_embed_shape_and_norm():
    img = Image.new("RGB", (80, 80), color="white")
    vec = embed_image(img)
    assert vec.shape[0] >= 256, "Expected an embedding length (e.g., 512)"
    assert np.isfinite(vec).all()
    # unit-ish norm (allow tiny floating error)
    n = np.linalg.norm(vec)
    assert 0.99 <= n <= 1.01, f"Expected normalized vector, got norm={n:.3f}"

def test_faiss_self_match():
    import faiss
    D = 128
    rng = np.random.default_rng(42)
    X = rng.normal(size=(4, D)).astype("float32")
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(D)
    index.add(X)
    Dists, Ids = index.search(X[:1], 1)
    assert Ids[0,0] == 0, "Nearest neighbor of a vector to itself should be itself"
    assert Dists[0,0] <= 1.0001

def test_rerank_combines_reasonably():
    # fabricate a query + 3 candidates with different cue alignments
    import cv2
    q_bgr = np.zeros((64,64,3), dtype=np.uint8)
    q_bgr[:] = (30, 120, 200)  # some color for hue hist
    q_h = hue_hist(q_bgr)
    q_t = np.zeros(12, dtype="float32"); q_t[3] = 1.0  # oriented texture (fake)
    q_feats = {"h_hist": q_h, "t_vec": q_t, "sat_mean": 0.5, "grad_mag_mean": 0.5}

    # three candidates with made-up sims (already normalized scores)
    cands = [
        {"clip_sim": 0.80, "h_hist": q_h,           "t_vec": q_t},           # matches all
        {"clip_sim": 0.85, "h_hist": np.roll(q_h,1),"t_vec": np.roll(q_t,1)},# high clip, weaker palette/tex
        {"clip_sim": 0.60, "h_hist": q_h,           "t_vec": q_t},           # lower clip, strong palette/tex
    ]
    scores = combine_scores(q_feats, cands)
    assert len(scores) == 3 and np.isfinite(scores).all()
    # best should be either the all-match or the high-clip one
    best = int(np.argmax(scores))
    assert best in (0,1)

def test_feedback_schema_and_ranges():
    img = Image.new("RGB", (96, 96), color=(200, 120, 30))
    fb = make_feedback(np.array(img))
    for k in ["value", "composition", "color"]:
        assert k in fb, f"Missing section: {k}"
    v = fb["value"]
    assert 0.0 <= v["mean"] <= 1.0
    assert 0.0 <= v["contrast"] <= 1.0
    assert 0.0 <= v["dark_pct"] <= 1.0
    assert 0.0 <= v["light_pct"] <= 1.0

def test_tiny_e2e_on_dummy_vectors(tmp_path):
    """Simulate: index 5 vectors -> query first -> nearest=0; then run a minimal 'rerank' list."""
    import faiss
    rng = np.random.default_rng(0)
    D = 64
    X = rng.normal(size=(5, D)).astype("float32")
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(D)
    index.add(X)
    q = X[0:1]
    Dists, Ids = index.search(q, 3)
    assert Ids.shape == (1,3) and Ids[0,0] == 0
