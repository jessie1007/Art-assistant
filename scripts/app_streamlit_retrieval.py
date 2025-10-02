import os, json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
import faiss

# ---------- Cache: model ----------
@st.cache_resource
def load_model(model_id: str = "openai/clip-vit-base-patch32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    proc  = CLIPProcessor.from_pretrained(model_id)
    return model, proc, device

# ---------- Cache: FAISS + metadata ----------
@st.cache_resource
def load_index_and_meta(index_path: str, meta_path: str):
    """Load FAISS index and metadata. Supports .npy (dict) and .jsonl (list of dicts)."""
    index = faiss.read_index(str(index_path))

    p = Path(meta_path)
    if not p.exists():
        raise FileNotFoundError(f"Metadata not found: {p}")

    if p.suffix.lower() == ".npy":
        # meta.npy: saved via np.save(..., allow_pickle=True), a dict {id: {...}}
        meta_dict = np.load(str(p), allow_pickle=True).item()
        # Convert to list for metas[id] usage; fill missing ids with {}
        metas = [meta_dict.get(i, {}) for i in range(index.ntotal)]
    else:
        # Treat as JSONL text
        with open(p, "r", encoding="utf-8") as f:
            metas = [json.loads(line) for line in f]

    if len(metas) != index.ntotal:
        st.warning(f"Metadata rows ({len(metas)}) != index vectors ({index.ntotal}). Check build alignment.")
    return index, metas

def embed_image(img: Image.Image, model, proc, device) -> np.ndarray:
    img = img.convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        v = model.get_image_features(**inputs)
        v = torch.nn.functional.normalize(v, dim=-1)
    return v.squeeze(0).cpu().numpy().astype("float32")

# =====================================================================
# PUBLIC API: functions you can import in the main launcher tabs
# =====================================================================

def render_retrieval_tab(
    img,  # <-- shared image comes from the launcher
    index_path: str = "data/index_samples/index.faiss",
    meta_path: str  = "data/embeddings_samples/artwork_index.jsonl",
    topk: int = 5,
):
    if img is None:
        st.info("Upload an image above to see similar results.")
        return

    st.subheader("Retrieval results")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(img, caption="Current photos", width=900)


    model, proc, device = load_model()
    index, metas = load_index_and_meta(index_path, meta_path)

    # Embed + search
    q = embed_image(img, model, proc, device).reshape(1, -1)
    faiss.normalize_L2(q)  # cosine via inner product
    D, I = index.search(q, topk)

    # Render results
    st.subheader(f"Top {topk} similar images")
    cols = st.columns(topk)
    for col, i, score in zip(cols, I[0], D[0]):
        if 0 <= i < len(metas):
            m = metas[i]
            path   = m.get("local_path")
            title  = m.get("title")  or "Untitled"
            artist = m.get("artist") or "Unknown"
            year   = m.get("year")   or m.get("objectDate") or "—"
            style  = m.get("style")  or "—"
            if path and os.path.exists(path):
                col.image(path, use_container_width=True)
                cap = f"{title}\n{artist} — {year}\nStyle: {style}\nScore: {float(score):.3f}"
                col.caption(cap)
            else:
                col.write("(missing file)")
        else:
            col.write("(invalid index)")

