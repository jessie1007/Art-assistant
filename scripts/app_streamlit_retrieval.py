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
def load_index_and_meta(
    index_path: str = "data/index_samples/index.faiss",
    meta_path: str  = "data/embeddings_samples/artwork_index.jsonl",
):
    index = faiss.read_index(index_path)
    metas = [json.loads(l) for l in open(meta_path, "r", encoding="utf-8")]
    assert index.ntotal == len(metas), "Index size != metadata row count"
    return index, metas

def embed_image(img: Image.Image, model, proc, device) -> np.ndarray:
    img = img.convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        v = model.get_image_features(**inputs)
        v = torch.nn.functional.normalize(v, dim=-1)
    return v.squeeze(0).cpu().numpy().astype("float32")

# ---------- UI ----------
st.set_page_config(page_title="Art Assistant — Retrieval", layout="wide")
st.title("🎨 Similar References (MVP)")

# Optional: tweak paths / top-k from sidebar
with st.sidebar:
    st.header("Settings")
    index_path = st.text_input("FAISS index path", "data/index_samples/index.faiss")
    meta_path  = st.text_input("Metadata JSONL path", "data/embeddings_samples/artwork_index.jsonl")
    topk = st.slider("Top-K", 1, 10, 5)

uploaded = st.file_uploader("Upload a painting (jpg/png)", type=["jpg","jpeg","png"])

if uploaded:
    # Show query
    query = Image.open(uploaded)
    st.image(query, caption="Query", use_container_width=True)

    # Load heavy resources once
    model, proc, device = load_model()
    index, metas = load_index_and_meta(index_path, meta_path)

    # Embed + search
    q = embed_image(query, model, proc, device).reshape(1, -1)
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
else:
    st.info("Upload an image to see similar results.")
