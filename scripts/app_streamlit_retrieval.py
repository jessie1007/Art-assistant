import os, json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
import faiss

# ✅ value tools import at top (same folder)
from value_tools import (
    to_gray_np, posterize_k, flat_mid_gray,
    luminance_com, thirds_overlay, value_percentages
)

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
        st.image(img, caption="Query", width=800)


    model, proc, device = load_model()
    index, metas = load_index_and_meta(index_path, meta_path)

    q = embed_image(img, model, proc, device).reshape(1, -1)
    faiss.normalize_L2(q)
    D, I = index.search(q, topk)

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
                col.caption(f"{title}\n{artist} — {year}\nStyle: {style}\nScore: {float(score):.3f}")
            else:
                col.write("(missing file)")
        else:
            col.write("(invalid index)")


def render_value_tab(
    img,
    k_values: int = 5,
    show_grid: bool = True,
):
    """Render the Value Studies UI (no sidebars here; pass params from your launcher)."""
    if img is None:
        st.info("Upload an image above to analyze values.")
        return
    
    gray = to_gray_np(img)
    poster, centers = posterize_k(gray, k_values)
    dead = flat_mid_gray(gray)
    com = luminance_com(gray)
    img_overlay = thirds_overlay(img, com) if show_grid else img

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(
            img_overlay,
            caption="Original (+COM)" if show_grid else "Original",
            use_container_width=True
        )
    with c2:
        st.image(poster, caption=f"{k_values}-Value Posterization",
                 use_container_width=True, clamp=True)
        st.caption(f"Levels: {[int(v) for v in centers]}")
    with c3:
        st.image(dead, caption="Everything = mid gray (comparison)",
                 use_container_width=True, clamp=True)

    pct = value_percentages(poster, centers)
    st.subheader("Value coverage")
    st.write({f"band_{i}({int(centers[i])})": f"{100*pct[i]:.1f}%"
              for i in range(len(centers))})

    tips = []
    if len(pct) >= 3:
        dark, mid, light = pct[0], pct[len(pct)//2], pct[-1]
        if dark < 0.15: tips.append("Very few darks; add a few anchor accents for depth.")
        if mid < 0.4:   tips.append("Midtones are low; expand mid range to unify forms.")
        if light > 0.45: tips.append("Highlights dominate; reserve lights near the focal area.")
    x, y = com; thirds = [(1/3,1/3),(2/3,1/3),(1/3,2/3),(2/3,2/3)]
    dist = min(((x-a)**2 + (y-b)**2)**0.5 for a,b in thirds)
    tips.append("COM near a thirds hotspot—strong focal placement." if dist < 0.12
                else "Consider nudging value emphasis toward a thirds hotspot.")
    st.subheader("Suggestions")
    for t in tips[:2]:
        st.write("• " + t)

# =====================================================================
# Standalone runner (only when you launch this file directly)
# =====================================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Art Assistant", layout="wide")
    st.title("🎨 Art Assistant")

    # Local sidebar controls for standalone mode
    with st.sidebar:
        st.header("Settings")
        index_path = st.text_input("FAISS index path", "data/index_samples/index.faiss")
        meta_path  = st.text_input("Metadata JSONL path", "data/embeddings_samples/artwork_index.jsonl")
        topk = st.slider("Top-K", 1, 10, 5)
        k_values = st.slider("Value study — K (bands)", 3, 8, 5)
        show_grid = st.checkbox("Show thirds grid + COM dot", value=True)

    tab1, tab2 = st.tabs(["🔎 Retrieval", "🧪 Value Studies"])
    with tab1:
        render_retrieval_tab(index_path=index_path, meta_path=meta_path, topk=topk)
    with tab2:
        render_value_tab(k_values=k_values, show_grid=show_grid)
