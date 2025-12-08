import os, json, sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
import faiss

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import API client
try:
    from app.api_client import ArtAssistantAPIClient
    API_CLIENT_AVAILABLE = True
except ImportError:
    API_CLIENT_AVAILABLE = False
    st.warning("⚠️ API client not available. Using direct FAISS mode.")

# ---------- Cache: model (for direct FAISS mode) ----------
@st.cache_resource
def load_model(model_id: str = "openai/clip-vit-base-patch32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    proc  = CLIPProcessor.from_pretrained(model_id)
    return model, proc, device

# ---------- Cache: FAISS + metadata (for direct FAISS mode) ----------
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
    api_url: str = None,  # If provided, use API mode
    use_api: bool = True,  # Toggle between API and direct FAISS
):
    """
    Render retrieval tab with support for both API and direct FAISS modes.
    
    Args:
        img: PIL Image to search
        index_path: Path to FAISS index (for direct mode)
        meta_path: Path to metadata (for direct mode)
        topk: Number of results to return
        api_url: FastAPI server URL (if None, uses env var or localhost:8000)
        use_api: If True, use API mode; if False, use direct FAISS mode
    """
    if img is None:
        st.info("Upload an image above to see similar results.")
        return

    st.subheader("Retrieval results")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(img, caption="Current image", width=900)

    # Choose mode: API or direct FAISS
    if use_api and API_CLIENT_AVAILABLE:
        # ========== API MODE ==========
        try:
            client = ArtAssistantAPIClient(api_url=api_url)
            
            # Check API health
            with st.spinner("Checking API connection..."):
                if not client.is_available():
                    st.error("❌ API is not available. Please ensure FastAPI server is running.")
                    st.info("💡 Start the server with: `uvicorn app.api.main:app --reload`")
                    return
            
            # Search via API
            with st.spinner(f"Searching for top {topk} similar artworks..."):
                results = client.search_image(img, topk=topk)
            
            # Display results
            artworks = results.get("results", [])
            if not artworks:
                st.warning("No similar artworks found.")
                return
            
            st.subheader(f"Top {len(artworks)} similar images")
            cols = st.columns(len(artworks))
            
            for col, artwork in zip(cols, artworks):
                title = artwork.get("title") or "Untitled"
                artist = artwork.get("artist") or "Unknown"
                year = artwork.get("year") or artwork.get("objectDate") or "—"
                style = artwork.get("style") or "—"
                score = artwork.get("similarity_score", 0.0)
                
                # Try to load image
                local_path = artwork.get("local_path")
                image_available = artwork.get("image_available", False)
                
                if image_available and local_path and os.path.exists(local_path):
                    col.image(local_path, use_container_width=True)
                    cap = f"{title}\n{artist} — {year}\nStyle: {style}\nScore: {score:.3f}"
                    col.caption(cap)
                elif artwork.get("image_url"):
                    col.image(artwork.get("image_url"), use_container_width=True)
                    cap = f"{title}\n{artist} — {year}\nStyle: {style}\nScore: {score:.3f}"
                    col.caption(cap)
                else:
                    col.write(f"**{title}**")
                    col.write(f"{artist} — {year}")
                    col.write(f"Style: {style}")
                    col.write(f"Score: {score:.3f}")
                    col.caption("(image not available)")
        
        except Exception as e:
            st.error(f"❌ API request failed: {str(e)}")
            st.info("💡 Falling back to direct FAISS mode...")
            use_api = False  # Fallback to direct mode
    
    if not use_api or not API_CLIENT_AVAILABLE:
        # ========== DIRECT FAISS MODE (fallback) ==========
        try:
            model, proc, device = load_model()
            index, metas = load_index_and_meta(index_path, meta_path)

            # Embed + search
            with st.spinner("Embedding image and searching..."):
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
        except Exception as e:
            st.error(f"❌ Direct FAISS mode failed: {str(e)}")
            st.exception(e)

