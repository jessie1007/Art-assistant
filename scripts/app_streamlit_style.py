# scripts/app_streamlit_style.py
# --- make project root importable ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]   # project root (…/Art-assistant)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------

import streamlit as st
from PIL import Image
import yaml

from src.infer import ZeroShotCLIP


# ============================================================
# Public API: render_style_tab (used in main launcher)
# ============================================================
def render_style_tab(labels=None, device="cpu"):
    """Render the Style Classifier UI."""
    st.header("🎭 Style Classifier (Zero-Shot)")

    # fallback: load config if not provided
    if labels is None:
        cfg = yaml.safe_load(open("configs/style_v1.yaml"))
        labels = cfg["labels"]
        device = cfg.get("device", "cpu")

    zs = ZeroShotCLIP(device=device)

    uploaded = st.file_uploader(
        "Upload an artwork", 
        type=["jpg", "png", "jpeg"], 
        key="style_tab_upload"  # avoid widget collision
    )
    if not uploaded:
        st.info("Upload an image to classify.")
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Artwork", use_container_width=True)

    preds = zs.predict_topk(img, labels, k=3)
    st.write("### Top Predictions")
    for label, score in preds:
        st.write(f"{label}: {score:.2%}")


# ============================================================
# Standalone runner (so you can run this file by itself too)
# ============================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Style Classifier", layout="wide")
    st.title("🎨 Art Assistant")

    cfg = yaml.safe_load(open("configs/style_v1.yaml"))
    render_style_tab(labels=cfg["labels"], device=cfg.get("device", "cpu"))
