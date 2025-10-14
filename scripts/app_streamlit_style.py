import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
def render_style_tab(img, labels=None, device="cpu"):
    """Render the Style Classifier UI using a shared image from the launcher."""
    st.header("🎭 Style Classifier (Zero-Shot)")

    if img is None:
        st.info("Upload an image above to get predictions.")
        return

    # fallback: load config if not provided
    if labels is None:
        cfg = yaml.safe_load(open("configs/style_v1.yaml"))
        labels = cfg["labels"]
        device = cfg.get("device", "cpu")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(img, caption="Current image", width=900)


    zs = ZeroShotCLIP(device=device)
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
