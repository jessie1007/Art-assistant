import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import streamlit as st
import scripts.app_streamlit_retrieval as retr
import scripts.value_tools as value
import scripts.app_streamlit_style as style
import scripts.app_streamlit_recom as recom
import yaml, sys, pathlib
from PIL import Image

@st.cache_data(show_spinner=False)
def load_cfg(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

CFG_PATH = ROOT / "configs" / "style_v1.yaml"
cfg = load_cfg(CFG_PATH)
labels = cfg.get("labels", [])
device = cfg.get("device", "cpu")

if not labels:
    st.warning("No 'labels' found in style_v1.yaml — Style tab will show limited output.")

st.set_page_config(page_title="🎨 Art Assistant", layout="wide")
st.title("🎨 Art Assistant")

# ---- Global uploader (single place) ----
uploaded = st.file_uploader("Upload a painting (jpg/png)", type=["jpg","jpeg","png"], key="global_upload")
img = None
if uploaded:
    try:
        img = Image.open(uploaded).convert("RGB")
    except UnidentifiedImageError:
        st.error("That file isn't a readable image. Please upload a JPG/PNG/WEBP.")
        img = None


with st.sidebar:
    st.header("Retrieval Settings")
    index_path = st.text_input("FAISS index path", "data/index_samples/index.faiss", key="idx")
    meta_path  = st.text_input("Metadata JSONL path", "data/embeddings_samples/artwork_index.jsonl", key="meta")
    topk       = st.slider("Top-K", 1, 10, 5, key="topk")

    st.header("Value Study Settings")
    k_values   = st.slider("Value K (bands)", 3, 8, 5, key="kvals")
    #show_com   = st.checkbox("Show COM (balance point)", value=False, key="show_com")

tab1, tab2, tab3, tab4 = st.tabs(["🔎 Retrieval", "🧪 Value Studies", "🎭 Style Classifier", "🎨 Recommend & Critique"])
with tab1:
    retr.render_retrieval_tab(img=img, index_path=index_path, meta_path=meta_path, topk=topk)
with tab2:
    value.render_value_tab(img=img, k_values=k_values)
with tab3:
    style.render_style_tab(img=img, labels=labels, device=device)
with tab4:
    recom.render_recom_tab(
        img=img,
        labels=labels,
        device=device,
        index_path=index_path,
        meta_path=meta_path
    )
