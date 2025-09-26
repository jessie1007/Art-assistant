import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import scripts.app_streamlit_retrieval as retr
import scripts.app_streamlit_style as style
import yaml, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cfg = yaml.safe_load(open(ROOT / "configs" / "style_v1.yaml"))
labels = cfg["labels"]; device = cfg.get("device", "cpu")

st.set_page_config(page_title="🎨 Art Assistant", layout="wide")
st.title("🎨 Art Assistant")

with st.sidebar:
    st.header("Retrieval Settings")
    index_path = st.text_input("FAISS index path", "data/index_samples/index.faiss", key="idx")
    meta_path  = st.text_input("Metadata JSONL path", "data/embeddings_samples/artwork_index.jsonl", key="meta")
    topk       = st.slider("Top-K", 1, 10, 5, key="topk")
    st.header("Value Study Settings")
    k_values   = st.slider("Value K (bands)", 3, 8, 5, key="kvals")
    show_grid  = st.checkbox("Show thirds grid + COM", value=True, key="grid")

tab1, tab2, tab3 = st.tabs(["🔎 Retrieval", "🧪 Value Studies", "🎭 Style Classifier"])
with tab1:
    retr.render_retrieval_tab(index_path=index_path, meta_path=meta_path, topk=topk)
with tab2:
    retr.render_value_tab(k_values=k_values, show_grid=show_grid)
with tab3:
    style.render_style_tab(labels=labels, device=device)
