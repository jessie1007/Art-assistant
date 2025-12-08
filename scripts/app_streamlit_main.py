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
from PIL import Image, UnidentifiedImageError


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
    st.header("🔌 API Settings")
    use_api = st.checkbox("Use FastAPI backend", value=True, help="Connect to FastAPI server for image search")
    api_url = st.text_input(
        "API URL",
        value="http://localhost:8000",
        help="FastAPI server URL (default: http://localhost:8000)",
        key="api_url"
    )
    
    if use_api:
        # Quick health check
        try:
            from app.api_client import ArtAssistantAPIClient
            client = ArtAssistantAPIClient(api_url=api_url)
            health = client.health_check()
            if health.get("status") == "healthy":
                st.success(f"✅ API Connected\nIndex: {health.get('index_size', 0)} artworks")
            else:
                st.warning("⚠️ API not responding")
        except Exception as e:
            st.error(f"❌ API Error: {str(e)}")
    
    st.divider()
    
    st.header("Retrieval Settings")
    index_path = st.text_input("FAISS index path", "data/index_samples/index.faiss", key="idx", disabled=use_api)
    meta_path  = st.text_input("Metadata JSONL path", "data/embeddings_samples/artwork_index.jsonl", key="meta", disabled=use_api)
    topk       = st.slider("Top-K", 1, 10, 5, key="topk")

    st.header("Value Study Settings")
    k_values   = st.slider("Value K (bands)", 3, 8, 5, key="kvals")
    #show_com   = st.checkbox("Show COM (balance point)", value=False, key="show_com")

tab1, tab2, tab3, tab4 = st.tabs(["🔎 Retrieval", "🧪 Value Studies", "🎭 Style Classifier", "🎨 Recommend & Critique"])
with tab1:
    retr.render_retrieval_tab(
        img=img,
        index_path=index_path,
        meta_path=meta_path,
        topk=topk,
        api_url=api_url if use_api else None,
        use_api=use_api
    )
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
        meta_path=meta_path,
        cfg=cfg
    )
