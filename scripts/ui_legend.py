# scripts/ui_legend.py
import streamlit as st

def _legend_body_md() -> str:
    return """
**Recommendations (Retrieval)**
- First stage: CLIP cosine similarity from FAISS index.
- Re-ranking nudges: hue histogram, texture/edge energy, mean saturation.

**Value (grayscale)**
- Key (mean): low < **0.40**, mid **0.40–0.60**, high > **0.60**
- Contrast (σ): low < **0.12**, moderate **0.12–0.20**, high > **0.20**
- Dark% = pixels < **20%** gray; Light% = pixels > **80%** gray.

**Composition (no rule-of-thirds)**
- Attention spread (entropyₙ in 0–1):
  - **focused < 0.35**
  - **balanced 0.35–0.65**
  - **diffuse > 0.65**
- Focal strength: peak saliency ÷ average saliency
  - higher = clearer main subject
- Border pull: % of attention near edges (0–1)
  - high values = edges competing for attention

**Color**
- Mean saturation: **muted < 0.15**, **natural 0.15–0.35**, **vivid > 0.35**
- Palette extracted in Lab space → harmony & top hues.
"""

def render_legend_expander(title: str = "📎 Legend: how recommendations & critique work", expanded: bool = False):
    with st.expander(title, expanded=expanded):
        st.markdown(_legend_body_md())

def render_legend_inline(title: str = "📎 Legend"):
    st.markdown(f"### {title}")
    st.markdown(_legend_body_md())
