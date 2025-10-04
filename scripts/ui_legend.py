# scripts/ui_legend.py
import streamlit as st

def _legend_body_md() -> str:
    return """
**Recommendations**
- Start with CLIP cosine similarity from your FAISS index (paths in sidebar).
- Re-rank nudges: palette (hue histogram), texture/edge stats, mean saturation, edge energy.

**Value (grayscale)**
- Key (mean): low < **0.40**, mid **0.40–0.60**, high > **0.60**
- Contrast (σ): low < **0.12**, moderate **0.12–0.20**, high > **0.20**
- Dark% = pixels < **20%** gray; Light% = pixels > **80%** gray.

**Composition**
- Attention entropy: **focused < 12**, **balanced 12–18**, **diffuse > 18**
- COM = saliency center (x,y in 0..1); Thirds score 0..1 (closer to ⅓ hotspots is better).

**Color**
- Mean saturation: **muted < 0.15**, **natural 0.15–0.35**, **vivid > 0.35**
- If sat < **0.12** → palette labeled **neutral/achromatic**.
"""

def render_legend_expander(title: str = "📎 Legend: how recommendations & critique work", expanded: bool = False):
    # Use this ONLY at top-level (not inside another expander)
    with st.expander(title, expanded=expanded):
        st.markdown(_legend_body_md())

def render_legend_inline(title: str = "📎 Legend"):
    # Safe anywhere, including inside other expanders
    st.markdown(f"### {title}")
    st.markdown(_legend_body_md())


