from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------------------------------------
import streamlit as st
from PIL import Image
import numpy as np
import cv2

from scripts.llm_helper import build_coach_prompt, call_llm
from hf_embed_global import embed_image
from dotenv import load_dotenv
from rerank_tools import hue_hist, texture_vec, combine_scores
from feedback_tool import make_feedback, interpret_feedback
from faiss_utils import load_faiss_index, id_to_meta, search_index

# =========================
# Small local UI utilities
# =========================

def _show_palette_swatches(palette: list[dict], title: str = "Palette (dominance order)"):
    if not palette:
        return
    st.subheader(title)
    cols = st.columns(len(palette))
    for col, p in zip(cols, palette):
        hexv = p.get("hex", "#888888")
        pct = float(p.get("pct", 0.0)) * 100.0
        col.markdown(
            f"""
            <div style="border-radius:12px; overflow:hidden; border:1px solid rgba(0,0,0,.08);">
              <div style="height:72px; background:{hexv};"></div>
              <div style="padding:8px; font-size:0.9rem;">
                <div><code>{hexv}</code></div>
                <div style="opacity:.7;">{pct:.1f}%</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
import re

def _normalize(s: str) -> str:
    """Lowercase + strip punctuation for matching."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _dedupe_suggestions(suggestions, llm_text):
    """
    Remove rule suggestions that overlap with LLM advice.
    Also avoid near-duplicate rule tips by first 6 words.
    """
    if not suggestions:
        return suggestions

    llm_norm = _normalize(llm_text or "")
    keep = []
    seen = set()

    for s in suggestions:
        n = _normalize(s)
        # Skip if LLM already said something very similar
        if n and n in llm_norm:
            continue
        # Collapse near duplicates
        key = " ".join(n.split()[:6])
        if key in seen:
            continue
        seen.add(key)
        keep.append(s)

    # Cap to at most 3 extra rule tips
    return keep[:3]
def _merge_duplicate_swatches(palette):
    if not palette:
        return []
    agg = {}
    for p in palette:
        hexv = p.get("hex", "#000000").lower()
        agg[hexv] = agg.get(hexv, 0.0) + float(p.get("pct", 0.0))
    total = sum(agg.values()) or 1.0
    merged = [{"hex": h, "pct": v / total} for h, v in agg.items()]
    merged.sort(key=lambda x: x["pct"], reverse=True)
    return merged

def chip(label: str, tone: str = "neutral"):
    colors = {
        "good":   "#16a34a",  # green
        "warn":   "#d97706",  # amber
        "bad":    "#dc2626",  # red
        "neutral":"#475569",  # slate
    }
    c = colors.get(tone, colors["neutral"])
    st.markdown(
        f"""
        <span style="
          display:inline-block; padding:.35rem .6rem; margin:.15rem;
          border-radius:999px; font-size:.85rem; line-height:1;
          color:white; background:{c};">{label}</span>
        """,
        unsafe_allow_html=True
    )

def palette_row(palette: list[dict]):
    if not palette: 
        return
    cols = st.columns(min(6, len(palette)))
    for col, p in zip(cols, palette[:6]):
        hexv = p.get("hex", "#888")
        pct  = float(p.get("pct", 0))*100
        col.markdown(
            f"""
            <div style="border:1px solid rgba(0,0,0,.06); border-radius:10px; overflow:hidden;">
              <div style="height:48px; background:{hexv};"></div>
              <div style="text-align:center; padding:6px; font-size:.8rem;">
                <code>{hexv}</code><br/>{pct:.0f}%
              </div>
            </div>
            """, unsafe_allow_html=True
        )

def focus_chip_from(entropy_n: float, focal_strength: float) -> tuple[str, str]:
    # Simple rubric: combine spread + peak strength
    if entropy_n <= 0.40 and focal_strength >= 6.0:   return ("Focus: Clear", "good")
    if entropy_n <= 0.65 and focal_strength >= 3.5:   return ("Focus: Okay",  "neutral")
    return ("Focus: Diffuse", "warn")
# --- Friendly explanations for chips ---
def explain_key(label: str) -> str:
    table = {
        "low-key": "Image sits mostly in dark values (moody/cinematic).",
        "mid-key": "Mostly mid-tones (balanced, everyday lighting).",
        "high-key": "Mostly light values (airy, gentle).",
    }
    return table.get(label, "Overall value/brightness feel of the image.")

def explain_contrast(label: str) -> str:
    table = {
        "low": "Lights and darks are close together → softer, quieter read.",
        "moderate": "Some separation between lights/darks → clear but not harsh.",
        "high": "Strong separation → bold, dramatic, and high clarity.",
    }
    return table.get(label, "Separation between light and dark areas.")

def explain_color(label: str) -> str:
    table = {
        "muted": "Low saturation → restrained palette; accents will pop more.",
        "natural": "Moderate saturation → believable, harmonious color.",
        "vivid": "High saturation → punchy color; control accents to avoid noise.",
    }
    return table.get(label, "Overall saturation feel of the image.")

def explain_focus(entropy_n: float, focal_strength: float) -> tuple[str, str]:
    """
    Returns (chip_label, friendly_explainer). Uses same thresholds as your focus chip.
    """
    if entropy_n <= 0.40 and focal_strength >= 6.0:
        return ("Focus: Clear", "Attention converges well on a main subject.")
    if entropy_n <= 0.65 and focal_strength >= 3.5:
        return ("Focus: Okay", "Subject reads, but secondary areas still attract attention.")
    return ("Focus: Diffuse", "Attention spreads across the image; focal point feels vague.")

# ============================================================
# Public API: render_recom_tab (used in main launcher)
# ============================================================

def render_recom_tab(
    img: Image.Image,
    labels=None,
    device="cpu",
    index_path="data/index_samples/index.faiss",
    topk_initial=20,
    final_k=5,
    show_feedback=True,
    meta_path="data/index_samples/meta.npy",
    cfg=None
):
    """Render the Recommend & Critique UI without RAG facts/tips."""
    st.header("🎨 Recommend & Critique")

    if img is None:
        st.info("Upload an image above to get recommendations and feedback.")
        return

    # Center preview
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption="Current image", use_container_width=True)

    # Controls
    c1, c2 = st.columns(2)
    with c1:
        topk_initial = int(st.slider("Initial CLIP Top-K", 5, 200, int(topk_initial), 5))
    with c2:
        final_k = int(st.slider("Final results", 1, 12, int(final_k), 1))

    # Load FAISS index
    try:
        index = load_faiss_index(index_path)
    except TypeError:
        index = load_faiss_index()

    # --- Embed query (CLIP) ---
    q_vec = embed_image(img)  # (D,) float32

    # --- Initial retrieval ---
    with st.spinner("Searching similar works..."):
        cand_ids, cand_sims = search_index(index, q_vec, topk=topk_initial)

    # --- Query features for adaptive re-rank ---
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    q_h = hue_hist(bgr)
    q_t = texture_vec(bgr)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(hsv[..., 1].mean() / 255.0)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    grad_mag_mean = float(np.sqrt(gx * gx + gy * gy).mean() / 255.0)

    q_feats = {
        "h_hist": q_h,
        "t_vec": q_t,
        "sat_mean": sat_mean,
        "grad_mag_mean": grad_mag_mean,
    }

    # --- Build candidates (lookup meta per id) ---
    cands = []
    for _id, s in zip(cand_ids, cand_sims):
        m = id_to_meta(int(_id)) or {}
        thumb = m.get("thumbnail_path") or m.get("local_path") or m.get("path", "")
        cands.append({
            "id": int(_id),
            "title": m.get("title", str(_id)),
            "thumbnail_path": thumb,
            "clip_sim": float(s),
            "h_hist": m.get("h_hist", q_h),
            "t_vec":  m.get("t_vec",  q_t),
        })

    # --- Re-rank & slice final K ---
    scores = combine_scores(q_feats, cands)
    for c, sc in zip(cands, scores):
        c["score"] = float(sc)
    recs = sorted(cands, key=lambda x: x["score"], reverse=True)[:final_k]

    # --- Show recommendations ---
    st.write("### Similar works (re-ranked)")
    if not recs:
        st.info("No similar works found.")
    else:
        cols = st.columns(len(recs))
        for col, rec in zip(cols, recs):
            cap = f'{rec["title"]} · score {rec["score"]:.2f}'
            p = rec["thumbnail_path"]
            if p and Path(p).exists():
                col.image(p, caption=cap, use_container_width=True)
            else:
                col.write(cap)

    # --- Compute rule-based feedback ONCE ---
    rgb_np = np.array(img)
    fb = make_feedback(rgb_np, cfg=cfg)
    interp = interpret_feedback(fb)


    # --- LLM quick take (primary guidance) ---
    llm_text = ""
    llm_cfg = (cfg or {}).get("llm", {})
    try:
        sys_prompt, usr_prompt = build_coach_prompt(img=None, features=fb, interp=interp)
        llm_text = call_llm(
            system_prompt=sys_prompt,
            user_prompt=usr_prompt,
            model=None,
            max_tokens=llm_cfg.get("max_tokens", 600),
        ) or ""
        # collapse blank lines
        llm_text = "\n".join([ln for ln in llm_text.splitlines() if ln.strip()])
        st.subheader("LLM quick take")
        st.markdown(llm_text)
    except Exception as e:
        st.warning(f"LLM call failed: {e}")

    # --- Feedback (rules) ---
 
    # --- Snapshot (simple cues + one-liners) ---
    st.subheader("Snapshot")

    # 3–4 small chips
    c1, c2, c3, c4 = st.columns(4)
    key_lbl   = interp['ratings']['value_key']              # e.g., "mid-key"
    cont_lbl  = interp['ratings']['contrast']               # "low|moderate|high"
    color_lbl = interp['ratings']['saturation']             # "muted|natural|vivid"

    with c1:
        chip(f"Key: {key_lbl.replace('-key','').title()}")
    with c2:
        chip(f"Contrast: {cont_lbl.title()}")
    with c3:
        chip(f"Color: {color_lbl.title()}")

    comp = fb["composition"]
    f_label, f_text = explain_focus(
        comp.get('entropy_n', 0.5),
        comp.get('focal_strength', 1.0)
    )
    with c4:
        # tone decided by focus label
        tone = "good" if "Clear" in f_label else ("neutral" if "Okay" in f_label else "warn")
        chip(f_label, tone)

    # Short friendly explanations under the chips
    st.markdown(
        f"- **Key:** {explain_key(key_lbl)}\n"
        f"- **Contrast:** {explain_contrast(cont_lbl)}\n"
        f"- **Color:** {explain_color(color_lbl)}\n"
        f"- **{f_label}:** {f_text}"
    )

    # Optional: raw metrics only if user opens the expander (debugging / power users)
    with st.expander("Why this verdict (details)", expanded=False):
        st.caption(interp["summary"])
        st.json(fb)


# ============================================================
# Standalone runner
# ============================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Recommend & Critique", layout="wide")
    st.title("🎨 Art Assistant")

    uploaded = st.file_uploader("Upload a painting (jpg/png)", type=["jpg", "jpeg", "png"])
    img = Image.open(uploaded).convert("RGB") if uploaded else None
    render_recom_tab(img=img)
