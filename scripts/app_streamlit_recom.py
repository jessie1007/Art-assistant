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


from REDACTED import embed_image
from rerank_tools import hue_hist, texture_vec, combine_scores
from feedback_tool import make_feedback
from faiss_utils import load_faiss_index, id_to_meta, search_index
from feedback_tool import interpret_feedback
from scripts.rag_engine import load_tips, retrieve_tips


# ============================================================
# Public API: render_recom_tab (used in main launcher)
# ============================================================
import ui_legend as ui



def render_recom_tab(
    img,
    labels=None,
    device="cpu",
    index_path="data/index_samples/index.faiss",
    topk_initial=20,
    final_k=5,
    show_feedback=True,
    meta_path="data/index_samples/meta.npy",
):
    """Render the Recommend & Critique UI using the shared image from the launcher."""
    st.header("🎨 Recommend & Critique")


    if img is None:
        st.info("Upload an image above to get recommendations and feedback.")
        return

    # Center preview (same visual rhythm as your style tab)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption="Current image", use_container_width=True)

    # Controls (kept inside the tab like your style file)
    c1, c2, c3 = st.columns(3)
    with c1:
        topk_initial = int(st.slider("Initial CLIP Top-K", 5, 200, int(topk_initial), 5))
    with c2:
        final_k = int(st.slider("Final results", 1, 12, int(final_k), 1))
    with c3:
        show_feedback = st.checkbox("Show feedback", value=show_feedback)
    # Load FAISS index (support both signatures: with/without path arg)
    try:
        index = load_faiss_index(index_path)
    except TypeError:
        index = load_faiss_index()

    # --- Embed query (normalized CLIP vector) ---
    q_vec = embed_image(img)  # shape (D,), float32

    # --- Initial retrieval ---
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
        # your meta likely has local_path; fall back gracefully
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

    # --- Show recommendations in a row (like style preview) ---
    st.write("### Similar works (re-ranked)")
    cols = st.columns(max(1, len(recs)))
    for col, rec in zip(cols, recs):
        cap = f'{rec["title"]} · score {rec["score"]:.2f}'
        p = rec["thumbnail_path"]
        if p and Path(p).exists():
            col.image(p, caption=cap, use_container_width=True)
        else:
            col.write(cap)

    # --- Feedback (rule-based CV critique) ---
    if show_feedback:
        st.write("### Feedback")
        fb = make_feedback(np.array(img))
        interp = interpret_feedback(fb)

        # Badges + short summary
        cols = st.columns(3)
        cols[0].metric("Value key", interp["ratings"]["value_key"])
        cols[1].metric("Contrast",   interp["ratings"]["contrast"])
        cols[2].metric("Saturation", interp["ratings"]["saturation"])

        st.caption(interp["summary"])
        if interp["badges"]:
            st.write(" • ".join(f"`{b}`" for b in interp["badges"]))

        # A few quick sliders/bars (optional visual)
        st.progress(min(1.0, fb["composition"]["thirds_score"] or 0.0))

        # Actionable suggestions
        st.write("#### Suggested edits")
        for s in interp["suggestions"]:
            st.write(f"• {s}")
            #show legend

    ui.render_legend_expander()
    with st.expander("Raw metrics", expanded=False):
        st.json(fb)


    # build facts that align with your rules
    facts = {
        "thirds_score": float(fb["composition"].get("thirds_score", 0.0)),
        "entropy":      float(fb["composition"].get("entropy", 0.0)),
        "sat_mean":     float(fb["color"].get("saturation_mean", 0.0)),
        # simple focal contrast proxy: local contrast near COM vs global
        "focal_local_contrast": float(interp.get("focal_local_contrast", 0.0)) if "focal_local_contrast" in interp else 0.0,
    }

    tips_db = load_tips()
    tips    = retrieve_tips(facts, tips_db)

    st.write("#### Curated tips (RAG)")
    if tips:
        for t in tips:
            st.write(f"• {t}")
    else:
        st.info("No matching tips yet — add some to data/tips/tips.jsonl")

    # 3) Final critique (no LLM): your summary + selected tips
    st.write("#### Final critique")
    final_text = interp["summary"] + ("\n\n" + " ".join(tips) if tips else "")
    st.write(final_text)

    # 4) Raw metrics behind a toggle only
    with st.expander("See raw metrics", expanded=False):
        st.json(fb)

# ============================================================
# Standalone runner (so you can run this file by itself too)
# ============================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Recommend & Critique", layout="wide")
    st.title("🎨 Art Assistant")

    # Simple uploader for standalone use (main launcher already has one)
    uploaded = st.file_uploader("Upload a painting (jpg/png)", type=["jpg", "jpeg", "png"])
    img = Image.open(uploaded).convert("RGB") if uploaded else None

    render_recom_tab(img=img)


