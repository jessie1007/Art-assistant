from __future__ import annotations

from typing import List, Dict, Tuple
from PIL import Image
import numpy as np, cv2
import io 
import streamlit as st
from value_block import big_value_blocks, overlay_blocks_on


# ---------- Fast cached helpers ----------
@st.cache_data(show_spinner=False, max_entries=64)
def _prep_arrays(img_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Decode once → RGB & Gray arrays (hashable for cache)."""
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    rgb = np.array(pil)                            # (H,W,3) uint8
    gray = np.array(pil.convert("L"))              # (H,W) uint8
    return rgb, gray

@st.cache_data(show_spinner=False, max_entries=64)
def _blocks_cached(rgb: np.ndarray, k: int, spatial: float, downscale: int) -> np.ndarray:
    """Heavy step: big value blocks → uint8 grayscale (H,W)."""
    pil = Image.fromarray(rgb, "RGB")
    blocks = big_value_blocks(pil, k=k, spatial_weight=spatial, downscale=downscale)
    return np.array(blocks, dtype=np.uint8)

@st.cache_data(show_spinner=False, max_entries=64)
def _plan_kmeans(gray: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """3–8 value plan via k-means on luminance (kept for comparison)."""
    x = gray.reshape(-1, 1).astype(np.float32)
    k = max(2, int(k))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _labels, centers = cv2.kmeans(x, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    centers = np.sort(centers.flatten())
    idx = np.argmin(np.abs(gray[..., None] - centers[None, None, :]), axis=2)
    poster = centers[idx].astype(np.uint8)
    return poster, centers

def _plan_quantile(gray: np.ndarray, k: int, downscale_for_simplicity: int | None = 360) -> Tuple[np.ndarray, np.ndarray]:
    """3–8 value plan via quantiles (≈ equal pixels per band). Calmer default."""
    g = gray
    if downscale_for_simplicity is not None:
        h, w = g.shape
        if max(h, w) > downscale_for_simplicity:
            if w >= h:
                new_w = downscale_for_simplicity; new_h = int(h * (downscale_for_simplicity / w))
            else:
                new_h = downscale_for_simplicity; new_w = int(w * (downscale_for_simplicity / h))
            g_small = np.array(Image.fromarray(g, "L").resize((new_w, new_h), Image.Resampling.LANCZOS))
        else:
            g_small = g
    else:
        g_small = g

    qs = np.linspace(0, 100, k+1)
    edges = np.percentile(g_small, qs).astype(np.int32)
    centers = ((edges[:-1] + edges[1:]) // 2).astype(np.uint8)
    labels_small = np.digitize(g_small, edges[1:-1], right=False)
    plan_small = centers[labels_small].astype(np.uint8)

    if plan_small.shape != gray.shape:
        plan = np.array(Image.fromarray(plan_small, "L")
                        .resize((gray.shape[1], gray.shape[0]), Image.Resampling.NEAREST))
    else:
        plan = plan_small

    # tiny 3x3 median to knock speckle (pure NumPy)
    pad = np.pad(plan, 1, mode="edge")
    tiles = [pad[i:i+plan.shape[0], j:j+plan.shape[1]] for i in range(3) for j in range(3)]
    plan = np.median(np.stack(tiles, axis=0), axis=0).astype(np.uint8)
    return plan, centers

def value_percentages(poster: np.ndarray, centers: np.ndarray):
    """Percent of pixels in each value band (len=k, sums to 1.0)."""
    idx = np.argmin(np.abs(poster[..., None] - centers[None, None, :]), axis=2)
    counts = np.bincount(idx.ravel(), minlength=len(centers)).astype(np.float32)
    pct = counts / counts.sum()
    return pct  # array length k, sums to 1.0

def _get_thresholds(gray: np.ndarray,
                    cfg: dict | None,
                    mode: str = "fixed",
                    plan_edges: Tuple[np.ndarray, np.ndarray] | None = None) -> tuple[float, float]:
    """
    Returns (dark_t, light_t) in 0..1 based on mode:
      - "fixed": from cfg.value_thresholds or defaults (0.20/0.80)
      - "auto": per-image quantiles (q20, q80)
      - "plan": use value plan edges/centers if provided
    """
    # 1) fixed (pedagogical)
    if mode == "fixed":
        vt = (cfg or {}).get("value_thresholds", {})
        dark_t  = float(vt.get("dark",  0.20))
        light_t = float(vt.get("light", 0.80))
        return dark_t, light_t

    g = gray.astype(np.float32) / 255.0

    # 2) auto (data-driven)
    if mode == "auto":
        dark_t  = float(np.quantile(g, 0.20))
        light_t = float(np.quantile(g, 0.80))
        return dark_t, light_t

    # 3) plan-anchored (use plan bands)
    if mode == "plan" and plan_edges is not None:
        plan_img, centers = plan_edges  # plan_img: uint8 greys at band centers, centers: 0..255
        centers01 = np.sort(centers.astype(np.float32) / 255.0)
        # take 2nd darkest as dark threshold, 2nd lightest as light threshold (robust)
        if len(centers01) >= 4:
            return float(centers01[1]), float(centers01[-2])
        elif len(centers01) >= 3:
            return float(centers01[0]), float(centers01[-1])

    # fallback (degenerate)
    return 0.20, 0.80

def _thresholds_hybrid(
    gray: np.ndarray,
    cfg: dict | None = None,
    plan_edges: Tuple[np.ndarray, np.ndarray] | None = None
) -> tuple[float, float, str, str]:
    """
    Hybrid thresholds for dark/light (0..1):
      - default: fixed from cfg (0.20 / 0.80 if not set)
      - switch to AUTO (quantiles) if image is low-contrast or fixed thresholds
        produce almost no dark/light coverage
      - optionally use PLAN if provided and distribution is super extreme
    Returns: (dark_t, light_t, mode_used, reason)
    """
    # 1) fixed (pedagogical default)
    vt = (cfg or {}).get("value_thresholds", {})
    dark_fixed  = float(vt.get("dark",  0.20))
    light_fixed = float(vt.get("light", 0.80))

    g = gray.astype(np.float32) / 255.0
    contrast = float(g.std())

    # coverage under fixed thresholds
    dark_cov_fixed  = float((g < dark_fixed).mean())
    light_cov_fixed = float((g > light_fixed).mean())

    # 2) low-contrast or “no coverage” → AUTO quantiles (q20/q80)
    if (contrast < 0.11) or (dark_cov_fixed < 0.02 and light_cov_fixed < 0.02):
        dark_auto  = float(np.quantile(g, 0.20))
        light_auto = float(np.quantile(g, 0.80))
        # guard if degenerate
        if not np.isfinite(dark_auto) or not np.isfinite(light_auto) or dark_auto >= light_auto:
            return dark_fixed, light_fixed, "fixed", "fallback: degenerate auto thresholds"
        return dark_auto, light_auto, "auto", (
            f"auto: low contrast (σ={contrast:.2f}) or minimal fixed coverage "
            f"(dark={dark_cov_fixed:.1%}, light={light_cov_fixed:.1%})"
        )

    # 3) extremely skewed distributions → PLAN (if available)
    #    e.g., tons of darks or lights; plan avoids chasing tiny tails
    if plan_edges is not None:
        plan_img, centers = plan_edges
        centers01 = np.sort(centers.astype(np.float32) / 255.0)
        if len(centers01) >= 4:
            if dark_cov_fixed > 0.70 or light_cov_fixed > 0.70:
                return float(centers01[1]), float(centers01[-2]), "plan", (
                    f"plan: extreme coverage (dark={dark_cov_fixed:.1%}, light={light_cov_fixed:.1%})"
                )

    # 4) otherwise stick with fixed (stable for learning)
    return dark_fixed, light_fixed, "fixed", "default: pedagogical fixed thresholds"


# ================================
# Public UI: render_value_tab
# ================================

def render_value_tab(img: Image.Image, k_values: int = 7, cfg: dict | None = None):
    """
    Two rows layout (live updates, no form):
      Row 1: Reference | Big Value Blocks
      Row 2: Grayscale | Value Plan
    """

    st.header("🧪 Value Studies")
    if img is None:
        st.info("Upload a photo or painting in the main app (top).")
        return

    mode = st.radio("Value thresholds", ["fixed", "auto", "plan"], index=0,
        help="fixed=consistent; auto=adapts to image; plan=derived from value plan")

    # --- decode once & cache by bytes ---
    buf = io.BytesIO(); img.save(buf, format="PNG"); img_bytes = buf.getvalue()
    rgb, gray = _prep_arrays(img_bytes)  # cached

    # -------- Controls (live, no form) --------
    left, right = st.columns([1, 1])
    with left:
        k_blocks = st.slider("Blocks (K)", 4, 6, 5, 1,
                             help="How many big shapes. Fewer = simpler; more = more detail.")
        show_overlay = st.checkbox("Show overlay on photo", value=False,
                                   help="Off = pure abstraction. Toggle only if you need orientation.")
        preview_w = st.slider("Preview width (px)", 320, 960, 560, 20,
                              help="How wide to render each panel.")
    with right:
        with st.expander("Advanced (blocks engine)", expanded=False):
            spatial   = st.slider("Spatial coherence", 0.0, 1.0, 0.6, 0.05,
                                help="Higher → bigger, connected masses. Lower → follows exact values.")
            downscale = st.slider("Processing downscale (px)", 240, 600, 360, 20,
                                help="Smaller = simpler & faster. 300–420 is a good range.")

    # Value Plan controls (always shown)
    with st.expander("Refine values (Value Plan)", expanded=True):
        k_plan = st.slider("Plan K (bands)", 3, 8, k_values, 1,
                           help="How many steps in the refinement plan. 5–7 is common.")
        method = st.selectbox("Plan method", ["quantile", "kmeans"], index=0,
                              help="Quantile = even steps; K-means = adapts to histogram.")

    # -------- Compute (cached where heavy) --------
    # Big blocks (cached)
    blocks_np = _blocks_cached(rgb, k_blocks, spatial, downscale)
    blocks_pil = Image.fromarray(blocks_np, "L")
    overlay = overlay_blocks_on(img, blocks_pil, alpha=0.35) if show_overlay else None

    # Value Plan (live)
    if method == "quantile":
        plan, plan_centers = _plan_quantile(gray, k_plan)  # fast; not cached
    else:
        plan, plan_centers = _plan_kmeans(gray, k_plan)    # cached
    plan_pil = Image.fromarray(plan, "L")
    plan_pct = value_percentages(plan, plan_centers)

    g = gray.astype("float32") / 255.0
    mean_val = float(g.mean())
    contrast = float(g.std())

    dark_t, light_t, mode_used, note = _thresholds_hybrid(gray, cfg, plan_edges=(plan, plan_centers))
    dark_pct  = float((g < dark_t).mean())
    light_pct = float((g > light_t).mean())
    st.caption(f"Thresholds ({mode_used}): dark < {dark_t:.2f}, light > {light_t:.2f} — {note}")


    # ================= ROW 1 =================
    r1c1, r1c2 = st.columns([1, 1], gap="large")
    with r1c1:
        st.subheader("Reference")
        st.image(img, width=preview_w)
    with r1c2:
        st.subheader(f"Big Value Blocks (K={k_blocks})")
        st.image(blocks_pil, width=preview_w, caption="Flat greys — block these in first")
        if overlay is not None:
            st.image(overlay, width=preview_w, caption="Overlay (for orientation)")
        _tips_for_blocks(blocks_np, k_blocks)

    # ================= ROW 2 =================
    r2c1, r2c2 = st.columns([1, 1], gap="large")
    with r2c1:
        st.subheader("Grayscale")
        st.image(Image.fromarray(gray, "L"), width=preview_w, caption="Grayscale (squint test)")
        # Always show quick metrics underneath
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean (0–1)", f"{mean_val:.2f}")
        m2.metric("Contrast (σ)", f"{contrast:.2f}")
        m3.metric("Dark %", f"{dark_pct*100:.1f}%")
        m4.metric("Light %", f"{light_pct*100:.1f}%")

    with r2c2:
        st.subheader(f"Value Plan (K={k_plan}, method: {method})")
        st.image(plan_pil, width=preview_w, caption="Discrete steps — refine lights/shadows")
        st.caption("Centers (0–255): " + ", ".join(str(int(c)) for c in plan_centers))
        st.write("**Value distribution**: " + ", ".join(
            f"Step {i+1}: {p*100:.1f}%" for i, p in enumerate(plan_pct)
        ))



# ---------- Tiny heuristic tips ----------
def _tips_for_blocks(blocks_np: np.ndarray, k_blocks: int):
    """
    Lightweight coaching based on block areas & values.
    """
    vals, counts = np.unique(blocks_np, return_counts=True)
    if len(vals) != k_blocks:
        # sometimes tiny bands can collapse; still give guidance
        st.info("Some value steps merged; try lower K or raise spatial coherence.")
    total = max(1, blocks_np.size)
    areas = counts / total                         # area fraction per block
    order = np.argsort(areas)[::-1]                # largest first
    vals = vals[order]; areas = areas[order]

    # Tip 1: Dominant vs Secondary too similar
    if len(areas) >= 2:
        area_gap = areas[0] - areas[1]
        val_gap  = abs(int(vals[0]) - int(vals[1])) / 255.0
        if area_gap < 0.05 and val_gap < 0.08:
            st.warning("Your two biggest masses are close in size **and** value. Consider merging or pushing one darker/lighter for a clear dominant shape.")

    # Tip 2: Too busy (tiny slivers)
    if areas.min() < 0.03:
        st.info("Map looks busy (tiny slivers detected). Try **K=4** or increase **Spatial coherence** to simplify.")

    # Tip 3: Extremely balanced (can feel static)
    if len(areas) >= 3 and (areas[0] < 0.30 and areas[1] > 0.20):
        st.info("Masses are very balanced. Consider making one mass clearly dominant (≥ 30%).")

    # (You can add edge-plan or background separation tips later.)