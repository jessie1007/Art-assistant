from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, List, Optional

# ---------- VALUE ----------
def value_features(
    img_bgr: np.ndarray,
    dark_t: Optional[float] = None,
    light_t: Optional[float] = None,
    cfg: Optional[Dict] = None,
) -> dict:
    """
    Basic value stats.
    If dark_t/light_t are not provided, read them from cfg.value_thresholds
    or fall back to sensible defaults (0.20 / 0.80).
    """
    if dark_t is None or light_t is None:
        vt = (cfg or {}).get("value_thresholds", {})
        dark_t  = float(vt.get("dark",  0.20))
        light_t = float(vt.get("light", 0.80))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) / 255.0
    hist, _ = np.histogram(gray, bins=11, range=(0, 1), density=True)
    return {
        "mean": float(gray.mean()),
        "contrast": float(gray.std()),
        "dark_pct": float((gray < dark_t).mean()),
        "light_pct": float((gray > light_t).mean()),
        "zone_hist": hist.tolist(),
    }

# ---------- COMPOSITION ----------
def composition_features(bgr: np.ndarray) -> dict:
    """
    Composition cues (no rule-of-thirds):
      - entropy_n: attention spread [0..1] (0=focused, 1=diffuse)
      - focal_strength: peak saliency / mean saliency
      - center_of_mass: (cx, cy) in [0,1]
      - lr_balance, tb_balance: [-1..1] (0=balanced)
      - border_pull: saliency mass in 5% border band [0..1]
      - hotspots: top-3 peaks with normalized coords + strength
    """
    # --- saliency (spectral residual; Sobel fallback) ---
    try:
        _sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, sal = _sal.computeSaliency(bgr)
        if not ok:
            raise RuntimeError
        sal = sal.astype("float32")
    except Exception:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
        sal = cv2.magnitude(gx, gy)

    sal = cv2.normalize(sal, None, 0, 1, cv2.NORM_MINMAX)
    H, W = sal.shape[:2]
    p = sal / (sal.sum() + 1e-12)

    # entropy normalized by log2(H*W)
    ent = float(-(p * np.log2(p + 1e-12)).sum())
    ent_max = float(np.log2(H * W + 1e-12))
    entropy_n = float(ent / (ent_max + 1e-12))  # 0..1

    # center of mass
    ys, xs = np.mgrid[0:H, 0:W]
    cx = float((xs * p).sum()) / W
    cy = float((ys * p).sum()) / H

    # balance L/R, T/B
    left_mass  = float(p[:, :W//2].sum())
    right_mass = float(p[:, W//2:].sum())
    top_mass   = float(p[:H//2, :].sum())
    bot_mass   = float(p[H//2:, :].sum())
    lr_balance = float((right_mass - left_mass) / (right_mass + left_mass + 1e-12))
    tb_balance = float((bot_mass - top_mass)   / (bot_mass + top_mass + 1e-12))

    # border pull (5% band)
    band = max(2, int(0.05 * max(H, W)))
    border_mask = np.zeros_like(sal, dtype=np.float32)
    border_mask[:band, :] = 1; border_mask[-band:, :] = 1
    border_mask[:, :band] = 1; border_mask[:, -band:] = 1
    border_pull = float((p * border_mask).sum())  # 0..1

    # hotspots (top-3) via crude NMS
    sal_blur = cv2.GaussianBlur(sal, (0, 0), 1.0)
    peaks = []
    work = sal_blur.copy()
    for _ in range(3):
        y, x = np.unravel_index(np.argmax(work), work.shape)
        peak_val = float(work[y, x])
        if peak_val < 1e-6:
            break
        peaks.append({"x": x / float(W), "y": y / float(H), "strength": peak_val})
        rr = max(3, int(0.03 * max(H, W)))
        y0, y1 = max(0, y-rr), min(H, y+rr+1)
        x0, x1 = max(0, x-rr), min(W, x+rr+1)
        work[y0:y1, x0:x1] = 0.0

    focal_strength = float(sal.max() / (sal.mean() + 1e-6))

    return {
        "entropy_n": entropy_n,
        "center_of_mass": {"x": cx, "y": cy},
        "lr_balance": lr_balance,
        "tb_balance": tb_balance,
        "border_pull": border_pull,
        "focal_strength": focal_strength,
        "hotspots": peaks,
    }


# ---------- COLOR ----------
def color_features(img_bgr: np.ndarray, k: int = 5) -> dict:
    """
    Returns:
      - palette: [{hex, pct}] in dominance order
      - dominant_hues: top-3 hex
      - saturation_mean: mean sat (0..1)
      - harmony: rough relationship between top hues
    """
    # 1) Convert to OpenCV 8-bit Lab; run k-means on float32 copies
    lab_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)  # uint8 in 0..255
    H, W, _ = lab_u8.shape
    X = lab_u8.reshape(-1, 3).astype(np.float32)       # kmeans input

    k = int(max(3, min(8, k)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _ret, labels, centers_lab32 = cv2.kmeans(
        X, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.ravel()

    # 2) Cast centers back to uint8 BEFORE Lab->BGR
    centers_lab_u8 = np.clip(np.round(centers_lab32), 0, 255).astype(np.uint8)
    centers_bgr_u8 = cv2.cvtColor(centers_lab_u8[None, :, :], cv2.COLOR_Lab2BGR)[0]
    centers_hsv_u8 = cv2.cvtColor(centers_bgr_u8[None, :, :], cv2.COLOR_BGR2HSV)[0]

    # 3) Palette in dominance order
    counts = np.bincount(labels, minlength=k).astype(np.float32)
    order = np.argsort(counts)[::-1]
    total = float(counts.sum()) + 1e-12

    palette = []
    for i in order:
        b, g, r = map(int, centers_bgr_u8[i])
        hexv = f"#{r:02x}{g:02x}{b:02x}"
        pct = float(counts[i] / total)
        palette.append({"hex": hexv, "pct": pct})

    dominant_hues = [p["hex"] for p in palette[:3]]

    # 4) Stable saturation mean from centers
    sats = centers_hsv_u8[:, 1].astype(np.float32) / 255.0
    saturation_mean = float(np.mean(sats))

    # 5) Harmony from hue gaps (top-3)
    hues_deg = centers_hsv_u8[:, 0].astype(np.float32) * (360.0 / 179.0)
    top_idx = order[:3]

    def circ_dist(a, b):
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    harmony = "mixed"
    if len(top_idx) >= 2:
        hs = np.sort(hues_deg[top_idx])
        gaps = [circ_dist(hs[0], hs[1])]
        if len(top_idx) == 3:
            gaps += [circ_dist(hs[1], hs[2]), circ_dist(hs[0], hs[2])]
        gmin, gmax = min(gaps), max(gaps)
        if gmin < 25:
            harmony = "analogous"
        elif 100 < gmax < 140:
            harmony = "complementary-ish"
        elif len(top_idx) == 3 and 90 < min(gaps) < 150 and 170 < max(gaps) < 200:
            harmony = "triadic-ish"

    return {
        "palette": palette,
        "dominant_hues": dominant_hues,
        "saturation_mean": saturation_mean,
        "harmony": harmony,
    }

# ---------- PUBLIC ENTRY ----------
def make_feedback(img_rgb: np.ndarray, cfg: Optional[Dict] = None) -> dict:
    """img_rgb: np.array RGB; cfg optional for thresholds."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return {
        "value": value_features(bgr, cfg=cfg),   # now passes cfg; no missing args
        "composition": composition_features(bgr),
        "color": color_features(bgr),
        "comment": "Automated critique from CV features.",
    }

# ---------- INTERPRETER ----------
def _bucket(x, edges, labels):
    idx = np.digitize([x], edges)[0]
    idx = max(0, min(idx, len(labels) - 1))
    return labels[idx]

def interpret_feedback(fb: Dict) -> Dict:
    """
    Turn numeric features into human-readable summary + suggestions.
    Returns: {
      "summary": str,
      "badges": List[str],
      "suggestions": List[str],
      "ratings": {...}
    }
    """
    out = {"summary": "", "badges": [], "suggestions": [], "ratings": {}}

    # VALUE
    v = fb.get("value", {})
    v_mean = float(v.get("mean", 0.5))
    v_contrast = float(v.get("contrast", 0.15))
    dark_pct = float(v.get("dark_pct", 0.05))
    light_pct = float(v.get("light_pct", 0.02))

    key = _bucket(v_mean, [0.4, 0.6], ["low-key", "mid-key", "high-key"])
    contrast_lbl = _bucket(v_contrast, [0.12, 0.20], ["low", "moderate", "high"])
    out["ratings"]["value_key"] = key
    out["ratings"]["contrast"] = contrast_lbl

    if dark_pct < 0.05:
        out["suggestions"].append("Add deeper shadow masses (≈5–10% of the frame) to anchor the composition.")
    if light_pct < 0.03:
        out["suggestions"].append("Introduce a small crisp highlight near the focal area to add sparkle.")
    if contrast_lbl == "low":
        out["suggestions"].append("Increase edge or value separation around the focal point for a clearer read.")

    # COMPOSITION (no “thirds rule” tip)
    # --- in interpret_feedback ---
    c = fb.get("composition", {})
    entropy_n = float(c.get("entropy_n", 0.5))
    fs        = float(c.get("focal_strength", 1.0))
    lr        = float(c.get("lr_balance", 0.0))
    tb        = float(c.get("tb_balance", 0.0))
    border    = float(c.get("border_pull", 0.0))
    hotspots  = c.get("hotspots", [])

    # attention spread label
    ent_lbl = _bucket(entropy_n, [0.35, 0.65], ["focused", "balanced", "diffuse"])
    out["ratings"]["attention"] = ent_lbl
    out["ratings"]["focal_strength"] = f"{fs:.2f}"
    out["ratings"]["balance_lr"] = f"{lr:+.2f}"
    out["ratings"]["balance_tb"] = f"{tb:+.2f}"
    out["ratings"]["border_pull"] = f"{border:.2f}"

    # suggestions (no thirds, no “move off-center”)
    if ent_lbl == "diffuse" or fs < 1.6:
        out["suggestions"].append("Clarify the subject by simplifying detail away from it and increasing local contrast/edges at the main forms.")

    # competing hotspots (hierarchy)
    if len(hotspots) >= 2:
        svals = sorted([h["strength"] for h in hotspots], reverse=True)
        if svals[1] > 0.75 * svals[0]:
            out["suggestions"].append("Several areas compete for attention—subdue one with softer edges or lower chroma to establish a clear hierarchy.")

    # edge tangents risk
    if border > 0.18:  # tune after eyeballing
        out["suggestions"].append("Too much pull at the edges—quiet silhouettes or soften high-contrast shapes near the frame.")

    # balance nudges (neutral language)
    if abs(lr) > 0.35:
        out["suggestions"].append("Visual weight leans to one side—consider reinforcing the opposite side with a quieter counter-shape for stability.")
    if abs(tb) > 0.35:
        out["suggestions"].append("Visual weight is very top/bottom heavy—add or reduce mass to ease the tilt.")

    # summary (no thirds mention)
    cx, cy = c.get("center_of_mass", {}).get("x", 0.5), c.get("center_of_mass", {}).get("y", 0.5)


    # COLOR
    col = fb.get("color", {})
    sat = float(col.get("saturation_mean", 0.2))
    sat_lbl = _bucket(sat, [0.15, 0.35], ["muted", "natural", "vivid"])
    out["ratings"]["saturation"] = sat_lbl

    if sat_lbl == "muted":
        out["badges"].append("muted/neutral palette")
        out["suggestions"].append("If you want more color presence, add a small area of higher chroma near the focus.")
    elif sat_lbl == "vivid":
        out["badges"].append("vivid palette")
        out["suggestions"].append("Balance high-chroma accents with larger, quieter shapes to avoid noise.")

    # ---- SUMMARY  ----
    summary_value = f"This reads as **{key}** with **{contrast_lbl}** global contrast."
    summary_comp  = f" Attention is **{ent_lbl}** (focal strength {fs:.2f}); " \
                    f"balance LR {lr:+.2f}, TB {tb:+.2f}; border pull {border:.2f}; " \
                    f"COM~({cx:.2f},{cy:.2f})."
    summary_color = f" Color is **{sat_lbl}**."

    out["summary"] = summary_value + summary_comp + summary_color


    # de-dup
    seen = set(); dedup = []
    for s in out["suggestions"]:
        if s not in seen:
            dedup.append(s); seen.add(s)
    out["suggestions"] = dedup[:6]
    return out
