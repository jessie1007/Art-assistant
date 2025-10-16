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
    Simple composition proxies:
      - entropy: saliency entropy (higher = more spread-out attention)
      - center_of_mass: (cx, cy) normalized to [0,1]
      - thirds_score: proximity to thirds points (metric only; no tips emitted)
    """
    try:
        _sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, sal = _sal.computeSaliency(bgr)
        if not ok:
            raise RuntimeError
        sal = cv2.normalize(sal, None, 0, 1, cv2.NORM_MINMAX).astype("float32")
    except Exception:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
        sal = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 1, cv2.NORM_MINMAX).astype("float32")

    H, W = sal.shape[:2]
    p = sal / (sal.sum() + 1e-12)  # sum to 1
    ys, xs = np.mgrid[0:H, 0:W]
    cx = float((xs * p).sum()) / W
    cy = float((ys * p).sum()) / H

    thirds = np.array([[1/3, 1/3], [2/3, 1/3], [1/3, 2/3], [2/3, 2/3]], dtype=np.float32)
    dmin = float(np.sqrt(((thirds - np.array([cx, cy]))**2).sum(axis=1)).min())
    thirds_score = float(1.0 - np.clip(dmin / 0.25, 0.0, 1.0))

    return {
        "entropy": float(-(p * np.log2(p + 1e-12)).sum()),
        "center_of_mass": {"x": cx, "y": cy},
        "thirds_score": thirds_score,  # metric only
    }

# ---------- COLOR ----------
def color_features(img_bgr: np.ndarray) -> dict:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].mean() / 255.0
    return {
        "saturation_mean": float(sat),
        "dominant_hues": ["#aaaaaa"],  # placeholder
        "harmony": "analogous-ish",
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
    c = fb.get("composition", {})
    entropy = float(c.get("entropy", 16.0))
    com = c.get("center_of_mass", {"x": 0.5, "y": 0.5})
    cx, cy = float(com.get("x", 0.5)), float(com.get("y", 0.5))
    ent_lbl = _bucket(entropy, [12.0, 18.0], ["focused", "balanced", "diffuse"])
    out["ratings"]["attention"] = ent_lbl
    out["ratings"]["focal_xy"] = f"({cx:.2f},{cy:.2f})"

    if ent_lbl == "diffuse":
        out["suggestions"].append("Reduce detail/contrast in non-focal areas to tighten attention.")
    if abs(cx - 0.5) < 0.05 and abs(cy - 0.5) < 0.05:
        out["suggestions"].append("Current focus sits near center—consider a slight offset for more dynamism.")

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

    out["summary"] = (
        f"This reads as **{key}** with **{contrast_lbl}** global contrast. "
        f"Attention feels **{ent_lbl}** (COM~({cx:.2f},{cy:.2f})). "
        f"Color is **{sat_lbl}**."
    )

    # de-dup
    seen = set(); dedup = []
    for s in out["suggestions"]:
        if s not in seen:
            dedup.append(s); seen.add(s)
    out["suggestions"] = dedup[:6]
    return out
