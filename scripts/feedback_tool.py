# scripts/feedback_tools.py
import cv2, numpy as np
from skimage.color import rgb2lab

def value_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    hist, _ = np.histogram(gray, bins=11, range=(0, 1), density=True)
    return {
        "mean": float(gray.mean()),
        "contrast": float(gray.std()),
        "dark_pct": float((gray < 0.25).mean()),
        "light_pct": float((gray > 0.75).mean()),
        "zone_hist": hist.tolist(),
    }

def composition_features(bgr: np.ndarray) -> dict:
    """
    Compute simple composition metrics from a BGR image.
    Returns:
      - entropy: saliency entropy (higher = more spread-out attention)
      - center_of_mass: (cx, cy) in [0,1] coords
      - thirds_score: closeness of COM to rule-of-thirds points (0..1)
    """
    # --- get a 2D saliency map in [0,1] ---
    # If you implemented saliency_map() already, use it; else a gradient fallback:
    try:
        _sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, sal = _sal.computeSaliency(bgr)
        if not ok: raise RuntimeError
        sal = cv2.normalize(sal, None, 0, 1, cv2.NORM_MINMAX).astype("float32")
    except Exception:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
        sal = cv2.magnitude(gx, gy)
        sal = cv2.normalize(sal, None, 0, 1, cv2.NORM_MINMAX).astype("float32")

    H, W = sal.shape[:2]
    sal = sal.reshape(H, W).astype("float32")
    sal = np.clip(sal, 0.0, 1.0)

    # --- entropy over normalized saliency ---
    p = sal / (sal.sum() + 1e-12)                     # sum to 1
    entropy = float(-(p * np.log2(p + 1e-12)).sum())  # scalar

    # --- center of mass (normalized 0..1) ---
    ys, xs = np.mgrid[0:H, 0:W]
    total = p.sum() + 1e-12
    cx = float((xs * p).sum() / total) / W
    cy = float((ys * p).sum() / total) / H

    # --- rule-of-thirds proximity score (0..1, higher is closer) ---
    thirds = np.array([[1/3,1/3],[2/3,1/3],[1/3,2/3],[2/3,2/3]], dtype=np.float32)
    dmin = float(np.sqrt(((thirds - np.array([cx, cy]))**2).sum(axis=1)).min())
    thirds_score = float(1.0 - np.clip(dmin / 0.25, 0.0, 1.0))

    return {
        "entropy": entropy,
        "center_of_mass": {"x": cx, "y": cy},
        "thirds_score": thirds_score,
    }

def color_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].mean() / 255.0
    return {
        "saturation_mean": float(sat),
        "dominant_hues": ["#aaaaaa"],  # placeholder for palette chips
        "harmony": "analogous-ish"
    }

def make_feedback(img_rgb):
    """img_rgb: np.array RGB"""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return {
        "value": value_features(bgr),
        "composition": composition_features(bgr),
        "color": color_features(bgr),
        "comment": "Automated critique from CV features."
    }


from typing import Dict, List
import numpy as np

def _bucket(x, edges, labels):
    """Place x into a bucket defined by edges (ascending)."""
    idx = np.digitize([x], edges)[0]
    idx = max(0, min(idx, len(labels)-1))
    return labels[idx]

def interpret_feedback(fb: Dict) -> Dict:
    """
    Turn numeric features into human-readable summary + suggestions.
    fb: dict returned by make_feedback(...)
    Returns: {
      "summary": str,
      "badges": List[str],
      "suggestions": List[str],
      "ratings": {"contrast":"low|moderate|high", "saturation": "...", ...}
    }
    """
    out = {"summary": "", "badges": [], "suggestions": [], "ratings": {}}

    # --- VALUE ---
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

    # --- COMPOSITION ---
    c = fb.get("composition", {})
    entropy = float(c.get("entropy", 16.0))
    com = c.get("center_of_mass", {"x":0.5,"y":0.5})
    cx, cy = float(com.get("x", 0.5)), float(com.get("y", 0.5))
    thirds = float(c.get("thirds_score", 0.5))

    ent_lbl = _bucket(entropy, [12.0, 18.0], ["focused", "balanced", "diffuse"])
    thirds_lbl = _bucket(thirds, [0.4, 0.7], ["off-thirds", "near-thirds", "on-thirds"])

    out["ratings"]["attention"] = ent_lbl
    out["ratings"]["thirds"] = thirds_lbl

    if ent_lbl == "diffuse":
        out["suggestions"].append("Reduce detail/contrast in non-focal areas to tighten attention.")
    if thirds_lbl == "off-thirds":
        out["suggestions"].append("Shift or crop so the focal mass sits closer to a rule-of-thirds hotspot.")
    # gentle guidance on COM drift
    if abs(cx-0.5) < 0.05 and abs(cy-0.5) < 0.05:
        out["suggestions"].append("Current focus sits near center—consider a slight offset for more dynamism.")

    # --- COLOR ---
    col = fb.get("color", {})
    sat = float(col.get("saturation_mean", 0.2))
    dom = col.get("dominant_hues", []) or []

    sat_lbl = _bucket(sat, [0.15, 0.35], ["muted", "natural", "vivid"])
    out["ratings"]["saturation"] = sat_lbl

    # Clean up 'neutral' detection
    if sat_lbl == "muted":
        palette_note = "neutral/achromatic" if (len(dom)==1 and dom[0].lower()=="#aaaaaa") else "muted palette"
        out["badges"].append(palette_note)
        out["suggestions"].append("If you want more color presence, add a small area of higher chroma near the focus.")
    elif sat_lbl == "vivid":
        out["badges"].append("vivid palette")
        out["suggestions"].append("Balance high-chroma accents with larger, quieter shapes to avoid noise.")

    # --- Compose summary text ---
    out["summary"] = (
        f"This reads as **{key}** with **{contrast_lbl}** global contrast. "
        f"Attention feels **{ent_lbl}** (COM at ~({cx:.2f},{cy:.2f}), {thirds_lbl}). "
        f"Color is **{sat_lbl}**."
    )

    # De-duplicate suggestions while preserving order
    seen = set(); dedup = []
    for s in out["suggestions"]:
        if s not in seen:
            dedup.append(s); seen.add(s)
    out["suggestions"] = dedup[:6]  # keep it punchy
    return out
