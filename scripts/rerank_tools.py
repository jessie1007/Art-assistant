# scripts/rerank_tools.py
import numpy as np
import cv2

def hue_hist(img_bgr, bins=36):
    """Hue histogram (normalized)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].ravel()
    hist = np.bincount((h * (bins / 180)).astype(int), minlength=bins).astype(float)
    return hist / (hist.sum() + 1e-9)

def texture_vec(img_bgr, bins=12):
    """Texture orientation histogram using Sobel gradients."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi) % (2 * np.pi)
    t = np.bincount(
        (ang * (bins / (2 * np.pi))).astype(int).ravel(),
        weights=mag.ravel(),
        minlength=bins
    )
    return t / (t.sum() + 1e-9)

def _entropy(p):
    p = np.clip(p, 1e-9, 1.0); p /= p.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))

def query_reliabilities(q_h, q_t, sat_mean, grad_mag_mean):
    """Adaptive gates for palette and texture usefulness."""
    H_norm = _entropy(q_h)                 # high entropy = no dominant hue
    kappa = float((q_t.max() - q_t.mean()) / (q_t.std() + 1e-9))  # peakiness of texture
    r_pal = np.clip(sat_mean * (1 - H_norm), 0.0, 1.0)
    r_tex = np.clip(grad_mag_mean * np.tanh(kappa), 0.0, 1.0)
    return r_pal, r_tex

def combine_scores(q_feats, candidates):
    """
    Re-rank candidates adaptively:
    q_feats: {"h_hist","t_vec","sat_mean","grad_mag_mean"}
    candidates: list of {"clip_sim","h_hist","t_vec"}
    """
    clip_s = np.array([c["clip_sim"] for c in candidates])
    pal_s  = np.array([float(q_feats["h_hist"] @ c["h_hist"]) for c in candidates])
    tex_s  = np.array([float(q_feats["t_vec"]  @ c["t_vec"])  for c in candidates])

    # z-normalize each sim so scales are comparable
    def z(x): return (x - x.mean()) / (x.std() + 1e-9)
    clip_z, pal_z, tex_z = z(clip_s), z(pal_s), z(tex_s)

    r_pal, r_tex = query_reliabilities(q_feats["h_hist"], q_feats["t_vec"],
                                       q_feats["sat_mean"], q_feats["grad_mag_mean"])

    w_clip = 0.6
    w_pal  = 0.35 * r_pal
    w_tex  = 0.25 * r_tex
    Z = w_clip + w_pal + w_tex
    w_clip, w_pal, w_tex = w_clip/Z, w_pal/Z, w_tex/Z

    final = w_clip*clip_z + w_pal*pal_z + w_tex*tex_z
    return final
