from PIL import Image, ImageDraw
import numpy as np, cv2
# at top of file add:
import streamlit as st


def to_gray_np(img_pil: Image.Image) -> np.ndarray:
    return np.array(img_pil.convert("L"))

def posterize_k(gray: np.ndarray, k: int):
    x = gray.reshape(-1, 1).astype(np.float32)
    k = max(2, int(k))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(x, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    centers = np.sort(centers.flatten())
    idx = np.argmin(np.abs(gray[..., None] - centers[None, None, :]), axis=2)
    poster = centers[idx].astype(np.uint8)
    return poster, centers

def flat_mid_gray(gray: np.ndarray) -> np.ndarray:
    m = int(np.round(gray.mean()))
    return np.full_like(gray, m, dtype=np.uint8)

def luminance_com(gray: np.ndarray):
    h, w = gray.shape
    Y = gray.astype(np.float32) + 1e-6
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    cx = (x_coords * Y).sum() / Y.sum()
    cy = (y_coords * Y).sum() / Y.sum()
    return float(cx / w), float(cy / h)

def thirds_overlay(img: Image.Image, com_xy):
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for x in (w/3, 2*w/3):
        draw.line([(x, 0), (x, h)], fill=(200, 200, 200), width=1)
    for y in (h/3, 2*h/3):
        draw.line([(0, y), (w, y)], fill=(200, 200, 200), width=1)
    cx, cy = com_xy
    px, py = int(cx*w), int(cy*h)
    r = max(3, int(0.01*min(w, h)))
    draw.ellipse([(px-r, py-r), (px+r, py+r)], fill=(255, 0, 0))
    return img

def value_percentages(poster: np.ndarray, centers: np.ndarray):
    idx = np.argmin(np.abs(poster[..., None] - centers[None, None, :]), axis=2)
    counts = np.bincount(idx.ravel(), minlength=len(centers)).astype(np.float32)
    pct = counts / counts.sum()
    return pct  # array length k, sums to 1.0


# ================================
# Public UI: render_value_tab
# ================================
def render_value_tab(img, k_values: int = 5, show_grid: bool = True):
    """Value studies tab. Uses the global image passed from the main app."""
    st.header("🧪 Value Studies")

    if img is None:
        st.info("Upload an image above to analyze values.")
        return

    # 1) grayscale + posterization
    gray = to_gray_np(img)                           # uint8 (H, W)
    poster, centers = posterize_k(gray, k_values)    # uint8 (H, W), sorted centers

    # 2) simple metrics
    g = gray.astype("float32") / 255.0
    mean_val = float(g.mean())
    contrast = float(g.std())
    dark_pct  = float((g < 0.20).mean())   # < 20% as "dark"
    light_pct = float((g > 0.80).mean())   # > 80% as "light"
    com_xy = luminance_com(gray)           # (cx, cy) in 0..1

    # 3) optional thirds overlay preview
    preview = thirds_overlay(img, com_xy) if show_grid else img

    # 4) layout
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.subheader("Original")
        st.image(preview, use_container_width=True)
        st.caption(f"COM ≈ ({com_xy[0]:.2f}, {com_xy[1]:.2f})")

    with c2:
        st.subheader("Grayscale")
        st.image(Image.fromarray(gray, mode="L"), use_container_width=True)

    with c3:
        st.subheader(f"Posterized (k={k_values})")
        st.image(Image.fromarray(poster, mode="L"), use_container_width=True)
        st.caption("Centers (0–255): " + ", ".join(str(int(c)) for c in centers))

    # 5) value distribution summary
    pct = value_percentages(poster, centers)  # len=k, sums to 1.0
    st.write("### Value distribution")
    st.write(", ".join(f"Zone {i}: {p*100:.1f}%" for i, p in enumerate(pct)))

    # 6) quick metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean (0–1)", f"{mean_val:.2f}")
    m2.metric("Contrast (σ)", f"{contrast:.2f}")
    m3.metric("Dark %", f"{dark_pct*100:.1f}%")
    m4.metric("Light %", f"{light_pct*100:.1f}%")
