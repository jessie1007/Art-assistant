from PIL import Image, ImageDraw
import numpy as np, cv2

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
