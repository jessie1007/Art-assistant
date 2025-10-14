# value_blocks.py
from PIL import Image, ImageOps
import numpy as np

def big_value_blocks(img, k=5, spatial_weight=0.6, downscale=360):
    # 1) downscale
    w,h = img.size
    if max(w,h) > downscale:
        if w >= h:
            new_w, new_h = downscale, int(h*(downscale/w))
        else:
            new_h, new_w = downscale, int(w*(downscale/h))
        small = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    else:
        small = img.copy()
    # 2) luminance + spatial features
    arr = np.asarray(small.convert("RGB")).astype(np.float32)/255.0
    Y = (0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2])[...,None]
    Hh,Ww = Y.shape[:2]
    xs, ys = np.meshgrid(np.linspace(0,1,Ww), np.linspace(0,1,Hh))
    feat = np.concatenate([Y, spatial_weight*xs[...,None], spatial_weight*ys[...,None]], axis=2).reshape(-1,3)
    # 3) k-means (value-sorted init)
    percentiles = np.linspace(0,100,k+2)[1:-1]
    init_vals = np.percentile(feat[:,0], percentiles)
    centers = np.stack([feat[np.argmin(np.abs(feat[:,0]-v))] for v in init_vals], axis=0).astype(np.float32)
    for _ in range(30):
        d = ((feat[:,None,:]-centers[None,:,:])**2).sum(2)
        labels = d.argmin(1)
        for i in range(k):
            m = labels==i
            if np.any(m): centers[i] = feat[m].mean(0)
    # 4) map each cluster to its mean value (flat gray)
    meanY = np.array([feat[labels==i,0].mean() if np.any(labels==i) else 0.0 for i in range(k)])
    order = np.argsort(meanY); remap = np.zeros_like(order); remap[order]=np.arange(k)
    labels = remap[labels]; meanY = meanY[order]
    blocks_small = (meanY[labels]*255).astype(np.uint8).reshape(Hh,Ww)
    blocks = Image.fromarray(blocks_small, "L").resize(img.size, Image.Resampling.NEAREST)
    return blocks

def overlay_blocks_on(img, blocks_gray, alpha=0.35):
    overlay_rgb = ImageOps.colorize(blocks_gray, black="black", white="white")
    return Image.blend(img.convert("RGB"), overlay_rgb.convert("RGB"), alpha)
