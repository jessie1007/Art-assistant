# Understanding `--thumb` and `--longest-side` Flags

## 1. `--thumb` Flag

### What it does:
Controls which image URL to download from the Met Museum API.

### How it works:

**Without `--thumb` (default):**
```python
# Line 181: Tries full-size image first, falls back to thumbnail
url = obj.get("primaryImage") or obj.get("primaryImageSmall")
```

**With `--thumb`:**
```python
# Line 181: Uses thumbnail only
url = obj.get("primaryImageSmall")
```

### Image Sizes:

| Flag | Image Type | Typical Size | Example |
|------|------------|--------------|---------|
| **No `--thumb`** | `primaryImage` (full size) | 2000-4000px | High quality, large file |
| **`--thumb`** | `primaryImageSmall` (thumbnail) | 200-400px | Lower quality, small file |

### Example:

**Met Museum API returns:**
```json
{
  "primaryImage": "https://images.metmuseum.org/CRDImages/ad/web-large/25599.jpg",  // Full size
  "primaryImageSmall": "https://images.metmuseum.org/CRDImages/ad/web-small/25599.jpg"  // Thumbnail
}
```

- **Without `--thumb`**: Downloads `primaryImage` (full size, ~2MB)
- **With `--thumb`**: Downloads `primaryImageSmall` (thumbnail, ~50KB)

### Why use `--thumb`?

✅ **Faster downloads** - Smaller files download quicker
✅ **Less storage** - Saves disk space
✅ **Faster processing** - Smaller images process faster
❌ **Lower quality** - Less detail for similarity search

**For similarity search:** Thumbnails are usually fine (CLIP works well with smaller images)

---

## 2. `--longest-side` Flag

### What it does:
Resizes downloaded images so the **longest dimension** (width or height) is at most this many pixels.

### How it works:

```python
# Line 117: Resize image
img.thumbnail((longest_side, longest_side))
```

**What `thumbnail()` does:**
- Maintains aspect ratio (doesn't distort image)
- Resizes so **both** width and height are ≤ `longest_side`
- If image is 2000×3000px and `longest_side=512`:
  - Result: 341×512px (maintains 2:3 ratio)

### Examples:

| Original Size | `--longest-side` | Result Size | File Size |
|---------------|------------------|-------------|-----------|
| 3000×2000px | 512 | 512×341px | ~50KB |
| 3000×2000px | 1024 | 1024×683px | ~200KB |
| 3000×2000px | 256 | 256×171px | ~15KB |

### Default:
```python
ap.add_argument("--longest-side", type=int, default=512, ...)
```
Default is **512 pixels**.

### Why resize?

✅ **Saves storage** - Smaller files take less space
✅ **Faster processing** - CLIP embeddings faster on smaller images
✅ **Consistent size** - All images similar dimensions
❌ **Lower quality** - Less detail (but usually fine for ML)

**For CLIP embeddings:** 512px is usually sufficient (CLIP was trained on 224px images)

---

## How They Work Together

### Scenario 1: Full-size images, resize to 512px
```bash
python scripts/download_met_images.py --longest-side 512
```
1. Downloads `primaryImage` (full size, e.g., 3000×2000px)
2. Resizes to 512px longest side (result: 512×341px)
3. Saves as JPEG

**Result:** High quality source, but resized for storage

### Scenario 2: Thumbnails, no resize
```bash
python scripts/download_met_images.py --thumb --longest-side 2000
```
1. Downloads `primaryImageSmall` (thumbnail, e.g., 400×300px)
2. Resizes to 2000px (but image is already smaller, so no change)
3. Saves as JPEG

**Result:** Small files, fast download

### Scenario 3: Thumbnails, resize to 256px (smallest)
```bash
python scripts/download_met_images.py --thumb --longest-side 256
```
1. Downloads `primaryImageSmall` (thumbnail, e.g., 400×300px)
2. Resizes to 256px longest side (result: 256×192px)
3. Saves as JPEG

**Result:** Very small files, very fast, minimal storage

---

## Recommendations

### For Similarity Search (Your Use Case):

**Best balance:**
```bash
--thumb --longest-side 512
```

**Why:**
- Thumbnails are fast to download
- 512px is good quality for CLIP
- Small file size
- Fast embedding generation

### For Maximum Quality:

```bash
--longest-side 1024
```

**Why:**
- Higher resolution
- Better for detailed analysis
- Still reasonable file size

### For Maximum Speed/Storage:

```bash
--thumb --longest-side 256
```

**Why:**
- Smallest files
- Fastest processing
- Good enough for basic similarity

---

## Visual Comparison

```
Original: 3000×2000px (2MB)

--longest-side 512:  512×341px  (50KB)  ✅ Good for ML
--longest-side 1024: 1024×683px (200KB) ✅ Better quality
--longest-side 256:  256×171px  (15KB)  ✅ Fastest
```

---

## Summary

| Flag | Purpose | Impact |
|------|---------|--------|
| `--thumb` | Use smaller image URL from API | Faster download, less storage |
| `--longest-side N` | Resize to max N pixels | Controls final image size |

**Default behavior:**
- No `--thumb`: Downloads full-size images
- `--longest-side 512`: Resizes to 512px (default)

**Recommended for your use case:**
```bash
--thumb --longest-side 512
```

This gives you good quality for similarity search while keeping files small and downloads fast!

