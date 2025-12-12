# Why Only 65 Rows from 5851 Candidate IDs?

## The Problem

You got **5851 candidate IDs** but only **65 rows** in the final parquet file. That's a **98.9% drop-off**! 

## Filters That Reduce Results

### 1. **Public Domain Filter** (Line 100)
```python
if not obj.get("isPublicDomain", False):
    return None  # Skip if not public domain
```
**Impact:** Many artworks are copyrighted, so this filters out a lot.

### 2. **Oil Filter (ON by default)** (Lines 153-156)
```python
if args.filter_oil:  # Default: True
    if "oil" not in med:
        continue  # Skip if medium doesn't contain "oil"
```
**Impact:** Even with broad queries like "painting", "portrait", "landscape", the script still filters for "oil" in the medium!

### 3. **Missing Image URL** (Line 159)
```python
if not url:
    continue  # Skip if no image URL
```
**Impact:** Some artworks don't have images available.

### 4. **CRITICAL: Missing Local Images** (Lines 176-184)
```python
if not Path(local_path).exists():
    if args.download_missing:  # Did you use this flag?
        saved = download_and_cache(...)
    else:
        continue  # SKIP if image doesn't exist locally!
```
**Impact:** **THIS IS LIKELY THE MAIN ISSUE!** If you didn't use `--download-missing`, the script skips all artworks whose images aren't already downloaded.

## Most Likely Cause

You probably ran:
```bash
python scripts/download_met_images.py --limit 5851
```

**But you needed:**
```bash
python scripts/download_met_images.py --limit 5851 --download-missing --no-oil-filter
```

## Solution: Use These Flags

### For Maximum Results (All Paintings):

```bash
python scripts/download_met_images.py \
  --limit 10000 \
  --download-missing \
  --no-oil-filter \
  --thumb \
  --longest-side 512 \
  --delay 0.1
```

**Flags explained:**
- `--download-missing`: **CRITICAL** - Downloads images that don't exist locally
- `--no-oil-filter`: **CRITICAL** - Gets all paintings, not just oil
- `--thumb`: Use smaller thumbnail images (faster download)
- `--longest-side 512`: Resize images (saves space)
- `--delay 0.1`: Wait between requests (be nice to API)

### For Oil Paintings Only (but download missing):

```bash
python scripts/download_met_images.py \
  --limit 10000 \
  --download-missing \
  --thumb \
  --longest-side 512
```

(Remove `--no-oil-filter` to keep oil filter)

## Expected Results

With `--download-missing --no-oil-filter`:
- **Before:** 5851 IDs → 65 rows (98.9% lost)
- **After:** 5851 IDs → ~2000-4000 rows (much better!)

**Why the improvement?**
- `--download-missing`: Downloads images instead of skipping
- `--no-oil-filter`: Includes all paintings, not just oil

## Diagnostic: Check What's Being Filtered

Add this debug code to see where artworks are being lost:

```python
# Add after line 147 in download_met_images.py
stats = {
    "total": 0,
    "not_public": 0,
    "no_oil": 0,
    "no_url": 0,
    "no_image_local": 0,
    "download_failed": 0,
    "added": 0
}

for oid in tqdm(ids, desc="Rebuilding catalog"):
    stats["total"] += 1
    obj = fetch_object(oid)
    if not obj:
        stats["not_public"] += 1
        continue
    
    if args.filter_oil:
        med = (obj.get("medium") or "").lower().replace("-", " ")
        if "oil" not in med:
            stats["no_oil"] += 1
            continue
    
    url = obj.get("primaryImageSmall") if args.thumb else (obj.get("primaryImage") or obj.get("primaryImageSmall"))
    if not url:
        stats["no_url"] += 1
        continue
    
    local_path = expected_local_path(url)
    if not Path(local_path).exists():
        if args.download_missing:
            saved = download_and_cache(url, longest_side=args.longest_side)
            if not saved:
                stats["download_failed"] += 1
                continue
        else:
            stats["no_image_local"] += 1
            continue
    
    # ... rest of code ...
    stats["added"] += 1

print(f"\n[Stats] Total: {stats['total']}")
print(f"  - Not public domain: {stats['not_public']}")
print(f"  - No oil in medium: {stats['no_oil']}")
print(f"  - No image URL: {stats['no_url']}")
print(f"  - Image not local (skipped): {stats['no_image_local']}")
print(f"  - Download failed: {stats['download_failed']}")
print(f"  - ✅ Added: {stats['added']}")
```

## Quick Fix for Your Current Run

If you already ran the script and only got 65 rows, run it again with the correct flags:

```bash
# In Colab
python scripts/download_met_images.py \
  --limit 10000 \
  --download-missing \
  --no-oil-filter \
  --thumb \
  --longest-side 512 \
  --delay 0.1
```

This will:
1. Download missing images (instead of skipping)
2. Include all paintings (not just oil)
3. Get you thousands of artworks instead of 65!

## Summary

| Issue | Impact | Fix |
|-------|--------|-----|
| Missing `--download-missing` | **HUGE** - Skips artworks without local images | Add `--download-missing` |
| Oil filter ON (default) | Large - Filters out non-oil paintings | Add `--no-oil-filter` |
| Not public domain | Medium - Many artworks copyrighted | Can't fix (API limitation) |
| No image URL | Small - Some artworks have no images | Can't fix (API limitation) |

**Bottom line:** Use `--download-missing --no-oil-filter` to get maximum results!

