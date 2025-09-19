#!/usr/bin/env python3
"""
Rebuild data/corpus/met_oil_paintings.parquet from already-downloaded images.

Strategy:
- Query Met API (paintings-only) with several oil-related queries to get objectIDs
- For each object, check public domain + medium contains "oil"
- Compute expected cache filename from image URL (SHA1)
- If file exists in data/images/, add metadata row (no download)
- Optionally --download-missing to fetch any that aren't cached yet
- Append/merge into parquet (dedupe by source_id)

Usage:
  python scripts/rebuild_met_catalog.py --limit 10000 --thumb --longest-side 512
  python scripts/rebuild_met_catalog.py --download-missing --limit 20000 --thumb
"""

import os, io, time, argparse, hashlib
from pathlib import Path
from typing import Optional, Dict, List, Iterable

import requests
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

MET_SEARCH = "https://collectionapi.metmuseum.org/public/collection/v1/search"
MET_OBJECT = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{objectID}"

OUT_DIR = Path("data")
IMG_DIR = OUT_DIR / "images"
PARQUET = OUT_DIR / "corpus" / "met_oil_paintings.parquet"

def ensure_dirs():
    (OUT_DIR / "corpus").mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

def _safe_name(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

def expected_local_path(url: str) -> str:
    return str((IMG_DIR / (_safe_name(url) + ".jpg")).as_posix())

def search_ids_for_query(q: str) -> List[int]:
    # paintings only, has images
    params = {"q": q, "hasImages": "true", "medium": "Paintings"}
    r = requests.get(MET_SEARCH, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("objectIDs") or []

def fetch_ids_union(queries: Iterable[str], limit: Optional[int] = None) -> List[int]:
    seen = set()
    for q in queries:
        for oid in search_ids_for_query(q):
            seen.add(oid)
    ids = sorted(seen)
    if limit:
        ids = ids[:limit]
    return ids

def fetch_object(obj_id: int) -> Optional[Dict]:
    r = requests.get(MET_OBJECT.format(objectID=obj_id), timeout=30)
    if r.status_code != 200:
        return None
    obj = r.json()
    if not obj.get("isPublicDomain", False):
        return None
    return obj

def download_and_cache(url: str, longest_side: int = 512) -> Optional[str]:
    out_path = expected_local_path(url)
    p = Path(out_path)
    if p.exists():
        return out_path
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        from PIL import Image  # local import to avoid top-level heavy import on dry runs
        img = Image.open(io.BytesIO(r.content))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((longest_side, longest_side))
        img.save(p, format="JPEG", quality=90, optimize=True)
        return out_path
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap candidate IDs after union")
    ap.add_argument("--delay", type=float, default=0.03, help="sleep between requests (sec)")
    ap.add_argument("--thumb", action="store_true", help="prefer primaryImageSmall")
    ap.add_argument("--longest-side", type=int, default=512, help="resize longest side if downloading")
    ap.add_argument("--download-missing", action="store_true", help="download any images not already cached")
    ap.add_argument("--queries", nargs="*", default=["oil", "oil painting", "oil on canvas", "oil on wood"])
    args = ap.parse_args()

    ensure_dirs()

    ids = fetch_ids_union(args.queries, limit=args.limit)
    print(f"[info] candidate painting IDs: {len(ids)} from queries={args.queries}")

    rows = []
    for oid in tqdm(ids, desc="Rebuilding catalog"):
        obj = fetch_object(oid)
        if not obj:
            time.sleep(args.delay); continue

        med = (obj.get("medium") or "").lower().replace("-", " ")
        if "oil" not in med:
            time.sleep(args.delay); continue

        url = obj.get("primaryImageSmall") if args.thumb else (obj.get("primaryImage") or obj.get("primaryImageSmall"))
        if not url:
            time.sleep(args.delay); continue

        # compute expected local path and either use it or (optionally) download
        local_path = expected_local_path(url)
        if not Path(local_path).exists():
            if args.download_missing:
                saved = download_and_cache(url, longest_side=args.longest_side)
                if not saved:
                    time.sleep(args.delay)
                    continue
            else:
                # skip if not in cache and not downloading
                time.sleep(args.delay)
                continue


        rows.append({
            "source": "met",
            "source_id": obj.get("objectID"),
            "title": obj.get("title"),
            "artist": obj.get("artistDisplayName"),
            "year": obj.get("objectDate"),
            "medium": obj.get("medium"),
            "objectName": obj.get("objectName"),
            "object_url": obj.get("objectURL"),
            "image_url": url,
            "local_path": local_path,
        })

        time.sleep(args.delay)

    new_df = pd.DataFrame(rows)

    # Append/merge with existing parquet
    if os.path.exists(PARQUET) and len(new_df) > 0:
        try:
            old_df = pd.read_parquet(PARQUET)
            merged = pd.concat([old_df, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["source_id"])
        except Exception:
            merged = new_df
    else:
        merged = new_df

    merged.to_parquet(PARQUET, index=False)
    print(f"[done] {len(merged)} rows -> {PARQUET}")

if __name__ == "__main__":
    main()
