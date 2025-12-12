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
  python scripts/rebuild_met_catalog.py --limit 10000  --longest-side 512
  python scripts/rebuild_met_catalog.py --download-missing --limit 20000
"""

import os, io, time, argparse, hashlib
from pathlib import Path
from typing import Optional, Dict, List, Iterable

import requests
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


HEADERS = {"User-Agent": "ArtAssistant/1.0 (contact: you@example.com)"}
YEAR_RE = re.compile(r"\b(\d{4})\b")

def make_session(max_retries: int = 5, backoff: float = 0.5) -> requests.Session:
    retry = Retry(
        total=max_retries, connect=max_retries, read=max_retries, status=max_retries,
        backoff_factor=backoff, status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]), raise_on_status=False,
    )
    s = requests.Session()
    s.headers.update(HEADERS)
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

SESSION = make_session()

def extract_year(object_date: str) -> Optional[int]:
    if not isinstance(object_date, str): return None
    m = YEAR_RE.search(object_date)
    return int(m.group(1)) if m else None

def to_century_bin(y: Optional[int]) -> str:
    if y is None: return "Unknown"
    return f"{(y // 100) * 100}s"


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
    r = SESSION.get(MET_SEARCH, params=params, timeout=30)
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

def fetch_object(obj_id: int, require_public_domain: bool = False) -> Optional[Dict]:
    try:
        r = SESSION.get(MET_OBJECT.format(objectID=obj_id), timeout=30)
        if r.status_code != 200:
            return None
        obj = r.json()
        if require_public_domain and not obj.get("isPublicDomain", False):
            return None
        return obj
    except Exception:
        return None

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
    ap.add_argument("--longest-side", type=int, default=512, help="resize longest side if downloading")
    ap.add_argument("--download-missing", action="store_true", help="download any images not already cached")
    ap.add_argument("--queries", nargs="*", default=["oil", "oil painting", "oil on canvas", "oil on wood","stilllife","painting", "portrait","landscape"])
    ap.add_argument("--thumb", action="store_true", help="use thumbnail images (primaryImageSmall) instead of full size")
    ap.add_argument("--public-domain-only", action="store_true",
                help="only include public domain artworks (default: include all)")
    ap.add_argument("--no-oil-filter", dest="filter_oil", action="store_false", default=True,
                help="skip filtering for 'oil' in medium (get all paintings)")
    ap.add_argument("--metadata-only", dest="metadata_only", action="store_true",
                help="build catalog from URLs only, skip downloads")
    ap.add_argument("--min-year", type=int, default=None, help="keep items with parsed year >= this")
    ap.add_argument("--max-year", type=int, default=None, help="keep items with parsed year <= this")
    ap.add_argument("--balance-by-century", action="store_true", help="after collection, sample up to --per-century per century bin")
    ap.add_argument("--per-century", type=int, default=200, help="target items per century bin when balancing")
    args = ap.parse_args()
    
    ensure_dirs()

    ids = fetch_ids_union(args.queries, limit=args.limit)
    print(f"[info] candidate painting IDs: {len(ids)} from queries={args.queries}")

    rows = []
    # Debug statistics
    stats = {
        "total": 0,
        "fetch_failed": 0,
        "not_public_domain": 0,
        "oil_filtered": 0,
        "no_image_url": 0,
        "year_filtered": 0,
        "download_failed": 0,
        "image_not_local_skipped": 0,
        "added": 0
    }
    
    for oid in tqdm(ids, desc="Rebuilding catalog"):
        stats["total"] += 1
        obj = fetch_object(oid, require_public_domain=args.public_domain_only)
        if not obj:
            stats["fetch_failed"] += 1
            time.sleep(args.delay); continue
        
        # Check public domain if filter is on
        if args.public_domain_only and not obj.get("isPublicDomain", False):
            stats["not_public_domain"] += 1
            time.sleep(args.delay); continue

        # Optional: filter for oil paintings only (can be disabled with --no-oil-filter)
        if args.filter_oil:
            med = (obj.get("medium") or "").lower().replace("-", " ")
            if "oil" not in med:
                stats["oil_filtered"] += 1
                time.sleep(args.delay); continue

        url = obj.get("primaryImageSmall") if args.thumb else (obj.get("primaryImage") or obj.get("primaryImageSmall"))
        if not url:
            stats["no_image_url"] += 1
            time.sleep(args.delay); continue

        # Parse year & century bin for diversity
        y = extract_year(obj.get("objectDate"))
        century = to_century_bin(y)

        # Optional year bounds filter
        if args.min_year is not None and (y is None or y < args.min_year):
            stats["year_filtered"] += 1
            time.sleep(args.delay); continue
        if args.max_year is not None and (y is None or y > args.max_year):
            stats["year_filtered"] += 1
            time.sleep(args.delay); continue

        if args.metadata_only:
            local_path = None
        else:
            local_path = expected_local_path(url)
            if not Path(local_path).exists():
                if args.download_missing:
                    saved = download_and_cache(url, longest_side=args.longest_side)
                    if not saved:
                        stats["download_failed"] += 1
                        time.sleep(args.delay)
                        continue
                else:
                    stats["image_not_local_skipped"] += 1
                    time.sleep(args.delay)
                    continue

        rows.append({
            "source": "met",
            "source_id": obj.get("objectID"),
            "title": obj.get("title"),
            "artist": obj.get("artistDisplayName"),
            "year": obj.get("objectDate"),
            "numeric_year": y,  # Add parsed year
            "century_bin": century,  # Add century bin
            "medium": obj.get("medium"),
            "objectName": obj.get("objectName"),
            "object_url": obj.get("objectURL"),
            "image_url": url,
            "department": obj.get("department"),
            "culture": obj.get("culture"),
            **({"local_path": local_path} if local_path else {}),
        })
        stats["added"] += 1

        time.sleep(args.delay)
    
    # Print debug statistics
    print(f"\n[Stats] Processing summary:")
    print(f"  Total IDs processed: {stats['total']}")
    print(f"  ✅ Added to catalog: {stats['added']}")
    print(f"  ❌ Filtered out:")
    print(f"     - Fetch failed (API error): {stats['fetch_failed']}")
    if args.public_domain_only:
        print(f"     - Not public domain: {stats['not_public_domain']}")
    if args.filter_oil:
        print(f"     - No oil in medium: {stats['oil_filtered']}")
    print(f"     - No image URL: {stats['no_image_url']}")
    if args.min_year or args.max_year:
        print(f"     - Year out of range: {stats['year_filtered']}")
    if not args.metadata_only:
        if args.download_missing:
            print(f"     - Image download failed: {stats['download_failed']}")
        else:
            print(f"     - Image not local (skipped): {stats['image_not_local_skipped']}")

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
    
    # --- normalize dtypes before writing parquet ---


    # Numeric (nullable) columns
    for col in ["source_id", "numeric_year"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")

    # String-like columns
    for col in [
        "source","title","artist","year","medium","objectName",
        "object_url","image_url","department","culture","century_bin","local_path"
    ]:
        if col in merged.columns:
            merged[col] = merged[col].astype("string")

    # Optional: ensure ordering (nice to have)
    ordered = [
        "source","source_id","title","artist","year","numeric_year","century_bin",
        "department","culture","medium","objectName","object_url","image_url","local_path"
    ]
    merged = merged[[c for c in ordered if c in merged.columns] + [c for c in merged.columns if c not in ordered]]
    # --- end normalize ---

    merged.to_parquet(PARQUET, index=False)
    print(f"[done] {len(merged)} rows -> {PARQUET}")

if __name__ == "__main__":
    main()
