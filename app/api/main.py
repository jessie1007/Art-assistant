"""
FastAPI server for Art Assistant image search.

Endpoints:
- POST /search: Upload image, get similar artworks (with caching)
- POST /search/batch: Upload multiple images, get results for all
- GET /health: Health check
- GET /artwork/{id}: Get artwork metadata by index ID

Features:
- Rate limiting: Prevents abuse (default: 30 requests/minute per IP)
- Caching: Stores results for identical images (saves compute)
- Batch search: Process multiple images efficiently
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import hashlib
import time
from functools import lru_cache
from collections import OrderedDict

# Add project root to path (go up 3 levels: app/api/main.py -> root)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from PIL import Image
import numpy as np
import faiss
import io

# Import project modules
from scripts.hf_embed_global import embed_image
from scripts.faiss_utils import load_index, load_meta, id_to_meta

# Initialize FastAPI app
app = FastAPI(
    title="Art Assistant API",
    description="Image similarity search API using CLIP embeddings and FAISS",
    version="1.0.0"
)

# Rate limiting: 30 requests per minute per IP (adjustable via env var)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default paths (can be overridden via environment variables)
# This allows artifacts to be loaded from external storage (S3, persistent disk, etc.)
# without needing to rebuild them locally or include them in the Docker image
DEFAULT_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", str(ROOT / "data/index_samples/index.faiss")))
DEFAULT_META_PATH = Path(os.getenv("FAISS_META_PATH", str(ROOT / "data/index_samples/meta.npy")))

# Global variables for loaded index and metadata
_index = None
_metadata = None

# ============================================================================
# CACHING: Why it matters
# ============================================================================
# Even if different users upload different images, caching helps because:
# 1. Same user might re-upload the same image (testing, refinement)
# 2. Similar images might have identical embeddings (exact duplicates)
# 3. Reduces server load: embedding generation is expensive (CLIP model)
# 4. Faster response times for cached queries
#
# We use image hash (SHA256) as cache key. LRU cache with max 100 entries.
# ============================================================================

# Simple in-memory cache: {image_hash: (results, timestamp)}
# Using OrderedDict for LRU eviction
_search_cache: OrderedDict[str, tuple[Dict[str, Any], float]] = OrderedDict()
CACHE_MAX_SIZE = 100  # Max cached results
CACHE_TTL = 3600  # Cache expires after 1 hour (3600 seconds)


def get_image_hash(image_bytes: bytes) -> str:
    """Generate SHA256 hash of image bytes for caching."""
    return hashlib.sha256(image_bytes).hexdigest()


def get_cached_result(image_hash: str) -> Optional[Dict[str, Any]]:
    """Get cached search result if available and not expired."""
    if image_hash not in _search_cache:
        return None
    
    result, timestamp = _search_cache[image_hash]
    
    # Check if cache expired
    if time.time() - timestamp > CACHE_TTL:
        del _search_cache[image_hash]
        return None
    
    # Move to end (LRU: most recently used)
    _search_cache.move_to_end(image_hash)
    return result


def cache_result(image_hash: str, result: Dict[str, Any]):
    """Cache search result with LRU eviction."""
    # If cache is full, remove oldest entry
    if len(_search_cache) >= CACHE_MAX_SIZE:
        _search_cache.popitem(last=False)  # Remove oldest
    
    _search_cache[image_hash] = (result, time.time())


def load_index_and_metadata():
    """Lazy load FAISS index and metadata."""
    global _index, _metadata
    if _index is None:
        if not DEFAULT_INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found: {DEFAULT_INDEX_PATH}")
        _index = load_index(DEFAULT_INDEX_PATH)
        print(f"[API] Loaded FAISS index with {_index.ntotal} vectors")
    
    if _metadata is None:
        if DEFAULT_META_PATH.exists():
            _metadata = load_meta(DEFAULT_META_PATH)
            print(f"[API] Loaded metadata for {len(_metadata)} artworks")
        else:
            # Fallback: try to load from JSONL
            jsonl_path = ROOT / "data/embeddings_samples/artwork_index.jsonl"
            if jsonl_path.exists():
                _metadata = {}
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        _metadata[i] = json.loads(line)
                print(f"[API] Loaded metadata from JSONL for {len(_metadata)} artworks")
            else:
                _metadata = {}
                print("[API] Warning: No metadata found, using empty dict")
    
    return _index, _metadata


@app.on_event("startup")
async def startup_event():
    """Load index and metadata on startup."""
    try:
        load_index_and_metadata()
    except Exception as e:
        print(f"[API] Warning: Could not load index on startup: {e}")
        print("[API] Index will be loaded on first request")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    index, meta = load_index_and_metadata()
    return {
        "status": "healthy",
        "index_size": index.ntotal if index else 0,
        "metadata_count": len(meta) if meta else 0,
        "cache_size": len(_search_cache),
        "cache_max": CACHE_MAX_SIZE
    }


@app.get("/artwork/{artwork_id}")
async def get_artwork(artwork_id: int):
    """Get artwork metadata by index ID."""
    _, metadata = load_index_and_metadata()
    
    if artwork_id not in metadata:
        raise HTTPException(status_code=404, detail=f"Artwork ID {artwork_id} not found")
    
    artwork = metadata[artwork_id].copy()
    
    # Add image URL/path if available
    local_path = artwork.get("local_path")
    if local_path and Path(local_path).exists():
        artwork["image_available"] = True
    else:
        artwork["image_available"] = False
        # Use image_url if local_path doesn't exist
        if artwork.get("image_url"):
            artwork["image_url"] = artwork.get("image_url")
    
    return artwork


def _perform_search(img: Image.Image, topk: int) -> Dict[str, Any]:
    """
    Internal function to perform search (used by both single and batch endpoints).
    Returns results dictionary.
    """
    # Load index and metadata
    index, metadata = load_index_and_metadata()
    
    if index.ntotal == 0:
        raise HTTPException(status_code=503, detail="Index is empty")
    
    # Generate embedding
    try:
        query_vec = embed_image(img)  # (D,) float32, L2-normalized
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")
    
    # Search FAISS index
    query_vec = query_vec.reshape(1, -1)  # (1, D)
    faiss.normalize_L2(query_vec)  # Ensure normalized for cosine similarity
    
    scores, ids = index.search(query_vec, min(topk, index.ntotal))
    
    # Build response
    results = []
    for i, (artwork_id, score) in enumerate(zip(ids[0], scores[0])):
        artwork_id = int(artwork_id)
        if artwork_id < 0 or artwork_id >= index.ntotal:
            continue
        
        artwork = metadata.get(artwork_id, {}).copy()
        artwork["id"] = artwork_id
        artwork["similarity_score"] = float(score)
        artwork["rank"] = i + 1
        
        # Check if image file exists (handle both absolute and relative paths)
        local_path = artwork.get("local_path")
        if local_path:
            # Try absolute path first, then relative to ROOT
            if Path(local_path).exists():
                artwork["image_available"] = True
            elif (ROOT / local_path).exists():
                artwork["image_available"] = True
                artwork["local_path"] = str(ROOT / local_path)  # Update to absolute
            else:
                artwork["image_available"] = False
        else:
            artwork["image_available"] = False
        
        results.append(artwork)
    
    return {
        "query_image_size": img.size,
        "topk": len(results),
        "results": results
    }


@app.post("/search")
@limiter.limit("30/minute")  # Rate limit: 30 requests per minute per IP
async def search_similar(
    request: Request,
    file: UploadFile = File(...),
    topk: int = 5
):
    """
    Search for similar artworks by uploading an image.
    
    Features:
    - Rate limiting: 30 requests/minute per IP (prevents abuse)
    - Caching: Identical images return cached results (faster, saves compute)
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        topk: Number of similar artworks to return (default: 5, max: 50)
    
    Returns:
        Dictionary with query_image_size, topk, results, and cache_hit flag
    """
    if topk < 1 or topk > 50:
        raise HTTPException(status_code=400, detail="topk must be between 1 and 50")
    
    # Read image bytes
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    # Check cache
    image_hash = get_image_hash(contents)
    cached_result = get_cached_result(image_hash)
    
    if cached_result is not None:
        # Return cached result (but update topk if different)
        result = cached_result.copy()
        if result["topk"] != topk:
            # Need to re-search with different topk
            result = _perform_search(img, topk)
            cache_result(image_hash, result)
        else:
            result["cache_hit"] = True
            return result
    
    # Cache miss: perform search
    result = _perform_search(img, topk)
    result["cache_hit"] = False
    
    # Cache the result
    cache_result(image_hash, result)
    
    return result


@app.post("/search/batch")
@limiter.limit("10/minute")  # Stricter limit for batch (more expensive)
async def search_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    topk: int = 5
):
    """
    Batch search: Process multiple images in one request.
    
    Why batch search?
    - More efficient: Single HTTP request instead of N requests
    - Better for bulk processing: Upload gallery, get all results at once
    - Reduced overhead: Less network round-trips
    - Atomic operation: All succeed or all fail (easier error handling)
    
    Args:
        files: List of image files (max 10 per request)
        topk: Number of similar artworks per image (default: 5, max: 50)
    
    Returns:
        List of results, one per image (in same order as input)
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch request")
    
    if topk < 1 or topk > 50:
        raise HTTPException(status_code=400, detail="topk must be between 1 and 50")
    
    results = []
    
    for idx, file in enumerate(files):
        try:
            # Read and process image
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            
            # Check cache for this image
            image_hash = get_image_hash(contents)
            cached_result = get_cached_result(image_hash)
            
            if cached_result is not None and cached_result["topk"] == topk:
                result = cached_result.copy()
                result["cache_hit"] = True
                result["image_index"] = idx
                result["filename"] = file.filename
            else:
                # Perform search
                result = _perform_search(img, topk)
                result["cache_hit"] = False
                result["image_index"] = idx
                result["filename"] = file.filename
                
                # Cache the result
                cache_result(image_hash, result)
            
            results.append(result)
        
        except Exception as e:
            # Include error in results instead of failing entire batch
            results.append({
                "image_index": idx,
                "filename": file.filename if file else "unknown",
                "error": str(e),
                "success": False
            })
    
    return {
        "total_images": len(files),
        "successful": sum(1 for r in results if r.get("success", True)),
        "results": results
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Art Assistant API",
        "version": "1.0.0",
        "features": {
            "rate_limiting": "30 requests/minute per IP (configurable)",
            "caching": "LRU cache for identical images (100 entries, 1hr TTL)",
            "batch_search": "Process up to 10 images in one request"
        },
        "endpoints": {
            "search": "POST /search - Upload image to find similar artworks (with caching)",
            "search_batch": "POST /search/batch - Upload multiple images (max 10)",
            "health": "GET /health - Health check",
            "artwork": "GET /artwork/{id} - Get artwork metadata by ID"
        },
        "cache_stats": {
            "cached_entries": len(_search_cache),
            "max_size": CACHE_MAX_SIZE,
            "ttl_seconds": CACHE_TTL
        }
    }


@app.get("/favicon.ico")
async def favicon():
    """Return empty favicon to silence browser requests."""
    from fastapi.responses import Response
    return Response(content=b"", media_type="image/x-icon")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

