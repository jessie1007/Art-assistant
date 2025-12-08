"""
FastAPI server for Art Assistant image search.

Endpoints:
- POST /search: Upload image, get similar artworks
- GET /health: Health check
- GET /artwork/{id}: Get artwork metadata by index ID
"""

import sys
from pathlib import Path
from typing import List, Optional
import json

# Add project root to path (go up 3 levels: app/api/main.py -> root)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default paths
DEFAULT_INDEX_PATH = ROOT / "data/index_samples/index.faiss"
DEFAULT_META_PATH = ROOT / "data/index_samples/meta.npy"

# Global variables for loaded index and metadata
_index = None
_metadata = None


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
        "metadata_count": len(meta) if meta else 0
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


@app.post("/search")
async def search_similar(
    file: UploadFile = File(...),
    topk: int = 5
):
    """
    Search for similar artworks by uploading an image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        topk: Number of similar artworks to return (default: 5, max: 50)
    
    Returns:
        List of similar artworks with metadata and similarity scores
    """
    if topk < 1 or topk > 50:
        raise HTTPException(status_code=400, detail="topk must be between 1 and 50")
    
    # Load index and metadata
    index, metadata = load_index_and_metadata()
    
    if index.ntotal == 0:
        raise HTTPException(status_code=503, detail="Index is empty")
    
    # Read and process image
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
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


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Art Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "search": "POST /search - Upload image to find similar artworks",
            "health": "GET /health - Health check",
            "artwork": "GET /artwork/{id} - Get artwork metadata by ID"
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

