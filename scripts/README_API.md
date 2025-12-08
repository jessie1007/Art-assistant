# FastAPI Server for Art Assistant

A REST API for image similarity search using CLIP embeddings and FAISS.

## Setup

1. Install dependencies:
```bash
pip install fastapi uvicorn[standard] python-multipart
```

2. Make sure you have:
   - FAISS index: `data/index_samples/index.faiss`
   - Metadata: `data/index_samples/meta.npy` or `data/embeddings_samples/artwork_index.jsonl`

## Running the Server

### Development (with auto-reload):
```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production:
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Or:
```bash
python -m app.api.main
```

## API Endpoints

### `GET /`
Root endpoint with API information.

### `GET /health`
Health check endpoint.
- Returns: `{"status": "healthy", "index_size": N, "metadata_count": M}`

### `POST /search`
Search for similar artworks by uploading an image.

**Request:**
- `file`: Image file (multipart/form-data)
- `topk`: Number of results (1-50, default: 5)

**Response:**
```json
{
  "query_image_size": [width, height],
  "topk": 5,
  "results": [
    {
      "id": 0,
      "title": "Artwork Title",
      "artist": "Artist Name",
      "year": "1900",
      "similarity_score": 0.95,
      "rank": 1,
      "image_available": true,
      "local_path": "data/images/...",
      ...
    }
  ]
}
```

### `GET /artwork/{id}`
Get artwork metadata by index ID.

**Response:**
```json
{
  "id": 0,
  "title": "Artwork Title",
  "artist": "Artist Name",
  "year": "1900",
  "local_path": "data/images/...",
  "image_available": true,
  ...
}
```

## Testing

Run the test script:
```bash
python scripts/test_api.py
```

Or test manually with curl:
```bash
# Health check
curl http://localhost:8000/health

# Search
curl -X POST "http://localhost:8000/search?topk=5" \
  -F "file=@path/to/image.jpg"

# Get artwork
curl http://localhost:8000/artwork/0
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Notes

- The server loads the FAISS index and metadata on startup (lazy loading on first request if startup fails)
- Images are embedded using CLIP (openai/clip-vit-base-patch32)
- Similarity scores are cosine similarity (higher = more similar)
- The API supports CORS for web applications

