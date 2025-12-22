
# Art Assistant 🎨
*Analyze a painting or sketch → get style cues, value/shape simplification, similar references, and 2 actionable tips. — helping artists learn, iterate, and improve.*

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#) [![Python](https://img.shields.io/badge/python-3.11-blue)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](#)

**Who it’s for:** Artists/students practicing composition, values, and style.  
**What it does:** Upload an image → see simplified value study, related references, style tag, and concise feedback.

---

## 🚀 Demo
**Live App:** _link to Streamlit/HF Spaces_  ---TO BE ADDED SOON
**Quick look:**  
![Demo](docs/demo.gif)

---

## ✨ Key Features
- **Similar References (CLIP + FAISS):** Retrieve 5–10 visually/style-related works for inspiration.
- **Value Practice / Shape Simplification:** 3–5 value posterization + color block view to study big shapes.
- **Style Classification (ViT/CLIP):** Style label + 1–2 sentence context note.
- **Actionable Tips (2 bullets):** Data-driven suggestions based on value distribution & composition heuristics, wrapped in LLM model for human-readable feedback. 
- *(Planned)* **Segmentation (SAM) & Generative Previews:** Big-shape masks; optional contrast/palette “what-if” previews.

---

## 🧠 Why this project
Artists often struggle to **simplify values and see big shapes**. Current advice is scattered and manual.  
**Art Assistant** offers instant **value studies, references, and context** to accelerate learning.

---

## 🛠️ Tech Stack
- **Models:** CLIP (embeddings), ViT/ResNet (style classifier)  
- **Retrieval:** FAISS (nearest neighbor over embeddings)  
- **Vision & Metrics:** OpenCV / scikit-image (posterization, k-means color blocks, value distribution)  
- **App:** Streamlit (UI)  
- **Data:** WikiArt (images + style metadata)  
- **Ops:** GitHub Actions (CI), pytest
- **LLM:**: OpenAI / local (configurable) — prompts over extracted metrics (no raw image by default); low temperature (~0.2) for consistent, two-bullet tips.

How the Value Tools Work (in plain English)
- Big Value Blocks (K=4–6): groups the scene into a few, large, connected shapes by value (enforces simplification). Best starting map for painting.
- Value Plan (K=3–8): discrete value steps (quantile by default). Good for refining lights/shadows after blocking-in.
- Grayscale & Metrics: quick “squint test,” plus mean/contrast and dark/light coverage.
- Defaults that teach well: Blocks K=5, Plan K=7 (method: quantile).

Key App Settings (sidebar / panels)
- Retrieval
- FAISS index path: data/index_samples/index.faiss
- Metadata JSONL path: data/embeddings_samples/artwork_index.jsonl

Top-K: 5 (typical)
- Value Studies
- Blocks (K) 4–6 (fewer = simpler; more = more detail)
- Spatial coherence (0–1): higher → bigger, connected masses
- Downscale (px): 300–420 is a good speed/quality tradeoff

Value Plan: K=3–8 (quantile recommended)

Performance Tips
- Large images? The value tools downscale internally; keep long side ~360–420px for speed and readable shapes.
- Caching is built in. If things re-compute too often, ensure you’re passing hashable inputs (bytes/NumPy arrays) to cached functions.
- GPU optional: use faiss-gpu and set device: "cuda" if your style/retrieval models run on Torch.

Troubleshooting
- ModuleNotFoundError: value_blocks
- Ensure scripts/__init__.py exists, import with from scripts.value_blocks import ..., and run from repo root.
- Python < 3.10 union types, Replace int | None with Optional[int] or add from __future__ import annotations at the top of the file.
- OpenCV missing, pip install opencv-python.
- Unreadable image, Ensure file is JPG/PNG/WEBP. PIL will raise UnidentifiedImageError on malformed files.

Data Notes
- Embeddings/Index: artwork_index.jsonl should contain items with id, image_path/url, and any style tags you use; index.faiss must be built with the same embedding model as at inference.
- Licensing: If you use WikiArt or other datasets, follow their license/terms. Don’t redistribute copyrighted images.

Roadmap
- Block labeling (1..K) and Dominant/Secondary/Supporting caption ✅
- Similarity warnings for adjacent values (merge/split hints)
- Optional subject mask / background separation

---

## 🚀 Production Operations

### Expected Startup Time

**Cold Start (First Request):**
- **FAISS Index Load:** ~2-5 seconds (depends on index size)
  - Small index (<1K artworks): ~2 seconds
  - Medium index (1K-10K): ~3-5 seconds
  - Large index (>10K): ~5-10 seconds
- **CLIP Model Load:** ~25-35 seconds (lazy-loaded on first search)
  - Model download (if not cached): +30-60 seconds
  - Model initialization: ~25-35 seconds
- **Total Cold Start:** ~30-40 seconds (model + index)

**Warm Start (Subsequent Requests):**
- Index already loaded: ~0 seconds
- Model already loaded: ~0 seconds
- **Total Warm Start:** <1 second

**Note:** The API uses lazy loading - models and index load on first use, not at container startup. Health checks (`/health`) respond immediately without loading models.

### Expected Latency Range

**Single Image Search (`/search`):**
- **Cache Hit:** 10-50ms (image already processed)
- **Cache Miss (Model Loaded):** 1-3 seconds
  - Image embedding (CLIP): ~1-2 seconds
  - FAISS search: ~50-200ms
  - Metadata lookup: ~10-50ms
- **Cache Miss (Cold Model):** 30-40 seconds (includes model load)

**Batch Search (`/search/batch`):**
- **Per Image (Model Loaded):** ~1-2 seconds
- **Batch of 5 images:** ~5-10 seconds
- **Batch of 10 images:** ~10-20 seconds
- Scales roughly linearly (model shared across batch)

**Health Check (`/health`):**
- **Response Time:** <10ms (no model/index access)

**Factors Affecting Latency:**
- **CPU vs GPU:** GPU reduces embedding time by 5-10x (if available)
- **Index Size:** Larger indexes slightly increase search time (~50ms per 10K artworks)
- **Image Size:** Larger images take longer to process (API auto-resizes)
- **Network:** Upload time depends on image size and connection speed

### Monitoring

**Health Checks:**
```bash
# Basic health check
curl http://your-api-url/health

# Expected response:
{
  "status": "healthy",
  "index_size": 3148,
  "metadata_count": 3148,
  "cache_size": 5,
  "cache_max": 100
}
```

**Key Metrics to Monitor:**
1. **Health Status:** `status == "healthy"` (should always be true)
2. **Index Size:** Should match expected number of artworks
3. **Cache Hit Rate:** `cache_size / total_requests` (higher is better)
4. **Response Times:** Track p50, p95, p99 latencies
5. **Error Rate:** Monitor 4xx/5xx responses

**Logs:**
```bash
# Docker logs
docker logs art-assistant-api --tail 100 --follow

# Key log events:
# - Model loading: "[info] Using device: cpu" or "cuda"
# - Search requests: "POST /search HTTP/1.1" 200 OK
# - Errors: HTTP status codes, tracebacks
# - Rate limiting: "429 Too Many Requests"
```

**Error Rate Monitoring:**
- **429 (Rate Limit):** Normal under load, indicates need for rate limit adjustment
- **500 (Server Error):** Investigate immediately (model load failure, index corruption)
- **400 (Bad Request):** Client-side issue (invalid image format, missing file)

**Recommended Monitoring Setup:**
1. **Health Check Endpoint:** Poll `/health` every 30 seconds
2. **Application Logs:** Stream to centralized logging (Datadog, CloudWatch, etc.)
3. **Metrics:** Track response times, error rates, cache hit rate
4. **Alerts:** Set up alerts for:
   - Health check failures
   - Error rate > 1%
   - P95 latency > 5 seconds
   - Cache hit rate < 50%

### Refreshing Artifacts (Rebuilding Embeddings & Index)

**When to Refresh:**
- Adding new artworks to the collection
- Updating existing artwork metadata
- Changing embedding model (e.g., upgrading CLIP version)
- Index corruption or performance degradation

**Step-by-Step Process:**

**1. Build New Artifacts (Colab Recommended):**
```python
# In Google Colab or local environment
# Step 1: Download/update artwork data
!python scripts/download_met_images.py \
  --limit 5000 \
  --download-missing \
  --no-oil-filter \
  --thumb \
  --delay 0.1

# Step 2: Generate embeddings
!python scripts/hf_embed_min.py \
  --parquet data/corpus/met_oil_paintings.parquet \
  --out-dir data/embeddings_samples \
  --limit 5000

# Step 3: Build FAISS index
!python scripts/build_faiss_samples.py

# Step 4: Verify artifacts
import os
print(f"Index size: {os.path.getsize('data/index_samples/index.faiss') / 1024 / 1024:.2f} MB")
print(f"Metadata size: {os.path.getsize('data/index_samples/meta.npy') / 1024 / 1024:.2f} MB")
```

**2. Upload to Persistent Storage:**
- **Render:** Upload to persistent disk via dashboard or `rsync`
- **AWS S3:** Use `aws s3 cp` or boto3
- **Google Cloud Storage:** Use `gsutil` or gcloud SDK

**3. Update Environment Variables:**
```bash
# Set new artifact paths (if using external storage)
export FAISS_INDEX_PATH=/path/to/new/index.faiss
export FAISS_META_PATH=/path/to/new/meta.npy
```

**4. Restart Service:**
```bash
# Docker Compose
docker-compose restart

# Or rebuild and redeploy
docker-compose up -d --build

# Render/Railway: Trigger redeploy via dashboard or git push
```

**5. Verify Deployment:**
```bash
# Check health endpoint
curl http://your-api-url/health

# Should show new index_size matching your new artifact count
# Example: {"index_size": 5000, "metadata_count": 5000, ...}
```

**Best Practices:**
- ✅ **Test artifacts locally** before deploying to production
- ✅ **Keep backups** of previous artifacts (version control or S3)
- ✅ **Gradual rollout:** Deploy to staging first, then production
- ✅ **Monitor after refresh:** Check error rates and latency for 24 hours
- ✅ **Document changes:** Note what changed (new artworks, model version, etc.)

**Rollback Plan:**
If new artifacts cause issues:
1. Revert environment variables to previous artifact paths
2. Restart service
3. Investigate issues in staging environment
4. Fix and rebuild artifacts before retrying

---

📄 License

MIT — see LICENSE.
