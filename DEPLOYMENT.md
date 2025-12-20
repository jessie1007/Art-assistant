# Production Deployment Guide

## Why API Testing?

Before deploying, we test the API to:
1. **Verify it works with real data** - Ensures FAISS index loads correctly
2. **Catch errors early** - Find issues before users do
3. **Validate search quality** - Make sure results make sense
4. **Check performance** - Response times, memory usage
5. **End-to-end validation** - Full pipeline works correctly

## Prerequisites

- Docker and Docker Compose installed
- FAISS index files (built separately - see below)
- Metadata files (built separately - see below)

## Artifact Separation (Important!)

**Artifacts are separate from code!** The container loads them via configuration.

### What This Means:
- ✅ Build embeddings/index once (in Colab or locally)
- ✅ Upload artifacts to persistent storage (Render disk, S3, etc.)
- ✅ Container loads them via environment variables
- ✅ No need to rebuild embeddings/index on every deployment
- ✅ Faster deployments (just code changes)

### Where to Build Artifacts:
1. **In Colab** (recommended for large datasets):
   ```python
   # Build embeddings
   !python scripts/hf_embed_min.py --limit 4000
   
   # Build FAISS index
   !python scripts/build_faiss_samples.py
   
   # Download files
   from google.colab import files
   files.download('data/index_samples/index.faiss')
   files.download('data/index_samples/meta.npy')
   ```

2. **Locally** (for smaller datasets):
   ```bash
   python scripts/hf_embed_min.py --limit 4000
   python scripts/build_faiss_samples.py
   ```

Then upload to your deployment platform's persistent storage.

## Local Deployment with Docker

### Step 1: Prepare Data Files

Ensure you have these files in `data/`:
```
data/
├── index_samples/
│   ├── index.faiss
│   └── meta.npy
└── embeddings_samples/
    └── artwork_index.jsonl
```

### Step 2: Build Docker Image

```bash
docker build -t art-assistant-api .
```

### Step 3: Run with Docker Compose

```bash
docker-compose up -d
```

Or run directly:
```bash
docker run -d \
  --name art-assistant-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data:ro \
  art-assistant-api
```

### Step 4: Test

**Option A: Use test script (Recommended)**
```bash
# Run comprehensive test suite
python tests/test_api_deployment.py http://localhost:8000

# Or with custom image
python tests/test_api_deployment.py http://localhost:8000 --image path/to/image.jpg
```

**Option B: Manual testing**
```bash
# Health check
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

## Cloud Deployment Options

### Option 1: Render (Recommended - Easy & Free Tier)

Render is perfect for getting started with a production deployment.

#### Step-by-Step Render Deployment:

1. **Prepare your repository**
   - Push all code to GitHub (including Dockerfile)
   - Ensure `data/index_samples/` files are in repo or use persistent disk

2. **Create Render account**
   - Go to https://render.com
   - Sign up with GitHub

3. **Create new Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository

4. **Configure service**
   - **Name**: `art-assistant-api` (or your choice)
   - **Region**: Choose closest to you
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty (or `.` if needed)
   - **Runtime**: `Docker`
   - **Build Command**: (Leave empty - Render auto-detects Dockerfile)
   - **Start Command**: (Leave empty - Render uses Dockerfile CMD)

5. **Add environment variables**
   - `ART_ASSISTANT_API_URL`: Your Render URL (auto-set, optional)
   - `WORKERS`: `2` (or adjust based on plan, optional)
   - `FAISS_INDEX_PATH`: Path to index.faiss (default: `/app/data/index_samples/index.faiss`)
   - `FAISS_META_PATH`: Path to meta.npy (default: `/app/data/index_samples/meta.npy`)
   
   **Key Point**: Artifacts are separate from code! The container loads them via these paths.
   You don't need to rebuild embeddings/index locally - just point to where they're stored.

6. **Set up persistent disk** (For data files)
   - Go to "Disks" tab
   - Create new disk: `art-assistant-data`
   - Mount path: `/app/data`
   - **Important**: Upload your `data/index_samples/` files to this disk after first deploy
   - Files persist across deployments - you only upload once!

7. **Deploy**
   - Click "Create Web Service"
   - Render will build and deploy automatically
   - Wait ~5-10 minutes for first build

8. **Upload data files** (After first deploy)

   **Option A: Using Render Shell (Recommended)**
   - Go to your service → "Shell" tab
   - Create directories:
     ```bash
     mkdir -p /app/data/index_samples
     mkdir -p /app/data/embeddings_samples
     ```
   - Upload files using `scp` or Render's file upload feature
   - Or use `wget`/`curl` if files are hosted somewhere

   **Option B: Include in Git (For smaller files)**
   - Add data files to git (if under size limits)
   - Render will include them in build
   - ⚠️ Not recommended for large files (>100MB)

   **Option C: Download on container start**
   - Modify Dockerfile to download from S3/Google Drive on startup
   - Good for large files that change infrequently

9. **Get your public URL**
   - Render provides: `https://art-assistant-api.onrender.com`
   - Share this URL with others!

#### Render Free Tier Limits:
- ✅ 750 hours/month free
- ✅ Auto-sleeps after 15 min inactivity (wakes on request)
- ⚠️ First request after sleep takes ~30 seconds (cold start)
- 💰 Upgrade to paid plan for always-on service

#### Render Tips:
- Use "Manual Deploy" to trigger rebuilds
- Check "Logs" tab for debugging
- "Metrics" tab shows CPU/memory usage
- Enable "Auto-Deploy" for automatic updates on git push

---

### Option 2: AWS (For Later - Production Scale)

When you're ready for AWS, here are the options:

#### AWS ECS/Fargate (Recommended for AWS)

1. **Build and push to ECR**
   ```bash
   # Install AWS CLI and configure
   aws configure
   
   # Create ECR repository
   aws ecr create-repository --repository-name art-assistant-api
   
   # Get login token
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
   
   # Build and tag
   docker build -t art-assistant-api .
   docker tag art-assistant-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/art-assistant-api:latest
   
   # Push
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/art-assistant-api:latest
   ```

2. **Create ECS Task Definition**
   - Use Fargate launch type
   - Set CPU: 1 vCPU, Memory: 2 GB (minimum)
   - Configure container with your image
   - Set port mapping: 8000

3. **Create ECS Service**
   - Use Application Load Balancer
   - Set desired count: 1-2 tasks
   - Configure health checks

4. **Set up S3 for data files** (Optional)
   - Upload `index.faiss`, `meta.npy` to S3
   - Mount as EFS or download on container start

#### AWS App Runner (Simpler Alternative)

1. **Connect GitHub repo**
2. **Create App Runner service**
3. **Auto-deploys on push**
4. **Simpler than ECS but less control**

#### AWS Lambda + API Gateway (Serverless)

- Requires refactoring for serverless
- Good for cost optimization
- More complex setup

---

### Option 3: Railway (Alternative - Easy)

1. Install Railway CLI: `npm i -g @railway/cli`
2. Login: `railway login`
3. Initialize: `railway init`
4. Deploy: `railway up`

Railway auto-detects Docker and deploys.

---

### Option 4: Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/art-assistant-api

# Deploy
gcloud run deploy art-assistant-api \
  --image gcr.io/PROJECT_ID/art-assistant-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Environment Variables

Optional (defaults work):
- `ART_ASSISTANT_API_URL` - API base URL
- `WORKERS` - Number of uvicorn workers (default: 2)

## Production Considerations

### Security
- ✅ Non-root user in container
- ✅ Read-only data volume
- ✅ Rate limiting enabled
- ⚠️ Add authentication if exposing publicly
- ⚠️ Use HTTPS in production

### Performance
- Multi-worker setup (2 workers by default)
- Caching enabled for repeated searches
- FAISS index loaded in memory (fast)

### Monitoring
- Health check endpoint: `/health`
- API docs: `/docs`
- Consider adding logging/monitoring (Sentry, DataDog, etc.)

## Sharing Your API

Once deployed on Render, share the public URL:
- **Render**: `https://art-assistant-api.onrender.com` (or your custom domain)

Users can:
1. Visit `https://your-app.onrender.com/docs` for interactive API documentation
2. Use `/search` endpoint to upload images
3. Get similar artworks back

### Example: Share with Friends

```
🎨 Art Assistant API
Find similar artworks using AI!

Try it: https://art-assistant-api.onrender.com/docs
Upload an image and get similar paintings from the Met Museum!
```

### Custom Domain (Optional)

1. Go to Render dashboard → Your service → Settings
2. Add custom domain
3. Update DNS records
4. Your API will be available at `https://api.yourdomain.com`

## Troubleshooting

### Render-Specific Issues

#### Service won't start
- Check "Logs" tab in Render dashboard
- Look for errors about missing files or import failures
- Verify Dockerfile is in root directory

#### Index not found (Render)
- Ensure data files are uploaded to persistent disk
- Check disk mount path: `/app/data`
- Use Render Shell to verify: `ls -la /app/data/index_samples/`
- Files must be in persistent disk, not just in container

#### Cold start takes too long
- First request after sleep takes ~30 seconds (normal for free tier)
- Upgrade to paid plan for always-on service
- Or use Render's "Background Worker" to keep service warm

#### Out of memory (Render)
- Render free tier: 512 MB RAM
- Reduce workers in Dockerfile: Change `--workers 2` to `--workers 1`
- Or upgrade to paid plan for more memory

### General Issues

#### Container won't start (Local)
- Check logs: `docker logs art-assistant-api`
- Verify data files exist: `ls -la data/index_samples/`
- Check port availability: `lsof -i :8000`

#### Index not found (Local)
- Ensure `data/index_samples/index.faiss` exists
- Check volume mount: `docker exec art-assistant-api ls -la /app/data/index_samples/`

