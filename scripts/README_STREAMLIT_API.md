# Streamlit + FastAPI Integration Guide

## Overview

The Streamlit app now supports connecting to the FastAPI backend for image similarity search. This provides better separation of concerns and allows the API to be used by other clients as well.

## Architecture

```
┌─────────────┐         HTTP/REST          ┌──────────────┐
│  Streamlit  │ ────────────────────────> │   FastAPI    │
│   Frontend  │ <──────────────────────── │   Backend    │
└─────────────┘         JSON Response      └──────────────┘
```

## Setup

### 1. Start FastAPI Server

In one terminal:

```bash
cd /Users/jiezhao/Documents/Data_Sciense_Project/Art-assistant
source .venv/bin/activate  # if using virtual env
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`

### 2. Start Streamlit App

In another terminal:

```bash
cd /Users/jiezhao/Documents/Data_Sciense_Project/Art-assistant
source .venv/bin/activate
streamlit run scripts/app_streamlit_main.py
```

### 3. Configure in Streamlit UI

1. Open the Streamlit app in your browser (usually `http://localhost:8501`)
2. In the sidebar, check "Use FastAPI backend"
3. Verify API URL is `http://localhost:8000`
4. You should see a green checkmark: "✅ API Connected"

## Features

### API Client (`app/api_client.py`)

- **Environment Variable Support**: Set `ART_ASSISTANT_API_URL` to override default
- **Automatic Retries**: Configurable retry logic with exponential backoff
- **Error Handling**: Graceful error handling with informative messages
- **Type Hints**: Full type annotations for better IDE support

### Streamlit Integration

- **Dual Mode**: Supports both API mode and direct FAISS mode (fallback)
- **Health Checks**: Real-time API health status in sidebar
- **Error Recovery**: Automatically falls back to direct FAISS if API fails
- **Loading States**: Clear spinner messages during API calls

## API Endpoints Used

1. **`GET /health`** - Check API health and index status
2. **`POST /search`** - Upload image and get similar artworks
3. **`GET /artwork/{id}`** - Get artwork metadata by ID

## Environment Variables

```bash
# Optional: Set custom API URL
export ART_ASSISTANT_API_URL="http://localhost:8000"
```

## Best Practices

1. **Always start FastAPI first** before launching Streamlit
2. **Check health status** in sidebar before searching
3. **Use API mode in production** for better scalability
4. **Use direct FAISS mode** for development/testing without API

## Troubleshooting

### API not responding

- Check if FastAPI server is running: `curl http://localhost:8000/health`
- Verify port 8000 is not in use
- Check firewall settings

### Import errors

- Ensure `app/api_client.py` exists
- Check Python path includes project root
- Verify `requests` is installed: `pip install requests`

### Fallback to direct FAISS

- If API fails, Streamlit automatically uses direct FAISS mode
- Check console logs for error messages
- Verify FAISS index files exist in `data/index_samples/`

## Example Usage

```python
from app.api_client import ArtAssistantAPIClient
from PIL import Image

# Initialize client
client = ArtAssistantAPIClient(api_url="http://localhost:8000")

# Check health
health = client.health_check()
print(health)

# Search image
image = Image.open("my_image.jpg")
results = client.search_image(image, topk=5)

# Display results
for artwork in results["results"]:
    print(f"{artwork['title']} - Score: {artwork['similarity_score']:.3f}")
```

