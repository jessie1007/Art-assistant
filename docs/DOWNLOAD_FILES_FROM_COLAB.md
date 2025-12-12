# Downloading Files from Google Colab to Local

## Files Generated in Colab

You have these files in Google Drive:
```
/content/drive/MyDrive/Art-assistant/data/corpus/met_oil_paintings.parquet
/content/drive/MyDrive/Art-assistant/embeddings_samples/artwork_index.json
/content/drive/MyDrive/Art-assistant/embeddings_samples/embeddings.npy
/content/drive/MyDrive/Art-assistant/index_samples/index.faiss
/content/drive/MyDrive/Art-assistant/index_samples/meta.npy
```


**Current Local File Sizes:**
- `met_oil_paintings.parquet`: 11KB (likely small sample)
- `embeddings.npy`: 40KB (likely small sample)
- `index.faiss`: 40KB (likely small sample)
- `artwork_index.jsonl`: 1KB (likely small sample)
- `meta.npy`: **MISSING** ❌

## Where to Place Files Locally

### Directory Structure:
```
Art-assistant/
├── data/
│   ├── corpus/
│   │   └── met_oil_paintings.parquet          ← Download here
│   ├── embeddings_samples/
│   │   ├── artwork_index.jsonl                 ← Convert .json to .jsonl
│   │   └── embeddings.npy                      ← Download here
│   └── index_samples/
│       ├── index.faiss                          ← Download here
│       └── meta.npy                             ← Download here (MISSING!)
```

## Step-by-Step Download Instructions

### Download via Colab (Recommended)

Run this in a Colab cell to download all files:

```python
from google.colab import files
import os
from pathlib import Path

# Base path in Drive
base = "/content/drive/MyDrive/Art-assistant"

# Files to download
files_to_download = [
    f"{base}/data/corpus/met_oil_paintings.parquet",
    f"{base}/embeddings_samples/artwork_index.json",
    f"{base}/embeddings_samples/embeddings.npy",
    f"{base}/index_samples/index.faiss",
    f"{base}/index_samples/meta.npy",
]

# Download each file
for file_path in files_to_download:
    if os.path.exists(file_path):
        print(f"Downloading {file_path}...")
        files.download(file_path)
    else:
        print(f"⚠️  File not found: {file_path}")
```


## Verify Files After Download

```bash
cd /Users/jiezhao/Documents/Data_Sciense_Project/Art-assistant

# Check all required files exist
ls -lh data/corpus/met_oil_paintings.parquet
ls -lh data/embeddings_samples/artwork_index.jsonl
ls -lh data/embeddings_samples/embeddings.npy
ls -lh data/index_samples/index.faiss
ls -lh data/index_samples/meta.npy
```

## File Size Expectations

After downloading large dataset:
- `met_oil_paintings.parquet`: **MBs to GBs** (depends on number of images)
- `embeddings.npy`: **MBs to GBs** (512-dim vectors × number of images)
- `index.faiss`: **Similar to embeddings.npy**
- `artwork_index.jsonl`: **MBs** (metadata for each image)
- `meta.npy`: **Similar to artwork_index.jsonl**

## Quick Check: Compare File Sizes

**In Colab:**
```python
import os
from pathlib import Path

base = Path("/content/drive/MyDrive/Art-assistant")

files = {
    "parquet": base / "data/corpus/met_oil_paintings.parquet",
    "embeddings": base / "embeddings_samples/embeddings.npy",
    "index": base / "index_samples/index.faiss",
    "meta": base / "index_samples/meta.npy",
}

for name, path in files.items():
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{name}: {size_mb:.2f} MB")
    else:
        print(f"{name}: ❌ NOT FOUND")
```

**Locally:**
```bash
cd /Users/jiezhao/Documents/Data_Sciense_Project/Art-assistant
du -h data/corpus/met_oil_paintings.parquet
du -h data/embeddings_samples/embeddings.npy
du -h data/index_samples/index.faiss
du -h data/index_samples/meta.npy
```

If Colab files are **much larger**, definitely download them!

## After Downloading

1. **Test the API:**
   ```bash
   uvicorn app.api.main:app --reload
   curl http://localhost:8000/health
   ```

2. **Test Streamlit:**
   ```bash
   streamlit run scripts/app_streamlit_main.py
   ```

3. **Verify search works:**
   - Upload an image in Streamlit
   - Check if results appear
   - Verify no errors in console

## Troubleshooting

### "File not found" errors:
- Check file paths match exactly
- Ensure directories exist: `mkdir -p data/{corpus,embeddings_samples,index_samples}`

### "Invalid format" errors:
- Verify `.json` → `.jsonl` conversion worked
- Check JSONL format: one JSON object per line

### "Index size mismatch" errors:
- Ensure `embeddings.npy` and `index.faiss` are from same run
- Rebuild index if needed: `python scripts/build_faiss_samples.py`

