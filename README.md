
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



📄 License

MIT — see LICENSE.
