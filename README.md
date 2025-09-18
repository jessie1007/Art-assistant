
# Art Assistant 🎨
*Analyze a painting or sketch → get style cues, value/shape simplification, similar references, and 2 actionable tips. — helping artists learn, iterate, and improve.*

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#) [![Python](https://img.shields.io/badge/python-3.11-blue)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](#)

**Who it’s for:** Artists/students practicing composition, values, and style.  
**What it does:** Upload an image → see simplified value study, related references, style tag, and concise feedback.

---

## 🚀 Demo
**Live App:** _link to Streamlit/HF Spaces_  
**Quick look:**  
![Demo](docs/demo.gif)

---

## ✨ Key Features
- **Similar References (CLIP + FAISS):** Retrieve 5–10 visually/style-related works for inspiration.
- **Value Practice / Shape Simplification:** 3–5 value posterization + color block view to study big shapes.
- **Style Classification (ViT/CLIP):** Style label + 1–2 sentence context note.
- **Actionable Tips (2 bullets):** Data-driven suggestions based on value distribution & composition heuristics.  
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

---

## 📦 Getting Started
```bash
# 1) Clone
git clone https://github.com/<you>/art-assistant.git && cd art-assistant

# 2) Environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3) Run tests
pytest -q

# 4) Launch app
streamlit run app/app.py

## ✅ Week 1 — Retrieval MVP (Completed)

- Collected sample oil paintings from The Met Open Access.
- Generated CLIP embeddings (`embeddings.npy`) and saved metadata (`artwork_index.jsonl`).
- Built a FAISS index for fast nearest-neighbor search.
- Created a **Streamlit app** (`app_streamlit_retrieval.py`):
  - Upload a painting → embed with CLIP → search FAISS → show top-5 similar artworks.
  - Displays **title, artist, year, style**, and similarity score.

### Run Streamlit App