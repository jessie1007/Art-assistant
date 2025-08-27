## 2025-08-26 – Week 2, Day 2
### Goal
Baseline CLIP retrieval working with small index

### What I did
- Embedded 10k WikiArt images with ViT-B/32
- Built FAISS index; added top_k API in src/models/clip_index.py

### Key learnings
- CLIP works well for stylistic similarity vs ResNet features
- Index build time acceptable; memory OK for 30k images

### Challenges
- Some returns are semantically right but style-off

### Next steps
- Add style classifier head for comparison
- Start P@5 human eval rubric

## 2025-08-27
### Goal
- Kick off CLIP retrieval experiment

### What I did
- 

### Key learnings
- 

### Challenges
- 

### Next steps
- 

