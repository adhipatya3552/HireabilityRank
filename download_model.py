#!/usr/bin/env python3
"""
Run this ONCE with internet access to download and cache the model.
After this, rank.py works fully offline (no network needed).

Usage:
    python download_model.py
"""
from sentence_transformers import SentenceTransformer

MODEL = "all-MiniLM-L6-v2"
print(f"Downloading and caching model: {MODEL}")
model = SentenceTransformer(MODEL)

# Quick smoke test
vecs = model.encode(["test sentence"], normalize_embeddings=True)
print(f"✅ Model ready! Embedding dim: {vecs.shape[1]}")
print("You can now run rank.py offline (no network needed).")
