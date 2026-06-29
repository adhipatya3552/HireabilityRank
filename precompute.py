#!/usr/bin/env python3
"""
STEP 1 — Run this ONCE offline (outside the 5-minute submission window).
Computes sentence embeddings for all 100K candidates and builds a FAISS index.

Usage:
    python precompute.py --candidates candidates.jsonl
    (or .jsonl.gz — both are handled)

Output:
    precomputed/candidates.faiss   <- FAISS index for fast similarity search
    precomputed/metadata.pkl       <- compact candidate metadata (signals, skills, career)
"""

import argparse
import gzip
import json
import pickle
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"   # 80MB, fast on CPU, 384-dim
BATCH_SIZE = 512
OUTPUT_DIR = Path("precomputed")


def build_candidate_text(c: dict) -> str:
    """Create a single searchable text blob per candidate."""
    parts = []
    p = c.get("profile", {})

    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", "")[:400])

    # Weight advanced/expert skills more
    skills = c.get("skills", [])
    adv_skills = [s["name"] for s in skills if s.get("proficiency") in ("advanced", "expert")]
    all_skills = [s["name"] for s in skills]
    parts.append("Expert skills: " + ", ".join(adv_skills))
    parts.append("All skills: " + ", ".join(all_skills))

    # Career descriptions (recent 3 jobs)
    for job in c.get("career_history", [])[:3]:
        title = job.get("title", "")
        company = job.get("company", "")
        desc = job.get("description", "")[:250]
        parts.append(f"{title} at {company}: {desc}")

    return " | ".join(p for p in parts if p.strip())


def load_candidates(path: str) -> list:
    print(f"Loading candidates from {path} ...")
    opener = gzip.open if path.endswith(".gz") else open
    candidates = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"  → Loaded {len(candidates):,} candidates")
    return candidates


def extract_metadata(c: dict) -> dict:
    """Keep only what rank.py needs at runtime — saves memory."""
    p = c.get("profile", {})
    return {
        "candidate_id": c["candidate_id"],
        "profile": {
            "headline": p.get("headline", ""),
            "summary": p.get("summary", "")[:300],
            "location": p.get("location", ""),
            "country": p.get("country", ""),
            "years_of_experience": p.get("years_of_experience", 0),
            "current_title": p.get("current_title", ""),
            "current_company": p.get("current_company", ""),
        },
        "skills": [
            {
                "name": s.get("name", ""),
                "proficiency": s.get("proficiency", ""),
                "endorsements": s.get("endorsements", 0),
                "duration_months": s.get("duration_months", 0),
            }
            for s in c.get("skills", [])
        ],
        "career_history": [
            {
                "title": j.get("title", ""),
                "company": j.get("company", ""),
                "description": j.get("description", "")[:300],
                "duration_months": j.get("duration_months", 0),
                "is_current": j.get("is_current", False),
                "industry": j.get("industry", ""),
            }
            for j in c.get("career_history", [])[:4]
        ],
        "redrob_signals": c.get("redrob_signals", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.jsonl")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    candidates = load_candidates(args.candidates)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Building candidate text representations...")
    ids = [c["candidate_id"] for c in candidates]
    texts = [build_candidate_text(c) for c in candidates]

    print(f"Encoding {len(texts):,} candidates in batches of {BATCH_SIZE}...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine sim via inner product
    ).astype("float32")

    print("Building FAISS IndexFlatIP (inner product = cosine on normalized vecs)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(OUTPUT_DIR / "candidates.faiss"))
    print(f"  → FAISS index saved ({(OUTPUT_DIR/'candidates.faiss').stat().st_size/1e6:.1f} MB)")

    print("Saving compact metadata...")
    meta = {
        "ids": ids,
        "metadata": [extract_metadata(c) for c in candidates],
    }
    with open(OUTPUT_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(meta, f, protocol=4)
    print(f"  → metadata.pkl saved ({(OUTPUT_DIR/'metadata.pkl').stat().st_size/1e6:.1f} MB)")

    print("\n✅ Precompute complete! Now run: python rank.py --out YOUR_TEAM_ID.csv")


if __name__ == "__main__":
    main()
