#!/usr/bin/env python3
"""
STEP 2 — Main Ranker (submission-compliant)
Runs in <5 minutes on CPU, 16 GB RAM, zero network calls.

Usage:
    python rank.py --out YOUR_TEAM_ID.csv

Requirements:
    precomputed/candidates.faiss and precomputed/metadata.pkl
    must exist (run precompute.py first).
"""

import argparse
import csv
import pickle
from datetime import date, datetime
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME     = "all-MiniLM-L6-v2"
PRECOMPUTED    = Path("precomputed")
TOP_K_RETRIEVE = 5000    # FAISS top-K before reranking
TOP_N_SUBMIT   = 100     # final submission count

# ── JD Text for Embedding ───────────────────────────────────────────────────
# Crafted to match the ACTUAL vocabulary of top candidates in this domain.
JD_TEXT = """
Senior AI Engineer founding team Redrob AI talent intelligence Series A startup.
Production experience sentence-transformers embeddings BGE E5 OpenAI embeddings deployed real users.
Embedding drift index refresh retrieval quality regression production systems.
Vector databases hybrid search Pinecone Weaviate Qdrant Milvus FAISS Elasticsearch OpenSearch.
Strong Python code quality ranking retrieval matching systems.
Evaluation frameworks NDCG MRR MAP offline online A/B testing recruiter feedback loops.
LLM fine-tuning LoRA QLoRA PEFT learning-to-rank XGBoost neural reranking.
5 to 9 years production ML engineering retrieval ranking NLP information retrieval.
Product engineering ship fast real users metrics scrappy startup mentoring team.
Candidate job description matching at scale hybrid retrieval LLM reranking architecture.
Pune Noida India hybrid relocation Tier-1 Indian cities.
RAG retrieval augmented generation NLP transformers BERT language models.
"""

# ── JD skill sets ───────────────────────────────────────────────────────────
JD_REQUIRED = {
    "embeddings", "sentence-transformers", "sentence transformers", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "vector database", "hybrid search", "retrieval", "ranking", "python",
    "ndcg", "mrr", "information retrieval", "llm", "fine-tuning",
    "rag", "nlp", "bge", "e5", "openai embeddings", "bert", "transformer",
    "a/b testing", "production ml", "reranking",
}
JD_NICE = {
    "lora", "qlora", "peft", "xgboost", "lightgbm", "learning to rank",
    "langchain", "fastapi", "pytorch", "tensorflow", "mlflow",
    "langraph", "langfuse", "weaviate", "cohere", "huggingface",
}

# Tier-1 cities preferred by the JD
PREFERRED_CITIES = {
    "bangalore", "bengaluru", "pune", "noida", "delhi", "hyderabad",
    "mumbai", "chennai", "gurgaon", "gurugram", "ncr",
}

# ── Scoring Functions ────────────────────────────────────────────────────────

def skill_score(skills: list) -> float:
    """
    Match candidate skills against JD requirements.
    Trust multiplier penalises keyword stuffers (low endorsements + short duration).
    """
    if not skills:
        return 0.0

    req_hits, nice_hits = 0.0, 0.0
    PROF_W = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.6, "beginner": 0.3}

    for s in skills:
        name = s.get("name", "").lower().strip()
        prof_w = PROF_W.get(s.get("proficiency", "").lower(), 0.5)
        end = s.get("endorsements", 0)
        dur = s.get("duration_months", 0)

        # Keyword-stuffer trust multiplier
        trust = min(1.0, max(0.1, (end / 8.0) * 0.5 + (dur / 18.0) * 0.5))
        weight = prof_w * trust

        for req in JD_REQUIRED:
            if req in name or name in req:
                req_hits += weight
                break
        for nice in JD_NICE:
            if nice in name or name in nice:
                nice_hits += weight * 0.4
                break

    # Need ~25% of required signals for full score
    base = min(1.0, req_hits / (len(JD_REQUIRED) * 0.25))
    bonus = min(0.2, nice_hits / len(JD_NICE))
    return min(1.0, base * 0.85 + bonus * 0.15)


def experience_score(years) -> float:
    """Sweet spot 5–9 yrs per JD; taper outside."""
    if years is None:
        return 0.4
    y = float(years)
    if 5.0 <= y <= 9.0:   return 1.00
    if 4.0 <= y < 5.0:    return 0.85
    if 9.0 < y <= 11.0:   return 0.80
    if 3.0 <= y < 4.0:    return 0.65
    if y > 11.0:           return 0.60
    return 0.35


def location_score(profile: dict) -> float:
    loc = (profile.get("location", "") + " " + profile.get("country", "")).lower()
    for city in PREFERRED_CITIES:
        if city in loc:
            return 1.0
    if "india" in loc or loc.strip().endswith(" in"):
        return 0.65
    return 0.25


def career_signal_score(career_history: list) -> float:
    """
    Product-engineering tilt vs pure research.
    JD explicitly wants product shippers, not researchers.
    """
    PRODUCT_KW = {
        "deployed", "shipped", "production", "users", "metrics", "a/b",
        "scale", "latency", "api", "service", "pipeline", "product",
        "customer", "revenue", "growth", "recruiter", "real-world",
        "feature", "launch", "v2", "v3", "startup",
    }
    RESEARCH_KW = {
        "paper", "published", "lab", "experiment", "academic", "research",
        "benchmark", "arxiv", "university", "professor", "thesis",
        "annotation", "study", "grant", "phd", "postdoc",
    }

    prod, res = 0, 0
    for job in career_history[:4]:
        text = (job.get("description", "") + " " + job.get("title", "")).lower()
        for kw in PRODUCT_KW:
            if kw in text:
                prod += 1
        for kw in RESEARCH_KW:
            if kw in text:
                res += 1

    total = prod + res
    if total == 0:
        return 0.5
    return 0.35 + 0.65 * (prod / total)


def hireability_score(sig: dict) -> float:
    """
    Probability the candidate will actually be contacted, interviewed, and hired.
    This is the key differentiator — most rankers ignore behavioral signals.
    """
    if not sig:
        return 0.3

    score = 0.0

    # Intent signals (0.45 total)
    if sig.get("open_to_work_flag", False):
        score += 0.25

    last_active = sig.get("last_active_date", "")
    if last_active:
        try:
            days_ago = (date.today() - datetime.strptime(last_active, "%Y-%m-%d").date()).days
            if days_ago <= 7:      score += 0.20
            elif days_ago <= 30:   score += 0.14
            elif days_ago <= 90:   score += 0.06
        except Exception:
            pass

    # Responsiveness (0.25 total)
    rr = sig.get("recruiter_response_rate", 0.0)
    score += 0.15 * min(1.0, rr * 1.4)   # 0.71 → full score

    avg_resp = sig.get("avg_response_time_hours", 999)
    resp_score = max(0.0, 1.0 - avg_resp / 240.0)   # <240h = some score
    score += 0.10 * resp_score

    # Conversion signals (0.20 total)
    score += 0.12 * sig.get("interview_completion_rate", 0.0)
    score += 0.08 * sig.get("offer_acceptance_rate", 0.0)

    # Availability (0.10 total)
    notice = sig.get("notice_period_days", 90)
    score += 0.07 * max(0.0, (90 - notice) / 90.0)
    if sig.get("willing_to_relocate", False):
        score += 0.03

    return min(1.0, score)


def trap_penalty(cand: dict) -> float:
    """
    Detect honeypots and keyword stuffers.
    Returns 0.0 (clean) → 1.0 (almost certainly a trap).
    Keeps honeypot rate in top-100 well below the 10% DQ threshold.
    """
    sig = cand.get("redrob_signals", {})
    skills = cand.get("skills", [])
    flags = 0
    max_flags = 6

    # 1. Zero GitHub activity (suspicious for a Senior AI Engineer)
    if sig.get("github_activity_score", 0) == 0:
        flags += 1

    # 2. Zero profile views but lots of skills → invisible stuffed profile
    views = sig.get("profile_views_received_30d", 0)
    if views == 0 and len(skills) > 12:
        flags += 1

    # 3. Low avg endorsements + short duration per skill → keyword stuffing
    if skills:
        avg_end = sum(s.get("endorsements", 0) for s in skills) / len(skills)
        avg_dur = sum(s.get("duration_months", 0) for s in skills) / len(skills)
        if avg_end < 2.5 and avg_dur < 8:
            flags += 1

    # 4. Zero recruiter response rate → ghost profile
    if sig.get("recruiter_response_rate", 1.0) == 0.0:
        flags += 1

    # 5. Impossible completeness: >92% complete but zero saves, zero views
    saved = sig.get("saved_by_recruiters_30d", 0)
    complete = sig.get("profile_completeness_score", 0)
    if complete > 92 and saved == 0 and views == 0:
        flags += 1

    # 6. Not verified on either channel — untrustworthy identity
    if not sig.get("verified_email", True) and not sig.get("verified_phone", True):
        flags += 1

    return flags / max_flags


def make_reasoning(cand: dict, scores: dict, final: float) -> str:
    """Generate 1–2 sentence reasoning for CSV."""
    p = cand["profile"]
    yoe = p.get("years_of_experience", "?")
    title = p.get("current_title", "Engineer")
    company = p.get("current_company", "")
    loc = p.get("location", "")

    top_skills = [
        s["name"] for s in cand.get("skills", [])
        if s.get("proficiency") in ("expert", "advanced")
        and any(req in s["name"].lower() for req in JD_REQUIRED)
    ][:3]

    sig = cand.get("redrob_signals", {})
    open_work = sig.get("open_to_work_flag", False)

    parts = [f"{yoe:.0f}yr {title}" if isinstance(yoe, float) else f"{yoe}yr {title}"]
    if company:
        parts[0] += f" at {company}"
    if top_skills:
        parts.append(f"strong on {', '.join(top_skills)}")
    if loc:
        parts.append(f"{loc}-based")
    if open_work:
        parts.append("actively seeking")
    if scores["hireability"] > 0.7:
        parts.append("high hireability")
    parts.append(f"composite {final:.3f}")

    return "; ".join(parts) + "."


# ── Main ─────────────────────────────────────────────────────────────────────

def rank_candidates(out_path: str):
    import time
    t0 = time.time()

    print("Loading FAISS index...")
    index = faiss.read_index(str(PRECOMPUTED / "candidates.faiss"))

    print("Loading candidate metadata...")
    with open(PRECOMPUTED / "metadata.pkl", "rb") as f:
        data = pickle.load(f)
    ids      = data["ids"]
    metadata = data["metadata"]

    print("Loading embedding model (no network — model must be cached)...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding JD...")
    jd_vec = model.encode(
        [JD_TEXT], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    print(f"FAISS search: retrieving top {TOP_K_RETRIEVE}...")
    faiss_scores, faiss_idx = index.search(jd_vec, TOP_K_RETRIEVE)

    print("Reranking with multi-factor scoring (semantic + skill + XP + career + hireability)...")
    results = []
    for fs, ci in zip(faiss_scores[0], faiss_idx[0]):
        cand = metadata[ci]
        p    = cand["profile"]

        s_sem   = float(fs)
        s_skill = skill_score(cand["skills"])
        s_exp   = experience_score(p.get("years_of_experience"))
        s_loc   = location_score(p)
        s_car   = career_signal_score(cand["career_history"])
        s_hire  = hireability_score(cand["redrob_signals"])
        t_pen   = trap_penalty(cand)

        composite = (
            0.28 * s_sem   +
            0.25 * s_skill +
            0.12 * s_exp   +
            0.05 * s_loc   +
            0.12 * s_car   +
            0.18 * s_hire
        ) * (1.0 - 0.90 * t_pen)   # heavy penalty; traps sink below rank 100

        composite = min(1.0, max(0.0, composite))

        results.append({
            "candidate_id": ids[ci],
            "score": composite,
            "candidate": cand,
            "scores": {
                "semantic": s_sem, "skill": s_skill, "experience": s_exp,
                "location": s_loc, "career": s_car,
                "hireability": s_hire, "trap_penalty": t_pen,
            },
        })

    # Sort descending
    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:TOP_N_SUBMIT]

    # Guarantee spec: non-increasing scores
    for i in range(1, len(top)):
        if top[i]["score"] > top[i - 1]["score"]:
            top[i]["score"] = top[i - 1]["score"]

    # Write CSV
    out = Path(out_path)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_num, r in enumerate(top, 1):
            reasoning = make_reasoning(r["candidate"], r["scores"], r["score"])
            writer.writerow([r["candidate_id"], rank_num, f"{r['score']:.6f}", reasoning])

    elapsed = time.time() - t0
    trap_count = sum(1 for r in top if r["scores"]["trap_penalty"] > 0.4)
    avg_hire   = sum(r["scores"]["hireability"] for r in top) / len(top)

    print(f"\n✅ Submission saved: {out}")
    print(f"⏱️  Total runtime   : {elapsed:.1f}s")
    print(f"🪤  Traps in top-100: {trap_count}  (must be <10 to avoid DQ)")
    print(f"📊  Avg hireability : {avg_hire:.3f}")
    print(f"🥇  Top candidate   : {top[0]['candidate_id']}  score={top[0]['score']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HireabilityRank — submission ranker")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path (use your team ID as filename)")
    args = parser.parse_args()
    rank_candidates(args.out)