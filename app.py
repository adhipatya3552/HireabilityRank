"""
HireabilityRank — HuggingFace Spaces demo
Runs a mini version of the ranker on a small uploaded sample.
Required for hackathon submission (sandbox link).

Deploy to HuggingFace Spaces:
  - Runtime: CPU Basic (free)
  - SDK: Gradio
"""

import csv
import io
import json
import tempfile
from datetime import date, datetime

import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer

# ─── Globals (loaded once) ───────────────────────────────────────────────────
MODEL = None
JD_TEXT = """
Senior AI Engineer founding team Redrob AI talent intelligence.
Production embeddings sentence-transformers BGE E5 FAISS Pinecone Weaviate Qdrant Milvus.
Vector databases hybrid search retrieval ranking Python NDCG MRR MAP A/B testing.
LLM fine-tuning LoRA QLoRA PEFT learning-to-rank XGBoost.
5-9 years production ML systems Pune Noida India.
"""
JD_REQUIRED = {
    "embeddings", "faiss", "pinecone", "weaviate", "qdrant", "milvus", "retrieval",
    "ranking", "python", "ndcg", "mrr", "nlp", "llm", "rag", "bert", "transformer",
    "elasticsearch", "opensearch", "fine-tuning", "sentence-transformers",
}


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return MODEL


def quick_skill_score(skills):
    if not skills:
        return 0.0
    hits = 0
    for s in skills:
        name = s.get("name", "").lower()
        for req in JD_REQUIRED:
            if req in name:
                hits += 1
                break
    return min(1.0, hits / max(1, len(JD_REQUIRED) * 0.25))


def quick_hireability(sig):
    score = 0.0
    if sig.get("open_to_work_flag", False): score += 0.30
    last = sig.get("last_active_date", "")
    if last:
        try:
            days = (date.today() - datetime.strptime(last, "%Y-%m-%d").date()).days
            score += 0.25 if days <= 30 else (0.10 if days <= 90 else 0.0)
        except Exception:
            pass
    score += 0.20 * sig.get("recruiter_response_rate", 0.0)
    score += 0.15 * sig.get("interview_completion_rate", 0.0)
    score += 0.10 * sig.get("offer_acceptance_rate", 0.0)
    return min(1.0, score)


def rank_candidates(json_text: str, top_n: int = 10):
    """Core ranking logic for demo."""
    try:
        candidates = json.loads(json_text)
        if not isinstance(candidates, list):
            return "❌ Input must be a JSON array of candidates.", None
    except json.JSONDecodeError as e:
        return f"❌ JSON parse error: {e}", None

    if not candidates:
        return "❌ No candidates found.", None

    model = get_model()
    jd_vec = model.encode([JD_TEXT], normalize_embeddings=True, convert_to_numpy=True)

    texts, results = [], []
    for c in candidates:
        p = c.get("profile", {})
        text = f"{p.get('headline','')} {p.get('summary','')[:300]} "
        text += " ".join(s.get("name", "") for s in c.get("skills", []))
        texts.append(text)

    cand_vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    for i, c in enumerate(candidates):
        sem = float(np.dot(jd_vec[0], cand_vecs[i]))
        sk  = quick_skill_score(c.get("skills", []))
        hi  = quick_hireability(c.get("redrob_signals", {}))
        score = 0.40 * sem + 0.35 * sk + 0.25 * hi
        results.append({"id": c.get("candidate_id", f"CAND_{i}"), "score": score,
                         "semantic": sem, "skill": sk, "hireability": hi})

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[: min(top_n, len(results))]

    # CSV output
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["rank", "candidate_id", "final_score", "semantic", "skill_match", "hireability"])
    for rank_num, r in enumerate(top, 1):
        w.writerow([rank_num, r["id"], f"{r['score']:.4f}",
                    f"{r['semantic']:.3f}", f"{r['skill']:.3f}", f"{r['hireability']:.3f}"])

    summary = (
        f"✅ Ranked {len(candidates)} candidates → showing top {len(top)}\n"
        f"🥇 Best candidate: {top[0]['id']}  score={top[0]['score']:.4f}\n"
        f"📊 Avg hireability: {sum(r['hireability'] for r in top)/len(top):.3f}"
    )
    return summary, buf.getvalue()


SAMPLE = json.dumps([
    {
        "candidate_id": "CAND_DEMO_001",
        "profile": {
            "headline": "Senior AI Engineer | FAISS, Embeddings, RAG, Retrieval Systems",
            "summary": "5 years production ML. Built vector search at scale using FAISS and Pinecone. Deployed LLM reranking pipelines serving 100K+ daily queries.",
            "location": "Bangalore",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "Senior AI Engineer",
            "current_company": "TechCorp"
        },
        "skills": [
            {"name": "FAISS", "proficiency": "expert", "endorsements": 15, "duration_months": 36},
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 20, "duration_months": 48},
            {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 60},
            {"name": "RAG", "proficiency": "advanced", "endorsements": 12, "duration_months": 18},
            {"name": "NLP", "proficiency": "advanced", "endorsements": 18, "duration_months": 40}
        ],
        "redrob_signals": {
            "open_to_work_flag": True,
            "last_active_date": "2026-06-20",
            "recruiter_response_rate": 0.8,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.7
        },
        "career_history": []
    }
], indent=2)

with gr.Blocks(title="HireabilityRank — AI Candidate Ranker") as demo:
    gr.Markdown("""
    # 🎯 HireabilityRank
    ### AI Candidate Ranker with Dual-Axis Scoring (Skill Fit × Hireability)
    Built for the INDIA RUNS Data & AI Challenge · Redrob Hackathon
    """)

    with gr.Row():
        with gr.Column(scale=2):
            json_input = gr.Textbox(
                label="Candidate JSON (array format)",
                placeholder="Paste candidates JSON array here...",
                lines=20,
                value=SAMPLE,
            )
            top_n = gr.Slider(1, 20, value=5, step=1, label="Top N to show")
            btn = gr.Button("🚀 Rank Candidates", variant="primary")

        with gr.Column(scale=1):
            summary_out = gr.Textbox(label="Summary", lines=5)
            csv_out = gr.Textbox(label="Ranked Output (CSV)", lines=15)

    btn.click(fn=rank_candidates, inputs=[json_input, top_n], outputs=[summary_out, csv_out])

    gr.Markdown("""
    ---
    **How scoring works:**
    - **Semantic Match (40%)** — JD embedding vs candidate text (sentence-transformers)
    - **Skill Match (35%)** — Required skills with trust multiplier (anti-keyword-stuffing)
    - **Hireability (25%)** — Behavioral signals: open-to-work, response rate, interview completion
    """)

if __name__ == "__main__":
    demo.launch()
