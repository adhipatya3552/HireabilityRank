# HireabilityRank 🎯
**AI Candidate Ranker with Dual-Axis Scoring: Skill Fit × Hireability**

Built for the **INDIA RUNS Data & AI Challenge** (Redrob Hackathon)

---

## The Core Idea

Most rankers score only on **skill match**. The real problem: a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is **not actually available**.

HireabilityRank scores on **two axes**:

| Axis | Weight | What it measures |
|------|--------|-----------------|
| Skill Fit | 65% | Semantic match + required skills + experience + career tilt |
| Hireability | 35% | Will they respond? Interview? Accept? (23 behavioral signals) |

This naturally eliminates keyword stuffers and honeypots before they enter the top 100.

---

## Architecture

```
candidates.jsonl (100K)
        ↓ precompute.py (run once, ~25 min)
FAISS index + compact metadata
        ↓ rank.py (<5 min, CPU, no network)
Top-5000 via FAISS cosine search
        ↓ Multi-factor reranking
Top-100 CSV with reasoning
```

---

## Scoring Formula

```python
composite = (
    0.28 * semantic_score    # JD embedding vs candidate text (FAISS)
  + 0.25 * skill_score       # Required skills + anti-keyword-stuffer trust multiplier
  + 0.12 * experience_score  # Sweet spot 5-9 years
  + 0.12 * career_signal     # Product-engineering tilt (not researcher)
  + 0.18 * hireability_score # 23 behavioral signals
) * (1.0 - 0.90 * trap_penalty)  # Kills honeypots
```

---

## Trap Detection (6-flag system)

| Flag | What it detects |
|------|----------------|
| github_activity = 0 | Inactive / fake Senior AI Engineer |
| views=0 + skills>12 | Keyword-stuffed invisible profile |
| avg endorsements<2.5 + duration<8mo | Skill list padding |
| recruiter_response_rate = 0 | Ghost profile |
| completeness>92 + saves=0 + views=0 | Impossible perfect profile |
| Not verified on email or phone | Untrustworthy identity |

6/6 flags → 90% score penalty → never reaches top 100.

---

## Quick Start

### Step 1 — Precompute (run once)
```bash
pip install -r requirements.txt
python precompute.py --candidates candidates.jsonl
# Takes ~20-25 min; creates precomputed/ directory
```

### Step 2 — Rank (run for submission)
```bash
python rank.py --out YOUR_TEAM_ID.csv
# Runs in <5 min on CPU, no network needed
```

### Step 3 — Validate
```bash
python validate_submission.py YOUR_TEAM_ID.csv
```

---

## Demo

Live sandbox: [HuggingFace Space](https://huggingface.co/spaces/adhipatya3552/hireabilityrank)

Upload a small JSON sample of candidates and see ranked output with scores.

---

## Compute Environment
- Model: `all-MiniLM-L6-v2` (80MB, CPU-fast, 384-dim, no GPU needed)
- FAISS: `IndexFlatIP` (exact cosine search on normalized vectors)
- Runtime: ~90 seconds for 5000 candidates on 8-core CPU
- Memory: <4 GB during ranking
- No network calls during ranking