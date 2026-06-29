#!/usr/bin/env python3
"""
Local sanity-check — no internet, no large dataset needed.
Tests every scoring component on the 50-sample file,
produces a dummy 100-row CSV, and runs the validator.

Usage:
    python test_local.py
"""
import csv
import json
import re
import sys
from collections import Counter

sys.path.insert(0, ".")
from rank import (
    JD_NICE, JD_REQUIRED, JD_TEXT, TOP_N_SUBMIT,
    career_signal_score, experience_score, hireability_score,
    location_score, make_reasoning, skill_score, trap_penalty,
)

SAMPLE_FILE = "sample_candidates.json"
OUT_FILE    = "local_test_submission.csv"

# ── Lightweight TF-IDF semantic proxy (no model download) ─────────────────
_RE = re.compile(r"[a-zA-Z]{3,}")
_JD_TOKENS = Counter(t.lower() for t in _RE.findall(JD_TEXT))
_JD_SIZE   = sum(_JD_TOKENS.values())

def _tfidf_sim(c: dict) -> float:
    p = c.get("profile", {})
    text = " ".join([
        p.get("headline", ""), p.get("summary", ""),
        " ".join(s.get("name", "") for s in c.get("skills", [])),
        " ".join(j.get("title", "") + " " + j.get("description", "")[:200]
                 for j in c.get("career_history", [])[:3]),
    ])
    tokens = Counter(t.lower() for t in _RE.findall(text))
    dot = sum(tokens[t] * _JD_TOKENS[t] for t in tokens if t in _JD_TOKENS)
    denom = (sum(v**2 for v in tokens.values()) ** 0.5) * (sum(v**2 for v in _JD_TOKENS.values()) ** 0.5)
    return dot / denom if denom else 0.0


def run_test():
    print(f"Loading {SAMPLE_FILE} ...")
    with open(SAMPLE_FILE) as f:
        candidates = json.load(f)
    print(f"Loaded {len(candidates)} candidates\n")

    # ── Score every candidate ──────────────────────────────────────────────
    results = []
    for c in candidates:
        p = c.get("profile", {})
        sem = _tfidf_sim(c)
        sk  = skill_score(c.get("skills", []))
        exp = experience_score(p.get("years_of_experience"))
        loc = location_score(p)
        car = career_signal_score(c.get("career_history", []))
        hi  = hireability_score(c.get("redrob_signals", {}))
        tp  = trap_penalty(c)

        composite = (
            0.28 * sem +
            0.25 * sk  +
            0.12 * exp +
            0.05 * loc +
            0.12 * car +
            0.18 * hi
        ) * (1.0 - 0.90 * tp)
        composite = min(1.0, max(0.0, composite))

        results.append({
            "candidate_id": c["candidate_id"],
            "score": composite,
            "candidate": c,
            "scores": {
                "semantic": sem, "skill": sk, "experience": exp,
                "location": loc, "career": car,
                "hireability": hi, "trap_penalty": tp,
            },
        })

    # Tie-break: score desc, candidate_id asc
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # ── Print top-10 ──────────────────────────────────────────────────────
    print("── Top-10 Ranked Candidates ──────────────────────────────────────")
    print(f"{'#':<4} {'ID':<14} {'Title':<28} {'Score':>6} {'Sem':>5} {'Skill':>6} {'Hire':>5} {'Trap':>5}")
    print("─" * 80)
    for i, r in enumerate(results[:10], 1):
        s = r["scores"]
        title = r["candidate"]["profile"].get("current_title", "")[:27]
        print(
            f"{i:<4} {r['candidate_id']:<14} {title:<28} "
            f"{r['score']:>6.3f} {s['semantic']:>5.2f} {s['skill']:>6.2f} "
            f"{s['hireability']:>5.2f} {s['trap_penalty']:>5.2f}"
        )

    traps = [r for r in results if r["scores"]["trap_penalty"] > 0.4]
    print(f"\n🪤  Traps detected in sample: {len(traps)}")

    # ── Generate a valid 100-row CSV for format testing ───────────────────
    # Pad with mock IDs if sample < 100
    ids_used = {r["candidate_id"] for r in results}
    padded = list(results)
    mock_idx = 90001
    while len(padded) < 100:
        cid = f"CAND_{mock_idx:07d}"
        if cid not in ids_used:
            padded.append({"candidate_id": cid, "score": 0.001, "candidate": {}, "scores": {}})
        mock_idx += 1

    # Non-increasing scores
    for i in range(1, len(padded)):
        if padded[i]["score"] > padded[i - 1]["score"]:
            padded[i]["score"] = padded[i - 1]["score"]

    print(f"\nWriting {OUT_FILE} ...")
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_num, r in enumerate(padded[:100], 1):
            reasoning = ""
            if r.get("candidate") and r["candidate"].get("profile"):
                reasoning = make_reasoning(r["candidate"], r["scores"], r["score"])
            else:
                reasoning = f"Padded mock candidate; score {r['score']:.4f}."
            w.writerow([r["candidate_id"], rank_num, f"{r['score']:.6f}", reasoning])

    # ── Validate format ───────────────────────────────────────────────────
    print("Running validator ...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("val", "validate_submission.py")
    val  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(val)
    errors = val.validate_submission(OUT_FILE)
    if errors:
        print(f"\n❌ Validation FAILED ({len(errors)} issues):")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)
    else:
        print(f"\n✅ Format validation PASSED → {OUT_FILE}")
        print("All scoring components working correctly.")
        print("\nNext steps:")
        print("  1. python download_model.py        # needs internet once")
        print("  2. python precompute.py --candidates candidates.jsonl")
        print("  3. python rank.py --out YOUR_TEAM_ID.csv")
        print("  4. python validate_submission.py YOUR_TEAM_ID.csv")


if __name__ == "__main__":
    run_test()
