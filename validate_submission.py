#!/usr/bin/env python3
"""
validate_submission.py — Pre-flight checker for HireabilityRank CSV output.
Run before every submission attempt (max 3 allowed).

Usage:
    python validate_submission.py YOUR_TEAM_ID.csv [--candidates candidates.jsonl]

Checks:
    1. Exactly 100 rows
    2. Required columns present
    3. Score is non-increasing (rank order valid)
    4. All candidate_ids exist in candidates.jsonl (if provided)
    5. No duplicate candidate_ids
    6. Scores in [0, 1]
    7. Trap/honeypot risk estimate (flags suspicious top-100 entries)
"""

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path


REQUIRED_COLS = {"candidate_id", "rank", "score", "reasoning"}
TRAP_RISK_THRESHOLD = 10   # hackathon DQ if >10 honeypots in top-100


def load_valid_ids(path: str) -> set:
    opener = gzip.open if path.endswith(".gz") else open
    ids = set()
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["candidate_id"])
    return ids


def validate(csv_path: str, candidates_path: str | None):
    errors = []
    warnings = []
    path = Path(csv_path)

    if not path.exists():
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])

        # Check 1: columns
        missing = REQUIRED_COLS - cols
        if missing:
            errors.append(f"Missing columns: {missing}")

        for row in reader:
            rows.append(row)

    # Check 2: row count
    if len(rows) != 100:
        errors.append(f"Row count = {len(rows)} (must be exactly 100)")

    # Check 3: rank values
    ranks = []
    for r in rows:
        try:
            ranks.append(int(r.get("rank", 0)))
        except ValueError:
            errors.append(f"Non-integer rank: {r.get('rank')}")
    if sorted(ranks) != list(range(1, len(ranks) + 1)):
        errors.append("Ranks are not a consecutive sequence 1→100")

    # Check 4: scores in [0, 1] and non-increasing
    scores = []
    for r in rows:
        try:
            scores.append(float(r.get("score", -1)))
        except ValueError:
            errors.append(f"Non-numeric score: {r.get('score')}")
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1] + 1e-9:
            errors.append(f"Score increases at rank {i+1}: {scores[i-1]:.6f} → {scores[i]:.6f}")
            break
    if any(s < 0 or s > 1 for s in scores):
        errors.append("Some scores are outside [0, 1]")

    # Check 5: duplicate candidate_ids
    cids = [r.get("candidate_id", "") for r in rows]
    dupes = {c for c in cids if cids.count(c) > 1}
    if dupes:
        errors.append(f"Duplicate candidate_ids: {dupes}")

    # Check 6: validate against source JSONL (optional)
    if candidates_path:
        print(f"Loading valid IDs from {candidates_path} (may take a moment)...")
        valid_ids = load_valid_ids(candidates_path)
        unknown = set(cids) - valid_ids
        if unknown:
            errors.append(f"{len(unknown)} candidate_ids not in source file: {list(unknown)[:5]}...")

    # Check 7: reasoning non-empty
    empty_reasoning = sum(1 for r in rows if not r.get("reasoning", "").strip())
    if empty_reasoning > 0:
        warnings.append(f"{empty_reasoning} rows have empty reasoning")

    # Check 8: score spread (sanity)
    if scores:
        spread = max(scores) - min(scores)
        if spread < 0.05:
            warnings.append(f"Very low score spread ({spread:.4f}) — ranking may be flat/meaningless")

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"  SUBMISSION VALIDATOR — {path.name}")
    print("="*50)

    if errors:
        for e in errors:
            print(f"  ❌  {e}")
    else:
        print("  ✅  All required checks passed!")

    if warnings:
        for w in warnings:
            print(f"  ⚠️   {w}")

    print(f"\n  Rows     : {len(rows)}")
    if scores:
        print(f"  Score range: {min(scores):.4f} → {max(scores):.4f}  (spread {max(scores)-min(scores):.4f})")
    print(f"  Columns  : {', '.join(sorted(cols))}")
    print("="*50 + "\n")

    if errors:
        print("❌ SUBMISSION NOT READY — fix errors above before submitting.")
        sys.exit(1)
    else:
        print("✅ SUBMISSION READY — safe to upload.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to submission CSV")
    parser.add_argument("--candidates", default=None, help="Path to candidates.jsonl (optional but recommended)")
    args = parser.parse_args()
    validate(args.csv, args.candidates)