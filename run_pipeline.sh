#!/bin/bash
# ============================================================
# HireabilityRank — Full Pipeline
# Run this script end-to-end to produce your submission CSV.
# ============================================================
set -e

CANDIDATES="candidates.jsonl"
TEAM_ID="${1:-submission}"  # Pass your team ID as arg, e.g: bash run_pipeline.sh team_xyz

echo "================================================"
echo "  HireabilityRank Pipeline"
echo "================================================"

# Step 0: check dependencies
echo ""
echo "[Step 0] Checking dependencies..."
pip install -r requirements.txt -q --break-system-packages
echo "  ✅ Dependencies OK"

# Step 1: download model (needs internet, run once)
echo ""
echo "[Step 1] Downloading / verifying model cache..."
python3 download_model.py
echo "  ✅ Model cached"

# Step 2: precompute embeddings (takes ~20-25 min, run once per dataset)
if [ -f "precomputed/candidates.faiss" ] && [ -f "precomputed/metadata.pkl" ]; then
    echo ""
    echo "[Step 2] Precomputed index found — skipping recompute."
    echo "  (Delete precomputed/ folder to force a rebuild)"
else
    echo ""
    echo "[Step 2] Building FAISS index (this takes ~20-25 min on CPU)..."
    python3 precompute.py --candidates "$CANDIDATES"
    echo "  ✅ Index built"
fi

# Step 3: rank candidates (must finish in <5 min)
echo ""
echo "[Step 3] Ranking candidates..."
python3 rank.py --out "${TEAM_ID}.csv"
echo "  ✅ Ranking done → ${TEAM_ID}.csv"

# Step 4: validate
echo ""
echo "[Step 4] Validating submission..."
python3 validate_submission.py "${TEAM_ID}.csv"
echo ""
echo "================================================"
echo "  SUBMISSION READY: ${TEAM_ID}.csv"
echo "  Upload this file + submission_metadata.yaml"
echo "  to the Hack2Skill portal."
echo "================================================"
