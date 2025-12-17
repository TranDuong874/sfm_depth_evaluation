#!/bin/bash
set -e

# Config
SEQ_PATH="animal-related_objects/dog_collar/57-41030-21122"
SEQ_ID="animal-related_objects_dog_collar_57-41030-21122"
VIEWS=5
PYTHON=".venv/bin/python"

echo "================================================================"
echo "TESTING FULL PIPELINE"
echo "Sequence: $SEQ_ID"
echo "Views: $VIEWS"
echo "================================================================"

# 1. Sampling
echo -e "\n[Phase 1] Sampling..."
$PYTHON scripts/phase1_sampling.py \
    --co3d-root data/co3d \
    --output output \
    --sequence "$SEQ_PATH" \
    --n-views $VIEWS

# 2. SfM
echo -e "\n[Phase 2] SfM (COLMAP + MASt3R)..."
$PYTHON scripts/phase2_sfm.py \
    --input output/phase1_sampled \
    --output output/phase2_sfm \
    --sequence "$SEQ_ID" \
    --n-views $VIEWS \
    --methods colmap mast3r

# 3. Depth
echo -e "\n[Phase 3] Depth Estimation (All Methods)..."
$PYTHON scripts/phase3_depth.py \
    --sampled output/phase1_sampled \
    --sfm output/phase2_sfm \
    --output output/phase3_depth \
    --sequence "$SEQ_ID" \
    --n-views $VIEWS \
    --sfm-methods colmap mast3r \
    --depth-methods murre metric3d unidepth depth_anything

# 4. Reconstruction
echo -e "\n[Phase 4] Reconstruction..."
METHODS=(
    "colmap_murre" "colmap_metric3d" "colmap_unidepth" "colmap_depth_anything"
    "mast3r_murre" "mast3r_metric3d" "mast3r_unidepth" "mast3r_depth_anything"
    "colmap_sparse" "mast3r_sparse"
)
$PYTHON scripts/phase4_reconstruction.py \
    --sampled output/phase1_sampled \
    --sfm output/phase2_sfm \
    --depth output/phase3_depth \
    --output output/phase4_reconstruction \
    --sequence "$SEQ_ID" \
    --n-views $VIEWS \
    --methods "${METHODS[@]}"

# 5. Evaluation
echo -e "\n[Phase 5] Evaluation..."
$PYTHON scripts/phase5_evaluation.py \
    --reconstructions output/phase4_reconstruction \
    --sampled output/phase1_sampled \
    --co3d data/co3d \
    --output output/phase5_evaluation \
    --sequence "$SEQ_ID" \
    --n-views $VIEWS \
    --methods "${METHODS[@]}"

echo -e "\n================================================================"
echo "TEST COMPLETE"
echo "================================================================"
