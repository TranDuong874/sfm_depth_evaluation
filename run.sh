#!/bin/bash
# SfM Depth Evaluation Pipeline Runner
# Sets environment variables and runs the pipeline

# Fix COLMAP Qt display issue (no GUI needed)
export QT_QPA_PLATFORM=offscreen

# Default values
MAX_SEQUENCES=""
N_VIEWS="5 10 20"
METHODS=""
SEQUENCE=""
PHASES="1 2 3 4 5"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-sequences)
            MAX_SEQUENCES="--max-sequences $2"
            shift 2
            ;;
        --n-views)
            N_VIEWS="$2"
            shift 2
            ;;
        --sequence)
            SEQUENCE="--sequence $2"
            shift 2
            ;;
        --methods)
            METHODS="--methods $2"
            shift 2
            ;;
        --phases)
            PHASES="$2"
            shift 2
            ;;
        --quick)
            # Quick test: 2 sequences, 5 views only
            MAX_SEQUENCES="--max-sequences 2"
            N_VIEWS="5"
            shift
            ;;
        -h|--help)
            echo "Usage: ./run.sh [options]"
            echo ""
            echo "Options:"
            echo "  --max-sequences N   Limit number of sequences to process"
            echo "  --n-views \"5 10 20\" View counts to process (space-separated in quotes)"
            echo "  --sequence SEQ_ID   Process specific sequence only"
            echo "  --methods \"m1 m2\"   Methods to evaluate (space-separated in quotes)"
            echo "  --phases \"1 2 3 4 5\" Phases to run (space-separated in quotes)"
            echo "  --quick             Quick test: 2 sequences, 5 views"
            echo ""
            echo "Examples:"
            echo "  ./run.sh --quick                    # Quick test"
            echo "  ./run.sh --n-views \"5 10\"           # 5 and 10 views"
            echo "  ./run.sh --phases \"2 3 4\"           # Run only phases 2-4"
            echo "  ./run.sh --max-sequences 10         # First 10 sequences"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert n-views to command line format
N_VIEWS_ARG="--n-views $N_VIEWS"

echo "=========================================="
echo "SfM Depth Evaluation Pipeline"
echo "=========================================="
echo "N-views: $N_VIEWS"
echo "Phases: $PHASES"
[[ -n "$MAX_SEQUENCES" ]] && echo "Max sequences: $MAX_SEQUENCES"
[[ -n "$SEQUENCE" ]] && echo "Sequence: $SEQUENCE"
[[ -n "$METHODS" ]] && echo "Methods: $METHODS"
echo "=========================================="
echo ""

# Run phases
for phase in $PHASES; do
    case $phase in
        1)
            echo "[Phase 1] Sampling..."
            python scripts/phase1_sampling.py \
                --co3d-root data/co3d \
                --output output \
                $N_VIEWS_ARG \
                $MAX_SEQUENCES \
                $SEQUENCE
            ;;
        2)
            echo "[Phase 2] SfM..."
            python scripts/phase2_sfm.py \
                --input output/phase1_sampled \
                --output output/phase2_sfm \
                $N_VIEWS_ARG \
                $SEQUENCE
            ;;
        3)
            echo "[Phase 3] Depth Estimation..."
            python scripts/phase3_depth.py \
                --sampled output/phase1_sampled \
                --sfm output/phase2_sfm \
                --output output/phase3_depth \
                $N_VIEWS_ARG \
                $SEQUENCE
            ;;
        4)
            echo "[Phase 4] Reconstruction..."
            METHODS_ARG=""
            [[ -n "$METHODS" ]] && METHODS_ARG="--methods $METHODS"
            python scripts/phase4_reconstruction.py \
                --sampled output/phase1_sampled \
                --sfm output/phase2_sfm \
                --depth output/phase3_depth \
                --output output/phase4_reconstruction \
                $N_VIEWS_ARG \
                $METHODS_ARG \
                $SEQUENCE
            ;;
        5)
            echo "[Phase 5] Evaluation..."
            METHODS_ARG=""
            [[ -n "$METHODS" ]] && METHODS_ARG="--methods $METHODS"
            python scripts/phase5_evaluation.py \
                --reconstructions output/phase4_reconstruction \
                --sampled output/phase1_sampled \
                --co3d data/co3d \
                --output output/phase5_evaluation \
                $N_VIEWS_ARG \
                $METHODS_ARG \
                $SEQUENCE
            ;;
    esac
    echo ""
done

echo "=========================================="
echo "Pipeline complete!"
echo "Results: output/phase5_evaluation/results.csv"
echo "=========================================="
