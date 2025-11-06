#!/bin/bash

################################################################################
# Grid-Size Invariance Test Suite
#
# Tests trained FCN checkpoint on multiple grid sizes to validate
# spatial softmax grid-size invariance claims.
#
# Usage:
#   bash test_grid_invariance.sh [checkpoint_path] [num_episodes]
#
# Examples:
#   bash test_grid_invariance.sh checkpoints/fcn_final.pt 20
#   bash test_grid_invariance.sh checkpoints/fcn_checkpoint_ep800.pt 50
################################################################################

# Configuration
CHECKPOINT=${1:-"checkpoints/fcn_final.pt"}
TEST_EPISODES=${2:-20}
OUTPUT_DIR="results_grid_invariance_$(date +%Y%m%d_%H%M%S)"

echo "================================================================================"
echo "GRID-SIZE INVARIANCE TEST SUITE"
echo "================================================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Test episodes per size: $TEST_EPISODES"
echo "Output directory: $OUTPUT_DIR"
echo "================================================================================"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Available checkpoints:"
    ls -lh checkpoints/*.pt 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting tests..."
echo ""

# Define grid sizes to test
# 20 = training size (baseline)
# 25, 30, 35, 40 = moderate scaling (1.25x to 2.0x)
# 50 = aggressive scaling (2.5x)
GRID_SIZES=(20 25 30 35 40 50)

# Test each size individually (for progress tracking)
echo "Phase 1: Testing individual grid sizes"
echo "--------------------------------------------------------------------------------"

for SIZE in "${GRID_SIZES[@]}"; do
    echo ""
    echo "Testing ${SIZE}×${SIZE} grid..."
    
    python quick_test_fcn.py \
        --checkpoint "$CHECKPOINT" \
        --grid-size "$SIZE" \
        --test-episodes "$TEST_EPISODES" \
        --output-dir "$OUTPUT_DIR" \
        --map-types empty random room
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Test failed for size $SIZE"
        exit 1
    fi
done

echo ""
echo "================================================================================"
echo "Phase 2: Generating comparison analysis"
echo "================================================================================"
echo ""

# Run comparison analysis
python quick_test_fcn.py \
    --checkpoint "$CHECKPOINT" \
    --grid-sizes "${GRID_SIZES[@]}" \
    --test-episodes "$TEST_EPISODES" \
    --output-dir "$OUTPUT_DIR" \
    --compare

echo ""
echo "================================================================================"
echo "TEST SUITE COMPLETE"
echo "================================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - $OUTPUT_DIR/comparison.json       (Overall comparison)"
echo "  - $OUTPUT_DIR/results_size_*.json   (Individual results)"
echo ""
echo "Next steps:"
echo "  1. Review comparison.json for verdict"
echo "  2. Check degradation percentages"
echo "  3. Decide on next actions based on results"
echo "================================================================================"
