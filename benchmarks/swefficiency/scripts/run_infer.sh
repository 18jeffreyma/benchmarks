#!/bin/bash
# SWE-fficiency inference runner script
# This script provides a convenient wrapper for running SWE-fficiency evaluations

set -e

# Default values
LLM_CONFIG="${1:-.llm_config/example.json}"
DATASET="${2:-princeton-nlp/SWE-bench_Verified}"
SPLIT="${3:-test}"
MAX_ITERATIONS="${4:-100}"
WORKSPACE="${5:-docker}"

echo "Running SWE-fficiency inference with:"
echo "  LLM Config: $LLM_CONFIG"
echo "  Dataset: $DATASET"
echo "  Split: $SPLIT"
echo "  Max Iterations: $MAX_ITERATIONS"
echo "  Workspace: $WORKSPACE"

uv run swefficiency-infer "$LLM_CONFIG" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --max-iterations "$MAX_ITERATIONS" \
    --workspace "$WORKSPACE"

