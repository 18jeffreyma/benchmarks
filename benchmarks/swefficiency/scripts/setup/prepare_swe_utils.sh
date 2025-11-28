#!/bin/bash
# Prepare SWE utilities for SWE-fficiency evaluation
# This script sets up common utilities and helper functions

set -e

echo "Preparing SWE utilities for evaluation..."

# Ensure required directories exist
mkdir -p /workspace
mkdir -p /tmp/swe-eval

# Set up common environment variables
export PIP_CACHE_DIR=~/.cache/pip
export PYTHONUNBUFFERED=1

# Activate conda environment if available
if [ -f "./opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source ./opt/miniconda3/etc/profile.d/conda.sh
    conda activate testbed || true
fi

echo "SWE utilities prepared successfully"

