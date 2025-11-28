#!/bin/bash
# SWE entry point script for SWE-fficiency evaluation
# This script sets up the environment and runs the evaluation entry point

set -e

# Activate conda environment if available
if [ -f "./opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source ./opt/miniconda3/etc/profile.d/conda.sh
    conda activate testbed || true
fi

# Run the instance-specific entry script
exec "$@"

