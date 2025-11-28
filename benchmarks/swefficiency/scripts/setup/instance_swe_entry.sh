#!/bin/bash
# Instance-specific SWE entry point for SWE-fficiency evaluation
# This script is called for each evaluation instance

set -e

INSTANCE_ID="${1:-}"
REPO_PATH="${2:-/workspace}"

if [ -z "$INSTANCE_ID" ]; then
    echo "Error: INSTANCE_ID is required"
    exit 1
fi

echo "Setting up evaluation environment for instance: $INSTANCE_ID"
echo "Repository path: $REPO_PATH"

# Change to repository directory
cd "$REPO_PATH" || exit 1

# Run any instance-specific setup
# This can be customized per instance if needed

echo "Instance setup complete for $INSTANCE_ID"

