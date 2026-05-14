#!/usr/bin/env bash
#
# Fix /testbed permissions in all swefficiency base images.
#
# For each image:
#   1. Pull the image
#   2. Run chmod -R a+rX /testbed as root
#   3. Commit and push the fixed image
#
# Usage:
#   ./scripts/fix_image_permissions.sh                  # fix all 498 images
#   ./scripts/fix_image_permissions.sh --dry-run        # just report which need fixing
#   ./scripts/fix_image_permissions.sh --ids ids.txt    # fix only listed instance IDs
#
set -euo pipefail

REGISTRY="ghcr.io/swefficiency/swefficiency-images"
INSTANCE_IDS_FILE=""
DRY_RUN=false
PARALLEL=16

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true; shift ;;
        --ids)        INSTANCE_IDS_FILE="$2"; shift 2 ;;
        --parallel)   PARALLEL="$2"; shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Get instance IDs
if [[ -n "$INSTANCE_IDS_FILE" ]]; then
    mapfile -t IDS < "$INSTANCE_IDS_FILE"
else
    echo "Loading instance IDs from dataset..."
    mapfile -t IDS < <(python3 -c "
from datasets import load_dataset
ds = load_dataset('swefficiency/swefficiency', split='test')
for row in ds:
    print(row['instance_id'])
" 2>/dev/null)
fi

echo "Total images: ${#IDS[@]}"
echo "Dry run: $DRY_RUN"
echo ""

FIXED=0
SKIPPED=0
FAILED=0

fix_image() {
    local iid="$1"
    local image="${REGISTRY}:${iid}"

    # Pull
    if ! docker pull "$image" >/dev/null 2>&1; then
        echo "FAIL  $iid  (pull failed)"
        return 1
    fi

    # Check if any files under /testbed are not world-readable
    local bad_files
    bad_files=$(docker run --rm "$image" find /testbed -not -perm -o+r 2>/dev/null | head -1)

    if [[ -z "$bad_files" ]]; then
        echo "OK    $iid"
        # Clean up pulled image to save space
        docker rmi "$image" >/dev/null 2>&1 || true
        return 2  # signal: skipped
    fi

    if $DRY_RUN; then
        echo "FIX   $iid  (needs permission fix)"
        docker rmi "$image" >/dev/null 2>&1 || true
        return 0
    fi

    # Fix permissions: run chmod as root, commit, push
    local container_id
    container_id=$(docker run -d "$image" chmod -R a+rX /testbed)
    docker wait "$container_id" >/dev/null 2>&1

    # Commit the fixed container back to the same image tag
    docker commit "$container_id" "$image" >/dev/null 2>&1
    docker rm "$container_id" >/dev/null 2>&1

    # Push
    if docker push "$image" >/dev/null 2>&1; then
        echo "FIXED $iid"
    else
        echo "FAIL  $iid  (push failed)"
        docker rmi "$image" >/dev/null 2>&1 || true
        return 1
    fi

    # Clean up to save disk space
    docker rmi "$image" >/dev/null 2>&1 || true
    return 0
}

for iid in "${IDS[@]}"; do
    set +e
    fix_image "$iid"
    rc=$?
    set -e
    case $rc in
        0) FIXED=$((FIXED + 1)) ;;
        1) FAILED=$((FAILED + 1)) ;;
        2) SKIPPED=$((SKIPPED + 1)) ;;
    esac
done

echo ""
echo "Done. Fixed: $FIXED, Skipped (already OK): $SKIPPED, Failed: $FAILED"
