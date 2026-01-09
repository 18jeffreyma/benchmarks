#!/usr/bin/env python3
"""
Build agent-server images for all unique SWE-fficiency base images in a dataset split.

Example:
  uv run benchmarks/swefficiency/build_images.py \
    --dataset swefficiency/swefficiency --split test \
    --image ghcr.io/openhands/eval-agent-server --target source-minimal
"""

import sys

from benchmarks.utils.build_utils import (
    build_all_images,
    default_build_output_dir,
    get_build_parser,
)
from benchmarks.utils.dataset import get_dataset
from openhands.sdk import get_logger


logger = get_logger(__name__)


def get_official_docker_image(
    instance_id: str,
) -> str:
    """Get the official SWE-fficiency Docker image for an instance."""
    return f"ghcr.io/swefficiency/swefficiency-images:{instance_id}"


def extract_custom_tag(base_image: str) -> str:
    """
    Extract SWE-fficiency instance ID from official SWE-fficiency image name.

    Example:
        ghcr.io/swefficiency/swefficiency-images:scikit-learn__scikit-learn-11674
        -> scikit-learn__scikit-learn-11674
    """
    name_tag = base_image.split("/")[-1]
    if ":" in name_tag:
        return name_tag.split(":")[1]
    return name_tag


def collect_unique_base_images(
    dataset,
    split,
    n_limit,
    selected_instances_file: str | None = None,
):
    """Collect unique base images from the dataset."""
    df = get_dataset(
        dataset_name=dataset,
        split=split,
        eval_limit=n_limit if n_limit else None,
        selected_instances_file=selected_instances_file,
    )
    return sorted(
        {get_official_docker_image(str(row["instance_id"])) for _, row in df.iterrows()}
    )


def main(argv: list[str]) -> int:
    parser = get_build_parser()
    # Override the default dataset for SWE-fficiency
    parser.set_defaults(dataset="swefficiency/swefficiency", split="test")
    args = parser.parse_args(argv)

    base_images: list[str] = collect_unique_base_images(
        args.dataset,
        args.split,
        args.n_limit,
        args.select,
    )
    build_dir = default_build_output_dir(args.dataset, args.split)

    return build_all_images(
        base_images=base_images,
        target=args.target,
        build_dir=build_dir,
        image=args.image,
        push=args.push,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        base_image_to_custom_tag_fn=extract_custom_tag,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
