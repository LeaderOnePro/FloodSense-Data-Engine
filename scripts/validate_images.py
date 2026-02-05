#!/usr/bin/env python3
"""
Script to validate processed images using CLIP model.

Usage:
    python scripts/validate_images.py --input-dir data/processed/images
"""

import argparse
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from floodsense.validators.image_validator import ImageValidator
from floodsense.utils.config import Config
from floodsense.utils.file_utils import FileUtils


def main():
    parser = argparse.ArgumentParser(
        description="Validate processed images using CLIP model"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing images to validate",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=["flood", "flooding", "water damage", "disaster", "inundation"],
        help="Keywords for CLIP validation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't delete invalid images, just report",
    )

    args = parser.parse_args()

    # Load configuration
    config = Config.load(args.config)

    # Initialize validator
    validator = ImageValidator(
        clip_threshold=config.validator.clip_threshold,
        clip_model_name=config.validator.clip_model,
        enable_heuristic=config.validator.enable_heuristic,
        enable_clip=config.validator.enable_clip,
        device=config.validator.device,
    )

    # Collect all images
    image_paths = list(FileUtils.iter_images(args.input_dir))
    logger.info(f"Found {len(image_paths)} images to validate")

    valid_count = 0
    invalid_count = 0
    invalid_paths = []

    # Validate each image
    with tqdm(image_paths, desc="Validating images") as pbar:
        for img_path in pbar:
            is_valid = validator.validate(img_path, args.keywords)
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                invalid_paths.append(img_path)
                if not args.dry_run:
                    img_path.unlink()
            pbar.set_postfix({"valid": valid_count, "invalid": invalid_count})

    # Report
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total images: {len(image_paths)}")
    logger.info(f"Valid: {valid_count}")
    logger.info(f"Invalid: {invalid_count}")
    logger.info(f"Pass rate: {valid_count / len(image_paths) * 100:.1f}%")

    if args.dry_run:
        logger.info("(Dry run - no images were deleted)")
    else:
        logger.info(f"Deleted {invalid_count} invalid images")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
