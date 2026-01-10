#!/usr/bin/env python3
"""
Script to clean and process flood imagery data.

Usage:
    python scripts/clean_data.py --input-dir data/raw --output-dir data/processed
"""

import argparse
from pathlib import Path

from loguru import logger

from floodsense.processors.cleaning_pipeline import CleaningPipeline
from floodsense.utils.config import Config


def main():
    """Main function for data cleaning."""
    parser = argparse.ArgumentParser(
        description="Clean and process flood imagery data"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--no-video-frames",
        action="store_true",
        help="Skip video frame extraction",
    )
    parser.add_argument(
        "--no-blur-removal",
        action="store_true",
        help="Skip blur detection and removal",
    )
    parser.add_argument(
        "--no-deduplication",
        action="store_true",
        help="Skip duplicate removal",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Load configuration
    config = Config.load(args.config)

    # Initialize pipeline
    pipeline = CleaningPipeline(config=config.processor)

    # Run pipeline
    logger.info(f"Starting cleaning pipeline on {args.input_dir}")
    stats = pipeline.run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        extract_video_frames=not args.no_video_frames,
        remove_blur=not args.no_blur_removal,
        deduplicate=not args.no_deduplication,
    )

    logger.info("Cleaning pipeline completed")


if __name__ == "__main__":
    main()