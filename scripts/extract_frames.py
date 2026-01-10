#!/usr/bin/env python3
"""
Script to extract keyframes from videos with scene detection.

Usage:
    python scripts/extract_frames.py --input-dir data/raw --output-dir data/processed/video_frames
"""

import argparse
from pathlib import Path

from loguru import logger

from floodsense.processors.video_processor import VideoProcessor
from floodsense.utils.config import Config


def main():
    """Main function for video frame extraction."""
    parser = argparse.ArgumentParser(
        description="Extract keyframes from videos with scene detection"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save extracted frames",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=30.0,
        help="Scene change detection threshold (default: 30.0)",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Blur detection threshold (default: 100.0)",
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

    # Initialize processor
    processor = VideoProcessor(config=config.processor)

    # Extract frames
    logger.info(f"Extracting frames from videos in {args.input_dir}")
    extracted_paths = processor.process_video_directory(
        video_dir=args.input_dir,
        output_dir=args.output_dir,
        scene_threshold=args.scene_threshold,
        blur_threshold=args.blur_threshold,
    )

    logger.info(f"Extracted {len(extracted_paths)} keyframes")


if __name__ == "__main__":
    main()