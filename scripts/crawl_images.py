#!/usr/bin/env python3
"""
Script to crawl flood-related images from the web.

Usage:
    python scripts/crawl_images.py --keywords "flood trapped people" "submerged car"
"""

import argparse
from pathlib import Path

from loguru import logger

from floodsense.crawlers.image_spider import ImageSpider
from floodsense.utils.config import Config
from floodsense.validators.image_validator import ImageValidator


def main():
    """Main function for image crawling."""
    parser = argparse.ArgumentParser(
        description="Crawl flood-related images from the web"
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        required=True,
        help="List of search keywords",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        help="Maximum images per keyword (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for downloaded images",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume capability",
    )

    args = parser.parse_args()

    # Load configuration
    config = Config.load(args.config)

    # Initialize validator
    validator: Optional[ImageValidator] = None
    if config.validator.enabled:
        validator = ImageValidator(
            clip_threshold=config.validator.clip_threshold,
            clip_model_name=config.validator.clip_model,
            enable_heuristic=config.validator.enable_heuristic,
            enable_clip=config.validator.enable_clip,
            device=config.validator.device,
        )

    # Initialize crawler
    crawler = ImageSpider(
        config=config.crawler,
        output_dir=args.output_dir,
        validator=validator,
    )

    # Crawl images
    logger.info(f"Starting image crawl for keywords: {args.keywords}")
    downloaded_paths = crawler.crawl(
        keywords=args.keywords,
        max_results=args.max_results,
        enable_resume=not args.no_resume,
    )

    logger.info(f"Downloaded {len(downloaded_paths)} images")


if __name__ == "__main__":
    main()