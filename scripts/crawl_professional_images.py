#!/usr/bin/env python3
"""
Script to crawl flood-related images from professional sources.

Supports Unsplash, Pexels, Flickr, Wikimedia Commons, and NASA Earth Observatory.

Usage:
    python scripts/crawl_professional_images.py --keywords "flood" "hurricane damage"
    python scripts/crawl_professional_images.py --keywords "flood" --sources unsplash pexels
    python scripts/crawl_professional_images.py --keywords "flood" --max-results 50
"""

import argparse
from pathlib import Path
from typing import Optional

from loguru import logger

from floodsense.crawlers.api_crawlers.multi_source_crawler import MultiSourceCrawler
from floodsense.utils.config import Config
from floodsense.validators.image_validator import ImageValidator


AVAILABLE_SOURCES = ["unsplash", "pexels", "nasa", "wikimedia", "flickr"]


def main():
    """Main function for professional image crawling."""
    parser = argparse.ArgumentParser(
        description="Crawl flood-related images from professional sources "
        "(Unsplash, Pexels, Flickr, Wikimedia, NASA)"
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
        help="Maximum total images per keyword (default: 100)",
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
        "--sources",
        nargs="+",
        choices=AVAILABLE_SOURCES,
        default=None,
        help=f"Specific sources to use (default: all enabled). "
        f"Available: {', '.join(AVAILABLE_SOURCES)}",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable content validation",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show source status and exit",
    )

    args = parser.parse_args()

    # Load configuration
    config = Config.load(args.config)

    # Initialize multi-source crawler
    crawler = MultiSourceCrawler(
        config=config,
        output_dir=args.output_dir,
        sources=args.sources,
    )

    # Show status if requested
    if args.status:
        status = crawler.get_source_status()
        logger.info("Source status:")
        for source, info in status.items():
            enabled = "ENABLED" if info["enabled"] else "DISABLED"
            priority = info.get("priority", "N/A")
            remaining = info.get("remaining_requests", "N/A")
            logger.info(
                f"  {source}: {enabled} | priority: {priority} | "
                f"remaining requests: {remaining}"
            )
        return

    if not crawler.enabled_sources:
        logger.error(
            "No sources are enabled. Set API keys via environment variables:\n"
            "  FLOODSENSE_UNSPLASH_API_KEY=your_key\n"
            "  FLOODSENSE_PEXELS_API_KEY=your_key\n"
            "  FLOODSENSE_FLICKR_API_KEY=your_key\n"
            "Or enable keyless sources (nasa, wikimedia) in config."
        )
        return

    # Log enabled sources
    logger.info(f"Enabled sources: {', '.join(crawler.enabled_sources)}")
    logger.info(f"Keywords: {args.keywords}")
    logger.info(f"Max results per keyword: {args.max_results}")

    # Crawl images
    downloaded_paths = crawler.crawl(
        keywords=args.keywords,
        max_results=args.max_results,
    )

    logger.info(f"Downloaded {len(downloaded_paths)} images total")

    # Validate if enabled
    if not args.no_validation and config.validator.enabled and downloaded_paths:
        logger.info("Running content validation on downloaded images...")
        validator = ImageValidator(
            clip_threshold=config.validator.clip_threshold,
            clip_model_name=config.validator.clip_model,
            enable_heuristic=config.validator.enable_heuristic,
            enable_clip=config.validator.enable_clip,
            device=config.validator.device,
        )

        valid_count = 0
        removed_count = 0
        for path in downloaded_paths:
            if path.exists():
                is_valid = validator.validate(path, args.keywords)
                if is_valid:
                    valid_count += 1
                else:
                    path.unlink()
                    removed_count += 1

        logger.info(
            f"Validation complete: {valid_count} valid, {removed_count} removed"
        )

    # Final status
    final_status = crawler.get_source_status()
    for source, info in final_status.items():
        if info["enabled"] and info.get("remaining_requests") is not None:
            logger.info(
                f"  {source}: {info['remaining_requests']} requests remaining"
            )


if __name__ == "__main__":
    main()
