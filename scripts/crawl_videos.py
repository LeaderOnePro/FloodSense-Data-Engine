#!/usr/bin/env python3
"""
Script to crawl flood-related videos from YouTube.

Usage:
    python scripts/crawl_videos.py --keywords "flood disaster" "flash flood"
"""

import argparse
from pathlib import Path

from loguru import logger

from floodsense.crawlers.video_crawler import VideoCrawler
from floodsense.utils.config import Config


def main():
    """Main function for video crawling."""
    parser = argparse.ArgumentParser(
        description="Crawl flood-related videos from YouTube"
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
        default=10,
        help="Maximum videos per keyword (default: 10)",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="youtube",
        help="Video platform (default: youtube)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for downloaded videos",
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

    # Initialize crawler
    crawler = VideoCrawler(
        config=config.crawler,
        output_dir=args.output_dir,
    )

    # Crawl videos
    logger.info(f"Starting video crawl for keywords: {args.keywords}")
    downloaded_paths = crawler.crawl(
        keywords=args.keywords,
        max_results=args.max_results,
        platform=args.platform,
    )

    logger.info(f"Downloaded {len(downloaded_paths)} videos")


if __name__ == "__main__":
    main()