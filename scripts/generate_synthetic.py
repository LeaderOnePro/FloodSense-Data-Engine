#!/usr/bin/env python3
"""
Script to generate synthetic flood images using AI.

Usage:
    python scripts/generate_synthetic.py --prompts config/prompts.json --output-dir data/synthetic
"""

import argparse
from pathlib import Path

from loguru import logger

from floodsense.synthesizers.nano_banana_client import NanoBananaClient
from floodsense.utils.config import Config


def main():
    """Main function for synthetic data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic flood images using AI"
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="Path to JSON file containing prompts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable prompt enhancement",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="synthetic",
        help="Filename prefix for generated images",
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

    # Initialize client
    client = NanoBananaClient(config=config.synthesizer)

    # Generate images
    logger.info(f"Generating images from prompts in {args.prompts}")
    generated_paths = client.generate_from_file(
        prompts_file=args.prompts,
        output_dir=args.output_dir,
        enhance_prompt=not args.no_enhance,
    )

    logger.info(f"Generated {len(generated_paths)} images")


if __name__ == "__main__":
    main()