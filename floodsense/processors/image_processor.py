"""
Image processor for quality control and deduplication.

Handles resolution standardization, blur detection, and duplicate removal.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Set, Tuple
from loguru import logger
from PIL import Image
from tqdm import tqdm

from floodsense.utils.config import ProcessorConfig
from floodsense.utils.file_utils import FileUtils


class ImageProcessor:
    """
    Image processor for quality control and deduplication.
    """

    TARGET_RESOLUTION = (1920, 1080)  # 1080p
    MIN_RESOLUTION = (480, 360)       # 360p minimum (relaxed)

    def __init__(self, config: Optional[ProcessorConfig] = None) -> None:
        """
        Initialize ImageProcessor.

        Args:
            config: Processor configuration.
        """
        self.config = config or ProcessorConfig()

    def standardize_resolution(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Standardize image resolution.

        - Downsample images larger than 1080p
        - Remove images smaller than 720p

        Args:
            image_path: Path to input image.
            output_path: Path to save processed image.
                        If None, overwrites original.

        Returns:
            Path to processed image or None if removed.
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size

                # Check minimum resolution
                if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
                    logger.debug(
                        f"Removing {image_path}: resolution {width}x{height} "
                        f"below minimum {self.MIN_RESOLUTION[0]}x{self.MIN_RESOLUTION[1]}"
                    )
                    image_path.unlink()
                    return None

                # Check if resize needed
                if width > self.TARGET_RESOLUTION[0] or height > self.TARGET_RESOLUTION[1]:
                    # Resize using Lanczos resampling
                    img = img.resize(self.TARGET_RESOLUTION, Image.Resampling.LANCZOS)
                    logger.debug(f"Resized {image_path} from {width}x{height} to 1080p")

                # Save
                output_path = output_path or image_path
                img.save(output_path, quality=95, optimize=True)
                return output_path

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            return None

    def check_blur(
        self,
        image_path: Path,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Check if image is blurry using Laplacian variance.

        Args:
            image_path: Path to image.
            threshold: Blur threshold (default from config).

        Returns:
            True if image is not blurry, False otherwise.
        """
        threshold = threshold or self.config.blur_threshold

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            return laplacian_var >= threshold

        except Exception as e:
            logger.error(f"Failed to check blur for {image_path}: {e}")
            return False

    def calculate_phash(self, image_path: Path) -> Optional[str]:
        """
        Calculate perceptual hash for deduplication.

        Args:
            image_path: Path to image.

        Returns:
            Hex string of pHash or None if failed.
        """
        try:
            from imagehash import phash
            with Image.open(image_path) as img:
                return str(phash(img))
        except Exception as e:
            logger.error(f"Failed to calculate pHash for {image_path}: {e}")
            return None

    def deduplicate(
        self,
        image_paths: List[Path],
        threshold: Optional[int] = None,
    ) -> List[Path]:
        """
        Remove duplicate images using perceptual hashing.

        Args:
            image_paths: List of image paths.
            threshold: Hamming distance threshold (default from config).

        Returns:
            List of unique image paths.
        """
        from imagehash import hex_to_hash

        threshold = threshold or self.config.phash_threshold
        logger.info(f"Deduplicating {len(image_paths)} images with threshold {threshold}")

        seen_hashes: Set[str] = set()
        unique_paths: List[Path] = []

        for img_path in tqdm(image_paths, desc="Deduplicating"):
            phash = self.calculate_phash(img_path)
            if phash is None:
                continue

            # Check for similar hashes
            is_duplicate = False
            for seen_hash in seen_hashes:
                distance = hex_to_hash(phash) - hex_to_hash(seen_hash)
                if distance <= threshold:
                    is_duplicate = True
                    logger.debug(f"Duplicate found: {img_path} (distance: {distance})")
                    break

            if not is_duplicate:
                unique_paths.append(img_path)
                seen_hashes.add(phash)

        logger.info(f"Removed {len(image_paths) - len(unique_paths)} duplicates")
        return unique_paths

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        remove_blur: bool = True,
        deduplicate: bool = True,
    ) -> Tuple[List[Path], dict]:
        """
        Process all images in a directory.

        Args:
            input_dir: Directory containing input images.
            output_dir: Directory to save processed images.
            remove_blur: Whether to remove blurry images.
            deduplicate: Whether to remove duplicates.

        Returns:
            Tuple of (processed image paths, statistics).
        """
        stats = {
            "total": 0,
            "processed": 0,
            "removed_low_res": 0,
            "removed_blur": 0,
            "removed_duplicates": 0,
            "final": 0,
        }

        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all images
        image_paths = list(FileUtils.iter_images(input_dir))
        stats["total"] = len(image_paths)
        logger.info(f"Found {stats['total']} images in {input_dir}")

        processed_paths: List[Path] = []

        # Resolution standardization
        with tqdm(image_paths, desc="Standardizing resolution") as pbar:
            for img_path in pbar:
                # Use parent directory name as prefix to avoid filename collisions
                parent_name = img_path.parent.name
                output_path = output_dir / f"{parent_name}_{img_path.name}"
                result = self.standardize_resolution(img_path, output_path)

                if result is None:
                    stats["removed_low_res"] += 1
                else:
                    processed_paths.append(result)
                    stats["processed"] += 1
                pbar.set_postfix({"removed": stats["removed_low_res"]})

        # Blur detection
        if remove_blur:
            filtered_paths: List[Path] = []
            with tqdm(processed_paths, desc="Detecting blur") as pbar:
                for img_path in pbar:
                    if self.check_blur(img_path):
                        filtered_paths.append(img_path)
                    else:
                        img_path.unlink()
                        stats["removed_blur"] += 1
                    pbar.set_postfix({"removed": stats["removed_blur"]})
            processed_paths = filtered_paths

        # Deduplication
        if deduplicate:
            before_count = len(processed_paths)
            processed_paths = self.deduplicate(processed_paths)
            stats["removed_duplicates"] = before_count - len(processed_paths)

        stats["final"] = len(processed_paths)
        logger.info(f"Processing complete: {stats}")

        return processed_paths, stats