"""
Image validator using CLIP model and heuristic rules.

Combines fast heuristic filtering with accurate CLIP-based content verification.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
from loguru import logger

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    logger.warning("transformers or torch not installed. CLIP validation will be disabled.")
    CLIP_AVAILABLE = False


class ImageValidator:
    """
    Image validator using CLIP model and heuristic rules.

    Two-layer filtering:
    1. Heuristic rules (fast, ~1ms/image)
    2. CLIP similarity (accurate, ~100ms/image)
    """

    # Heuristic thresholds
    MIN_WIDTH = 200
    MIN_HEIGHT = 200
    MAX_ASPECT_RATIO = 4.0
    MIN_BLUE_RATIO = 0.05  # Minimum blue color ratio (water indicator)
    MAX_BLUE_RATIO = 0.95  # Maximum blue ratio (avoid solid blue images)
    MIN_EDGE_DENSITY = 0.01  # Minimum edge density

    # CLIP settings
    DEFAULT_CLIP_THRESHOLD = 0.25
    DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
        clip_model_name: str = DEFAULT_CLIP_MODEL,
        enable_heuristic: bool = True,
        enable_clip: bool = True,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize ImageValidator.

        Args:
            clip_threshold: CLIP similarity threshold (0-1).
            clip_model_name: CLIP model name.
            enable_heuristic: Enable heuristic filtering.
            enable_clip: Enable CLIP filtering.
            device: Device to use (cuda, cpu, or auto).
        """
        self.clip_threshold = clip_threshold
        self.enable_heuristic = enable_heuristic
        self.enable_clip = enable_clip and CLIP_AVAILABLE

        # Device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize CLIP model
        self.clip_model: Optional[CLIPModel] = None
        self.clip_processor: Optional[CLIPProcessor] = None

        if self.enable_clip:
            self._load_clip_model(clip_model_name)

        logger.info(
            f"ImageValidator initialized: heuristic={enable_heuristic}, "
            f"clip={self.enable_clip}, device={self.device}"
        )

    def _load_clip_model(self, model_name: str) -> None:
        """
        Load CLIP model.

        Args:
            model_name: CLIP model name.
        """
        try:
            logger.info(f"Loading CLIP model: {model_name}")
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.enable_clip = False

    def validate(
        self,
        image_path: Path,
        keywords: List[str],
        return_scores: bool = False,
    ) -> bool | Tuple[bool, dict]:
        """
        Validate image against keywords.

        Args:
            image_path: Path to image.
            keywords: List of keywords to match.
            return_scores: Whether to return validation scores.

        Returns:
            True if image is valid, False otherwise.
            If return_scores=True, returns (is_valid, scores_dict).
        """
        scores = {}

        # First layer: Heuristic filtering
        if self.enable_heuristic:
            heuristic_valid, heuristic_scores = self._heuristic_check(image_path)
            scores.update(heuristic_scores)

            if not heuristic_valid:
                logger.debug(f"Image failed heuristic check: {image_path}")
                if return_scores:
                    return False, scores
                return False

        # Second layer: CLIP filtering
        if self.enable_clip:
            clip_valid, clip_scores = self._clip_check(image_path, keywords)
            scores.update(clip_scores)

            if not clip_valid:
                logger.debug(f"Image failed CLIP check: {image_path}")
                if return_scores:
                    return False, scores
                return False

        if return_scores:
            return True, scores
        return True

    def _heuristic_check(self, image_path: Path) -> Tuple[bool, dict]:
        """
        Fast heuristic filtering.

        Checks:
        - Image dimensions
        - Aspect ratio
        - Color distribution (blue ratio for water detection)
        - Edge density (texture complexity)

        Args:
            image_path: Path to image.

        Returns:
            Tuple of (is_valid, scores_dict).
        """
        scores = {}

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Failed to read image: {image_path}")
                return False, scores

            height, width = img.shape[:2]

            # Check dimensions
            scores["width"] = width
            scores["height"] = height

            if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
                logger.debug(f"Image too small: {width}x{height}")
                return False, scores

            # Check aspect ratio
            aspect_ratio = max(width, height) / min(width, height)
            scores["aspect_ratio"] = aspect_ratio

            if aspect_ratio > self.MAX_ASPECT_RATIO:
                logger.debug(f"Aspect ratio too extreme: {aspect_ratio}")
                return False, scores

            # Color analysis - blue ratio (water indicator)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
            blue_ratio = np.sum(blue_mask > 0) / (width * height)
            scores["blue_ratio"] = blue_ratio

            if blue_ratio < self.MIN_BLUE_RATIO:
                logger.debug(f"Blue ratio too low: {blue_ratio}")
                return False, scores

            if blue_ratio > self.MAX_BLUE_RATIO:
                logger.debug(f"Blue ratio too high (solid image): {blue_ratio}")
                return False, scores

            # Edge density (texture complexity)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (width * height)
            scores["edge_density"] = edge_density

            if edge_density < self.MIN_EDGE_DENSITY:
                logger.debug(f"Edge density too low: {edge_density}")
                return False, scores

            return True, scores

        except Exception as e:
            logger.error(f"Error in heuristic check: {e}")
            return False, scores

    def _clip_check(self, image_path: Path, keywords: List[str]) -> Tuple[bool, dict]:
        """
        CLIP-based content verification.

        Args:
            image_path: Path to image.
            keywords: List of keywords to match.

        Returns:
            Tuple of (is_valid, scores_dict).
        """
        scores = {}

        if self.clip_model is None or self.clip_processor is None:
            logger.warning("CLIP model not available")
            return True, scores

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Prepare inputs
            text_inputs = keywords + ["flood", "disaster", "water damage", "emergency"]
            inputs = self.clip_processor(
                text=text_inputs,
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            # Compute similarity
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)

            # Get max probability for keyword matches
            max_prob = probs[0, :len(keywords)].max().item()
            scores["clip_max_prob"] = max_prob

            # Get average probability
            avg_prob = probs[0, :len(keywords)].mean().item()
            scores["clip_avg_prob"] = avg_prob

            # Check threshold
            is_valid = max_prob >= self.clip_threshold

            logger.debug(
                f"CLIP check: max_prob={max_prob:.3f}, avg_prob={avg_prob:.3f}, "
                f"threshold={self.clip_threshold}"
            )

            return is_valid, scores

        except Exception as e:
            logger.error(f"Error in CLIP check: {e}")
            return False, scores

    def validate_batch(
        self,
        image_paths: List[Path],
        keywords: List[str],
        batch_size: int = 32,
    ) -> List[bool]:
        """
        Validate multiple images efficiently using batching.

        Args:
            image_paths: List of image paths.
            keywords: List of keywords to match.
            batch_size: Batch size for CLIP inference.

        Returns:
            List of validation results.
        """
        results = []

        # First pass: heuristic filtering
        valid_paths = []
        for img_path in image_paths:
            if self.enable_heuristic:
                is_valid, _ = self._heuristic_check(img_path)
                if is_valid:
                    valid_paths.append(img_path)
                else:
                    results.append(False)
            else:
                valid_paths.append(img_path)

        # Second pass: CLIP filtering (batch processing)
        if self.enable_clip and valid_paths:
            clip_results = self._clip_check_batch(valid_paths, keywords, batch_size)

            # Merge results
            valid_set = set(valid_paths)
            clip_idx = 0
            for img_path in image_paths:
                if img_path in valid_set:
                    results.append(clip_results[clip_idx])
                    clip_idx += 1
                else:
                    results.append(False)

        return results

    def _clip_check_batch(
        self,
        image_paths: List[Path],
        keywords: List[str],
        batch_size: int,
    ) -> List[bool]:
        """
        Batch CLIP validation for efficiency.

        Args:
            image_paths: List of image paths.
            keywords: List of keywords to match.
            batch_size: Batch size.

        Returns:
            List of validation results.
        """
        if self.clip_model is None or self.clip_processor is None:
            return [True] * len(image_paths)

        results = []

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            try:
                # Load images
                images = [
                    Image.open(p).convert("RGB")
                    for p in batch_paths
                ]

                # Prepare inputs
                text_inputs = keywords + ["flood", "disaster", "water damage", "emergency"]
                inputs = self.clip_processor(
                    text=text_inputs,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                # Compute similarity
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=-1)

                # Check threshold
                batch_results = [
                    probs[j, :len(keywords)].max().item() >= self.clip_threshold
                    for j in range(len(batch_paths))
                ]
                results.extend(batch_results)

            except Exception as e:
                logger.error(f"Error in batch CLIP check: {e}")
                results.extend([False] * len(batch_paths))

        return results