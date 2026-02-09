"""
Image validator using CLIP model and heuristic rules.

Combines fast heuristic filtering with accurate CLIP-based content verification.
Uses a strategy pattern with pluggable BaseValidator implementations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
from loguru import logger

from floodsense.validators.base_validator import BaseValidator

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    logger.warning("transformers or torch not installed. CLIP validation will be disabled.")
    CLIP_AVAILABLE = False

_TORCH_ERRORS: tuple = (RuntimeError, OSError, ValueError)
if CLIP_AVAILABLE and hasattr(torch.cuda, "OutOfMemoryError"):
    _TORCH_ERRORS = (*_TORCH_ERRORS, torch.cuda.OutOfMemoryError)


class HeuristicValidator(BaseValidator):
    """Fast heuristic filtering based on image properties."""

    MIN_WIDTH = 200
    MIN_HEIGHT = 200
    MAX_ASPECT_RATIO = 4.0
    MIN_BLUE_RATIO = 0.05
    MAX_BLUE_RATIO = 0.95
    MIN_EDGE_DENSITY = 0.01

    def check(self, image_path: Path, keywords: List[str]) -> Tuple[bool, dict]:
        """
        Validate image using heuristic rules.

        Checks dimensions, aspect ratio, blue color ratio, and edge density.

        Args:
            image_path: Path to image.
            keywords: List of keywords (unused by heuristic).

        Returns:
            Tuple of (is_valid, scores_dict).
        """
        scores: dict = {}

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Failed to read image: {image_path}")
                return False, scores

            height, width = img.shape[:2]

            scores["width"] = width
            scores["height"] = height

            if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
                logger.debug(f"Image too small: {width}x{height}")
                return False, scores

            aspect_ratio = max(width, height) / min(width, height)
            scores["aspect_ratio"] = aspect_ratio

            if aspect_ratio > self.MAX_ASPECT_RATIO:
                logger.debug(f"Aspect ratio too extreme: {aspect_ratio}")
                return False, scores

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

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (width * height)
            scores["edge_density"] = edge_density

            if edge_density < self.MIN_EDGE_DENSITY:
                logger.debug(f"Edge density too low: {edge_density}")
                return False, scores

            return True, scores

        except (cv2.error, OSError) as e:
            logger.exception(f"Error in heuristic check: {e}")
            return False, scores


class CLIPValidator(BaseValidator):
    """CLIP-based content verification."""

    DEFAULT_THRESHOLD = 0.25
    DEFAULT_MODEL = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize CLIPValidator.

        Args:
            threshold: CLIP similarity threshold (0-1).
            model_name: CLIP model name.
            device: Device to use (cuda, cpu, or auto).
            batch_size: Batch size for CLIP inference.
        """
        self.threshold = threshold
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.clip_model: Optional[CLIPModel] = None
        self.clip_processor: Optional[CLIPProcessor] = None
        self._load_model(model_name)

    def _load_model(self, model_name: str) -> None:
        """Load CLIP model."""
        try:
            logger.info(f"Loading CLIP model: {model_name}")
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            logger.info("CLIP model loaded successfully")
        except _TORCH_ERRORS as e:
            logger.exception(f"Failed to load CLIP model: {e}")

    def check(self, image_path: Path, keywords: List[str]) -> Tuple[bool, dict]:
        """
        Validate image using CLIP similarity.

        Args:
            image_path: Path to image.
            keywords: List of keywords to match.

        Returns:
            Tuple of (is_valid, scores_dict).
        """
        scores: dict = {}

        if self.clip_model is None or self.clip_processor is None:
            logger.warning("CLIP model not available")
            return True, scores

        try:
            image = Image.open(image_path).convert("RGB")

            text_inputs = keywords + ["flood", "disaster", "water damage", "emergency"]
            inputs = self.clip_processor(
                text=text_inputs,
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)

            max_prob = probs[0, :len(keywords)].max().item()
            scores["clip_max_prob"] = max_prob

            avg_prob = probs[0, :len(keywords)].mean().item()
            scores["clip_avg_prob"] = avg_prob

            is_valid = max_prob >= self.threshold

            logger.debug(
                f"CLIP check: max_prob={max_prob:.3f}, avg_prob={avg_prob:.3f}, "
                f"threshold={self.threshold}"
            )

            return is_valid, scores

        except _TORCH_ERRORS as e:
            logger.exception(f"Error in CLIP check: {e}")
            return False, scores

    def check_batch(
        self,
        image_paths: List[Path],
        keywords: List[str],
    ) -> List[bool]:
        """
        Batch CLIP validation for efficiency.

        Args:
            image_paths: List of image paths.
            keywords: List of keywords to match.

        Returns:
            List of validation results.
        """
        if self.clip_model is None or self.clip_processor is None:
            return [True] * len(image_paths)

        results = []

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]

            try:
                images = [
                    Image.open(p).convert("RGB")
                    for p in batch_paths
                ]

                text_inputs = keywords + ["flood", "disaster", "water damage", "emergency"]
                inputs = self.clip_processor(
                    text=text_inputs,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=-1)

                batch_results = [
                    probs[j, :len(keywords)].max().item() >= self.threshold
                    for j in range(len(batch_paths))
                ]
                results.extend(batch_results)

            except _TORCH_ERRORS as e:
                logger.exception(f"Error in batch CLIP check: {e}")
                results.extend([False] * len(batch_paths))

        return results


class ImageValidator:
    """
    Composite image validator using pluggable strategies.

    Delegates validation to a list of BaseValidator implementations,
    short-circuiting on the first failure.
    """

    def __init__(
        self,
        clip_threshold: float = CLIPValidator.DEFAULT_THRESHOLD,
        clip_model_name: str = CLIPValidator.DEFAULT_MODEL,
        enable_heuristic: bool = True,
        enable_clip: bool = True,
        device: Optional[str] = None,
        validators: Optional[List[BaseValidator]] = None,
    ) -> None:
        """
        Initialize ImageValidator.

        Args:
            clip_threshold: CLIP similarity threshold (0-1).
            clip_model_name: CLIP model name.
            enable_heuristic: Enable heuristic filtering.
            enable_clip: Enable CLIP filtering.
            device: Device to use (cuda, cpu, or auto).
            validators: Optional explicit list of validators (overrides flags).
        """
        if validators is not None:
            self.validators = validators
        else:
            self.validators: List[BaseValidator] = []
            if enable_heuristic:
                self.validators.append(HeuristicValidator())
            if enable_clip and CLIP_AVAILABLE:
                self.validators.append(
                    CLIPValidator(
                        threshold=clip_threshold,
                        model_name=clip_model_name,
                        device=device,
                    )
                )

        logger.info(
            f"ImageValidator initialized with {len(self.validators)} strategies: "
            f"{[type(v).__name__ for v in self.validators]}"
        )

    def validate(
        self,
        image_path: Path,
        keywords: List[str],
        return_scores: bool = False,
    ) -> bool | Tuple[bool, dict]:
        """
        Validate image against keywords using all strategies.

        Args:
            image_path: Path to image.
            keywords: List of keywords to match.
            return_scores: Whether to return validation scores.

        Returns:
            True if image is valid, False otherwise.
            If return_scores=True, returns (is_valid, scores_dict).
        """
        all_scores: dict = {}

        for validator in self.validators:
            is_valid, scores = validator.check(image_path, keywords)
            all_scores.update(scores)

            if not is_valid:
                logger.debug(
                    f"Image failed {type(validator).__name__} check: {image_path}"
                )
                if return_scores:
                    return False, all_scores
                return False

        if return_scores:
            return True, all_scores
        return True

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
        results: List[Optional[bool]] = [None] * len(image_paths)
        remaining_indices = list(range(len(image_paths)))

        for validator in self.validators:
            if not remaining_indices:
                break

            # Use batch check for CLIPValidator
            if isinstance(validator, CLIPValidator):
                remaining_paths = [image_paths[i] for i in remaining_indices]
                batch_results = validator.check_batch(remaining_paths, keywords)
                still_remaining = []
                for idx, is_valid in zip(remaining_indices, batch_results):
                    if not is_valid:
                        results[idx] = False
                    else:
                        still_remaining.append(idx)
                remaining_indices = still_remaining
            else:
                still_remaining = []
                for idx in remaining_indices:
                    is_valid, _ = validator.check(image_paths[idx], keywords)
                    if not is_valid:
                        results[idx] = False
                    else:
                        still_remaining.append(idx)
                remaining_indices = still_remaining

        # Mark remaining (passed all validators) as valid
        for idx in remaining_indices:
            results[idx] = True

        return results
