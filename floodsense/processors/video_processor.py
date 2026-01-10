"""
Smart video frame extractor with scene detection and quality control.

Uses scene detection to extract keyframes and blur detection for quality control.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger
from tqdm import tqdm

from floodsense.utils.config import ProcessorConfig


class VideoProcessor:
    """
    Smart video frame extractor with scene detection and quality control.
    """

    TARGET_RESOLUTION = (1920, 1080)  # 1080p

    def __init__(self, config: Optional[ProcessorConfig] = None) -> None:
        """
        Initialize VideoProcessor.

        Args:
            config: Processor configuration.
        """
        self.config = config or ProcessorConfig()

    def extract_keyframes(
        self,
        video_path: Path,
        output_dir: Path,
        scene_threshold: Optional[float] = None,
        blur_threshold: Optional[float] = None,
    ) -> List[Path]:
        """
        Extract keyframes from video using scene detection.

        Args:
            video_path: Path to input video file.
            output_dir: Directory to save extracted frames.
            scene_threshold: Scene change detection threshold (0-100).
            blur_threshold: Laplacian variance threshold for blur detection.

        Returns:
            List of paths to extracted keyframes.
        """
        scene_threshold = scene_threshold or self.config.scene_threshold
        blur_threshold = blur_threshold or self.config.blur_threshold

        logger.info(f"Processing video: {video_path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []

        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video info: {total_frames} frames, {fps:.2f} fps")

        # Scene detection
        keyframe_indices = self._detect_scene_changes(
            cap, scene_threshold, total_frames
        )
        logger.info(f"Detected {len(keyframe_indices)} keyframes")

        # Extract keyframes
        extracted_paths: List[Path] = []
        video_name = video_path.stem

        with tqdm(
            total=len(keyframe_indices),
            desc=f"Extracting keyframes: {video_name}",
            unit="frame",
        ) as pbar:
            for idx, frame_idx in enumerate(keyframe_indices):
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"Failed to read frame {frame_idx}")
                    continue

                # Check blur
                if not self._check_blur(frame, blur_threshold):
                    logger.debug(f"Skipping blurry frame {frame_idx}")
                    continue

                # Resize to target resolution
                frame = self._resize_frame(frame, self.TARGET_RESOLUTION)

                # Save frame
                frame_path = output_dir / f"{video_name}_frame_{idx:05d}.jpg"
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted_paths.append(frame_path)

                pbar.update(1)

        cap.release()
        logger.info(f"Extracted {len(extracted_paths)} keyframes to {output_dir}")

        return extracted_paths

    def _detect_scene_changes(
        self,
        cap: cv2.VideoCapture,
        threshold: float,
        total_frames: int,
    ) -> List[int]:
        """
        Detect scene changes using frame difference.

        Args:
            cap: OpenCV VideoCapture object.
            threshold: Scene change threshold (0-100).
            total_frames: Total number of frames.

        Returns:
            List of keyframe indices.
        """
        keyframe_indices: List[int] = []
        prev_frame = None
        prev_hist = None

        # Always include first frame
        keyframe_indices.append(0)

        # Process frames in chunks to save memory
        chunk_size = 100
        for start_idx in range(0, total_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, total_frames)

            for frame_idx in range(start_idx, end_idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                # Calculate histogram difference
                hist = self._calculate_histogram(frame)

                if prev_hist is not None:
                    diff = self._calculate_histogram_diff(prev_hist, hist)

                    # Scene change detected
                    if diff > threshold:
                        keyframe_indices.append(frame_idx)

                prev_frame = frame
                prev_hist = hist

        return keyframe_indices

    @staticmethod
    def _calculate_histogram(frame: np.ndarray) -> np.ndarray:
        """
        Calculate color histogram for frame.

        Args:
            frame: Input frame (BGR).

        Returns:
            Flattened histogram array.
        """
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])

        # Normalize
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_r = cv2.normalize(hist_r, hist_r).flatten()

        return np.concatenate([hist_b, hist_g, hist_r])

    @staticmethod
    def _calculate_histogram_diff(hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Calculate histogram difference using correlation.

        Args:
            hist1: First histogram.
            hist2: Second histogram.

        Returns:
            Difference score (0-100).
        """
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        # Convert correlation to difference (0-100)
        diff = (1 - correlation) * 100
        return diff

    @staticmethod
    def _check_blur(frame: np.ndarray, threshold: float) -> bool:
        """
        Check if frame is blurry using Laplacian variance.

        Args:
            frame: Input frame.
            threshold: Blur threshold.

        Returns:
            True if frame is not too blurry, False otherwise.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var >= threshold

    @staticmethod
    def _resize_frame(
        frame: np.ndarray,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Resize frame to target resolution.

        Args:
            frame: Input frame.
            target_size: Target (width, height).

        Returns:
            Resized frame.
        """
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)

    def process_video_directory(
        self,
        video_dir: Path,
        output_dir: Path,
        **kwargs,
    ) -> List[Path]:
        """
        Process all videos in a directory.

        Args:
            video_dir: Directory containing videos.
            output_dir: Directory to save extracted frames.
            **kwargs: Additional arguments for extract_keyframes.

        Returns:
            List of paths to all extracted frames.
        """
        from floodsense.utils.file_utils import FileUtils

        all_frames: List[Path] = []

        video_paths = list(FileUtils.iter_videos(video_dir))
        logger.info(f"Found {len(video_paths)} videos in {video_dir}")

        for video_path in tqdm(video_paths, desc="Processing videos"):
            video_output_dir = output_dir / video_path.stem
            frames = self.extract_keyframes(video_path, video_output_dir, **kwargs)
            all_frames.extend(frames)

        return all_frames