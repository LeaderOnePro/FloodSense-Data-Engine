"""
Complete cleaning pipeline for flood imagery data.

Orchestrates all processing steps: resolution standardization,
blur detection, deduplication, and reporting.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from floodsense.processors.image_processor import ImageProcessor
from floodsense.processors.video_processor import VideoProcessor
from floodsense.utils.config import ProcessorConfig
from floodsense.utils.file_utils import FileUtils, ProgressTracker
from floodsense.validators.image_validator import ImageValidator


class CleaningPipeline:
    """
    Complete cleaning pipeline for flood imagery data.
    """

    def __init__(
        self,
        config: Optional[ProcessorConfig] = None,
        validator: Optional[ImageValidator] = None,
    ) -> None:
        """
        Initialize CleaningPipeline.

        Args:
            config: Processor configuration.
            validator: Optional image validator for content filtering.
        """
        self.config = config or ProcessorConfig()
        self.image_processor = ImageProcessor(config)
        self.video_processor = VideoProcessor(config)
        self.validator = validator

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        extract_video_frames: bool = True,
        remove_blur: bool = True,
        deduplicate: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete cleaning pipeline.

        Args:
            input_dir: Directory containing raw data (images and videos).
            output_dir: Directory to save processed data.
            extract_video_frames: Whether to extract frames from videos.
            remove_blur: Whether to remove blurry images.
            deduplicate: Whether to remove duplicates.

        Returns:
            Dictionary with pipeline statistics.
        """
        logger.info("Starting FloodSense cleaning pipeline")
        start_time = datetime.now()

        # Create output directories
        images_output_dir = output_dir / "images"
        videos_output_dir = output_dir / "video_frames"

        stats: Dict[str, Any] = {
            "start_time": start_time.isoformat(),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "images": {},
            "videos": {},
            "total": {},
        }

        # Process images
        image_stats = self._process_images(
            input_dir,
            images_output_dir,
            remove_blur,
            deduplicate,
        )
        stats["images"] = image_stats

        # Process videos
        if extract_video_frames:
            video_stats = self._process_videos(
                input_dir,
                videos_output_dir,
            )
            stats["videos"] = video_stats

        # Combine statistics
        stats["total"] = self._combine_stats(stats)

        # Save report
        end_time = datetime.now()
        stats["end_time"] = end_time.isoformat()
        stats["duration_seconds"] = (end_time - start_time).total_seconds()

        report_path = output_dir / f"cleaning_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        self._save_report(stats, report_path)

        self._print_summary(stats)

        return stats

    def _process_images(
        self,
        input_dir: Path,
        output_dir: Path,
        remove_blur: bool,
        deduplicate: bool,
    ) -> Dict[str, Any]:
        """
        Process all images in input directory.

        Args:
            input_dir: Directory containing images.
            output_dir: Directory to save processed images.
            remove_blur: Whether to remove blurry images.
            deduplicate: Whether to remove duplicates.

        Returns:
            Dictionary with image processing statistics.
        """
        logger.info("Processing images...")

        # Find all images
        image_paths = list(FileUtils.iter_images(input_dir))
        if not image_paths:
            logger.info("No images found in input directory")
            return {"total": 0, "processed": 0, "final": 0}

        # Process images
        processed_paths, stats = self.image_processor.process_directory(
            input_dir,
            output_dir,
            remove_blur=remove_blur,
            deduplicate=deduplicate,
        )

        # Content validation
        if self.validator and processed_paths:
            valid_paths = []
            for img_path in processed_paths:
                if self.validator.validate(img_path, keywords=[]):
                    valid_paths.append(img_path)
                else:
                    img_path.unlink(missing_ok=True)
            stats["removed_validation"] = len(processed_paths) - len(valid_paths)
            stats["final"] = len(valid_paths)

        return stats

    def _process_videos(
        self,
        input_dir: Path,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Process all videos in input directory.

        Args:
            input_dir: Directory containing videos.
            output_dir: Directory to save extracted frames.

        Returns:
            Dictionary with video processing statistics.
        """
        logger.info("Processing videos...")

        # Find all videos
        video_paths = list(FileUtils.iter_videos(input_dir))
        if not video_paths:
            logger.info("No videos found in input directory")
            return {"total": 0, "processed": 0, "final": 0}

        # Extract frames
        extracted_frames, video_stats = self.video_processor.process_directory(
            input_dir,
            output_dir,
        )

        return video_stats

    @staticmethod
    def _combine_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine statistics from different processing steps.

        Args:
            stats: Statistics dictionary.

        Returns:
            Combined statistics.
        """
        combined = {
            "total_input": 0,
            "total_output": 0,
            "total_removed": 0,
        }

        # Add image stats
        if "images" in stats:
            img_stats = stats["images"]
            combined["total_input"] += img_stats.get("total", 0)
            combined["total_output"] += img_stats.get("final", 0)
            combined["total_removed"] += img_stats.get("removed_low_res", 0)
            combined["total_removed"] += img_stats.get("removed_blur", 0)
            combined["total_removed"] += img_stats.get("removed_duplicates", 0)

        # Add video stats
        if "videos" in stats:
            vid_stats = stats["videos"]
            combined["total_input"] += vid_stats.get("total_videos", 0)
            combined["total_output"] += vid_stats.get("total_frames_extracted", 0)

        return combined

    @staticmethod
    def _save_report(stats: Dict[str, Any], report_path: Path) -> None:
        """
        Save cleaning report to JSON file.

        Args:
            stats: Statistics dictionary.
            report_path: Path to save report.
        """
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {report_path}")

    @staticmethod
    def _print_summary(stats: Dict[str, Any]) -> None:
        """
        Print summary of cleaning pipeline.

        Args:
            stats: Statistics dictionary.
        """
        logger.info("=" * 60)
        logger.info("CLEANING PIPELINE SUMMARY")
        logger.info("=" * 60)

        if "images" in stats:
            img = stats["images"]
            logger.info(f"Images:")
            logger.info(f"  Total input: {img.get('total', 0)}")
            logger.info(f"  Removed (low res): {img.get('removed_low_res', 0)}")
            logger.info(f"  Removed (blur): {img.get('removed_blur', 0)}")
            logger.info(f"  Removed (duplicates): {img.get('removed_duplicates', 0)}")
            logger.info(f"  Final: {img.get('final', 0)}")

        if "videos" in stats:
            vid = stats["videos"]
            logger.info(f"Videos:")
            logger.info(f"  Total videos: {vid.get('total_videos', 0)}")
            logger.info(f"  Frames extracted: {vid.get('total_frames_extracted', 0)}")

        if "total" in stats:
            tot = stats["total"]
            logger.info(f"Total:")
            logger.info(f"  Input items: {tot.get('total_input', 0)}")
            logger.info(f"  Output items: {tot.get('total_output', 0)}")
            logger.info(f"  Removed: {tot.get('total_removed', 0)}")

        duration = stats.get("duration_seconds", 0)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info("=" * 60)