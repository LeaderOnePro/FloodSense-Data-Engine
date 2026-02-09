"""
File utility functions.

Common operations for file handling, checkpointing, and progress tracking.
"""

import hashlib
import json
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from loguru import logger


class FileUtils:
    """Utility class for common file operations."""

    @staticmethod
    def get_file_hash(filepath: Path, algorithm: str = "md5") -> str:
        """
        Calculate hash of a file.

        Args:
            filepath: Path to file.
            algorithm: Hash algorithm (md5, sha256, etc.).

        Returns:
            Hex digest of file hash.
        """
        hash_func = hashlib.new(algorithm)
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    @staticmethod
    def ensure_dir(path: Path) -> Path:
        """
        Ensure directory exists, creating if necessary.

        Args:
            path: Directory path.

        Returns:
            The same path after ensuring existence.
        """
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def iter_images(
        directory: Path,
        extensions: Optional[Set[str]] = None,
        recursive: bool = True,
    ) -> Iterator[Path]:
        """
        Iterate over image files in a directory.

        Args:
            directory: Directory to search.
            extensions: Set of valid extensions (with dot).
            recursive: Whether to search recursively.

        Yields:
            Path to each image file.
        """
        if extensions is None:
            extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

        pattern = "**/*" if recursive else "*"
        for path in directory.glob(pattern):
            if path.is_file() and path.suffix.lower() in extensions:
                yield path

    @staticmethod
    def iter_videos(
        directory: Path,
        extensions: Optional[Set[str]] = None,
        recursive: bool = True,
    ) -> Iterator[Path]:
        """
        Iterate over video files in a directory.

        Args:
            directory: Directory to search.
            extensions: Set of valid extensions (with dot).
            recursive: Whether to search recursively.

        Yields:
            Path to each video file.
        """
        if extensions is None:
            extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}

        pattern = "**/*" if recursive else "*"
        for path in directory.glob(pattern):
            if path.is_file() and path.suffix.lower() in extensions:
                yield path

    @staticmethod
    def sanitize_keyword(keyword: str) -> str:
        """
        Sanitize keyword for use as directory name.

        Removes characters invalid in file paths and replaces spaces
        with underscores.

        Args:
            keyword: Raw keyword string.

        Returns:
            Sanitized keyword safe for use as a directory name.
        """
        sanitized = re.sub(r'[<>:"/\\|?*]', "", keyword)
        sanitized = sanitized.strip().replace(" ", "_")
        return sanitized


class CheckpointManager:
    """
    Manages checkpoints for resumable operations.

    Supports saving/loading progress for long-running tasks.
    """

    def __init__(self, checkpoint_dir: Path, task_name: str) -> None:
        """
        Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory for checkpoint files.
            task_name: Name of the task (used for filename).
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.task_name = task_name
        self.checkpoint_file = self.checkpoint_dir / f"{task_name}.checkpoint"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, data: Dict[str, Any]) -> None:
        """
        Save checkpoint data.

        Args:
            data: Dictionary of checkpoint data.
        """
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "task_name": self.task_name,
            "data": data,
        }
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(checkpoint, f)
        logger.debug(f"Checkpoint saved: {self.checkpoint_file}")

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint data if exists.

        Returns:
            Checkpoint data dictionary or None.
        """
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)
            logger.info(
                f"Checkpoint loaded from {checkpoint['timestamp']}: {self.checkpoint_file}"
            )
            return checkpoint["data"]
        except (pickle.PickleError, KeyError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def clear(self) -> None:
        """Remove checkpoint file."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info(f"Checkpoint cleared: {self.checkpoint_file}")

    def get_completed_items(self) -> Set[str]:
        """
        Get set of completed item IDs from checkpoint.

        Returns:
            Set of completed item identifiers.
        """
        data = self.load()
        if data and "completed" in data:
            return set(data["completed"])
        return set()

    def add_completed(self, item_id: str) -> None:
        """
        Add an item to completed set.

        Args:
            item_id: Identifier of completed item.
        """
        data = self.load() or {"completed": []}
        if "completed" not in data:
            data["completed"] = []
        if item_id not in data["completed"]:
            data["completed"].append(item_id)
        self.save(data)


class ProgressTracker:
    """
    Tracks progress statistics for data processing tasks.
    """

    def __init__(self) -> None:
        """Initialize ProgressTracker."""
        self.stats: Dict[str, int] = {
            "total": 0,
            "processed": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
        }
        self.errors: List[Dict[str, str]] = []
        self.start_time: Optional[datetime] = None

    def start(self, total: int) -> None:
        """
        Start tracking with total count.

        Args:
            total: Total number of items to process.
        """
        self.stats["total"] = total
        self.start_time = datetime.now()

    def record_success(self) -> None:
        """Record a successful operation."""
        self.stats["processed"] += 1
        self.stats["success"] += 1

    def record_failure(self, item_id: str, error: str) -> None:
        """
        Record a failed operation.

        Args:
            item_id: Identifier of failed item.
            error: Error message.
        """
        self.stats["processed"] += 1
        self.stats["failed"] += 1
        self.errors.append({"item": item_id, "error": error})

    def record_skip(self) -> None:
        """Record a skipped item."""
        self.stats["processed"] += 1
        self.stats["skipped"] += 1

    def get_progress(self) -> float:
        """
        Get current progress percentage.

        Returns:
            Progress as percentage (0-100).
        """
        if self.stats["total"] == 0:
            return 0.0
        return (self.stats["processed"] / self.stats["total"]) * 100

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of progress.

        Returns:
            Dictionary with progress statistics.
        """
        elapsed = None
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()

        return {
            **self.stats,
            "progress_pct": self.get_progress(),
            "elapsed_seconds": elapsed,
            "error_count": len(self.errors),
        }

    def save_report(self, filepath: Path) -> None:
        """
        Save detailed progress report to JSON file.

        Args:
            filepath: Path to save report.
        """
        report = {
            "summary": self.get_summary(),
            "errors": self.errors,
            "generated_at": datetime.now().isoformat(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Progress report saved to {filepath}")
