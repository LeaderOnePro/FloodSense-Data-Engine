"""Abstract base for pluggable validation strategies."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple


class BaseValidator(ABC):
    """Abstract base for pluggable validation strategies."""

    @abstractmethod
    def check(self, image_path: Path, keywords: List[str]) -> Tuple[bool, dict]:
        """Validate a single image. Returns (is_valid, scores)."""
        pass
