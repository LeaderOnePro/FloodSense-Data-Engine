"""Abstract base class for all data processors."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from floodsense.utils.config import ProcessorConfig


class BaseProcessor(ABC):
    """Abstract base class for all data processors."""

    def __init__(self, config: Optional[ProcessorConfig] = None) -> None:
        self.config = config or ProcessorConfig()

    @abstractmethod
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        **kwargs,
    ) -> Tuple[List[Path], Dict[str, Any]]:
        """Process all files in a directory. Returns (processed_paths, stats)."""
        pass
