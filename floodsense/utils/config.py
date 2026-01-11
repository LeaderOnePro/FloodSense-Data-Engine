"""
Configuration management module.

Loads and manages configuration from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class CrawlerConfig(BaseModel):
    """Configuration for web crawlers."""

    max_workers: int = Field(default=4, description="Maximum concurrent workers")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retries on failure")
    retry_delay: float = Field(default=1.0, description="Delay between retries")
    user_agents: list[str] = Field(
        default_factory=lambda: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ]
    )


class ProcessorConfig(BaseModel):
    """Configuration for image/video processors."""

    target_resolution: tuple[int, int] = Field(
        default=(1920, 1080), description="Target resolution (width, height)"
    )
    min_resolution: tuple[int, int] = Field(
        default=(1280, 720), description="Minimum acceptable resolution"
    )
    blur_threshold: float = Field(
        default=100.0, description="Laplacian variance threshold for blur detection"
    )
    scene_threshold: float = Field(
        default=30.0, description="Scene change detection threshold"
    )
    phash_threshold: int = Field(
        default=8, description="pHash hamming distance threshold for deduplication"
    )


class SynthesizerConfig(BaseModel):
    """Configuration for synthetic data generation."""

    api_key: Optional[str] = Field(default=None, description="API key for Gemini/NanoBanana")
    model: str = Field(
        default="gemini-2.0-flash-preview-image-generation",
        description="Model to use for image generation",
    )
    max_retries: int = Field(default=5, description="Maximum API retry attempts")
    retry_delay: float = Field(default=2.0, description="Base delay between retries")
    batch_size: int = Field(default=10, description="Batch size for generation")


class DataConfig(BaseModel):
    """Configuration for data directories."""

    raw_dir: Path = Field(default=Path("data/raw"))
    processed_dir: Path = Field(default=Path("data/processed"))
    synthetic_dir: Path = Field(default=Path("data/synthetic"))
    checkpoint_dir: Path = Field(default=Path("data/.checkpoints"))


class ProxyConfig(BaseModel):
    """Configuration for proxy settings."""

    enabled: bool = Field(default=False)
    http: Optional[str] = Field(default=None)
    https: Optional[str] = Field(default=None)
    rotation_enabled: bool = Field(default=False)
    proxy_list: list[str] = Field(default_factory=list)


class ValidatorConfig(BaseModel):
    """Configuration for image content validation."""

    enabled: bool = Field(default=True, description="Enable content validation")
    enable_heuristic: bool = Field(
        default=True, description="Enable fast heuristic filtering"
    )
    enable_clip: bool = Field(
        default=True, description="Enable CLIP-based validation"
    )
    clip_threshold: float = Field(
        default=0.25, description="CLIP similarity threshold (0-1)"
    )
    clip_model: str = Field(
        default="openai/clip-vit-base-patch32",
        description="CLIP model name",
    )
    device: Optional[str] = Field(
        default=None, description="Device to use (cuda/cpu/auto)"
    )
    batch_size: int = Field(
        default=32, description="Batch size for CLIP inference"
    )


class Config(BaseModel):
    """Main configuration container."""

    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)
    processor: ProcessorConfig = Field(default_factory=ProcessorConfig)
    synthesizer: SynthesizerConfig = Field(default_factory=SynthesizerConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """
        Load configuration from YAML file and environment variables.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            Loaded Config instance.
        """
        config_dict: Dict[str, Any] = {}

        # Load from YAML file if provided
        if config_path and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}

        # Override with environment variables
        env_mappings = {
            "FLOODSENSE_API_KEY": ("synthesizer", "api_key"),
            "FLOODSENSE_MODEL": ("synthesizer", "model"),
            "FLOODSENSE_PROXY_HTTP": ("proxy", "http"),
            "FLOODSENSE_PROXY_HTTPS": ("proxy", "https"),
            "FLOODSENSE_MAX_WORKERS": ("crawler", "max_workers"),
        }

        for env_var, (section, key) in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                if section not in config_dict:
                    config_dict[section] = {}
                # Type conversion for integers
                if key == "max_workers":
                    value = int(value)
                config_dict[section][key] = value

        return cls(**config_dict)

    def save(self, config_path: Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            config_path: Path to save configuration.
        """
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)

    def ensure_directories(self) -> None:
        """Create all data directories if they don't exist."""
        for dir_path in [
            self.data.raw_dir,
            self.data.processed_dir,
            self.data.synthetic_dir,
            self.data.checkpoint_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
