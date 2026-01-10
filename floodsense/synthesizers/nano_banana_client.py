"""
Synthetic data generator using Google Gemini API.

Generates flood-related images using generative AI models.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional
from io import BytesIO

import google.generativeai as genai
from loguru import logger
from PIL import Image
from tqdm import tqdm

from floodsense.utils.config import SynthesizerConfig


class NanoBananaClient:
    """
    Client for generating synthetic flood images using Google Gemini API.
    """

    PROMPT_SUFFIX = ", photorealistic, 1080p, high quality, detailed"

    def __init__(
        self,
        config: Optional[SynthesizerConfig] = None,
    ) -> None:
        """
        Initialize NanoBananaClient.

        Args:
            config: Synthesizer configuration.
        """
        self.config = config or SynthesizerConfig()

        if not self.config.api_key:
            raise ValueError(
                "API key is required. Set FLOODSENSE_API_KEY environment variable "
                "or provide in config."
            )

        # Initialize Gemini
        genai.configure(api_key=self.config.api_key)
        self.model = genai.GenerativeModel(self.config.model)

        logger.info(f"Initialized NanoBananaClient with model: {self.config.model}")

    def generate_image(
        self,
        prompt: str,
        enhance_prompt: bool = True,
    ) -> Optional[Image.Image]:
        """
        Generate a single image from prompt.

        Args:
            prompt: Text prompt for image generation.
            enhance_prompt: Whether to add quality suffixes to prompt.

        Returns:
            Generated PIL Image or None if failed.
        """
        # Enhance prompt
        if enhance_prompt:
            prompt = prompt + self.PROMPT_SUFFIX

        # Retry logic
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Generating image (attempt {attempt + 1}): {prompt}")

                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="image/png",
                    ),
                )

                # Extract image from response
                if response.parts and hasattr(response.parts[0], "inline_data"):
                    image_data = response.parts[0].inline_data.data
                    image = Image.open(BytesIO(image_data))
                    logger.debug("Image generated successfully")
                    return image
                else:
                    logger.warning(f"No image data in response: {response}")

            except Exception as e:
                logger.warning(
                    f"Generation failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )

                if attempt < self.config.max_retries - 1:
                    # Exponential backoff with jitter
                    delay = self.config.retry_delay * (2**attempt)
                    delay += time.uniform(0, 1)
                    time.sleep(delay)

        logger.error(f"All retry attempts failed for prompt: {prompt}")
        return None

    def generate_batch(
        self,
        prompts: List[str],
        output_dir: Path,
        enhance_prompt: bool = True,
        prefix: str = "synthetic",
    ) -> List[Path]:
        """
        Generate images from multiple prompts.

        Args:
            prompts: List of text prompts.
            output_dir: Directory to save generated images.
            enhance_prompt: Whether to add quality suffixes to prompts.
            prefix: Filename prefix.

        Returns:
            List of paths to generated images.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_paths: List[Path] = []

        with tqdm(
            total=len(prompts),
            desc="Generating images",
            unit="img",
        ) as pbar:
            for idx, prompt in enumerate(prompts):
                try:
                    image = self.generate_image(prompt, enhance_prompt)

                    if image:
                        # Save image
                        filename = f"{prefix}_{idx:05d}.png"
                        filepath = output_dir / filename
                        image.save(filepath, quality=95)
                        generated_paths.append(filepath)
                        logger.info(f"Saved: {filepath}")
                    else:
                        logger.warning(f"Failed to generate image for prompt {idx}")

                except Exception as e:
                    logger.error(f"Error processing prompt {idx}: {e}")

                pbar.update(1)

        logger.info(f"Generated {len(generated_paths)}/{len(prompts)} images")
        return generated_paths

    @staticmethod
    def load_prompts_from_file(filepath: Path) -> List[str]:
        """
        Load prompts from JSON file.

        Args:
            filepath: Path to JSON file containing prompts.

        Returns:
            List of prompts.
        """
        import json

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "prompts" in data:
            return data["prompts"]
        else:
            raise ValueError("Invalid prompts file format. Expected list or dict with 'prompts' key")

    def generate_from_file(
        self,
        prompts_file: Path,
        output_dir: Path,
        enhance_prompt: bool = True,
    ) -> List[Path]:
        """
        Generate images from prompts file.

        Args:
            prompts_file: Path to JSON file with prompts.
            output_dir: Directory to save generated images.
            enhance_prompt: Whether to add quality suffixes.

        Returns:
            List of paths to generated images.
        """
        prompts = self.load_prompts_from_file(prompts_file)
        logger.info(f"Loaded {len(prompts)} prompts from {prompts_file}")

        return self.generate_batch(prompts, output_dir, enhance_prompt)