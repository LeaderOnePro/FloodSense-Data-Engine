"""
Synthetic data generator using Google Gemini API.

Generates flood-related images using generative AI models.
"""

import json
import random
import time
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
from loguru import logger
from PIL import Image
from tqdm import tqdm

from floodsense.utils.config import SynthesizerConfig


class ImageGenClient:
    """
    Client for generating synthetic flood images using Google Gemini API.
    """

    PROMPT_SUFFIX = ", photorealistic, 1080p, high quality, detailed"

    def __init__(
        self,
        config: Optional[SynthesizerConfig] = None,
    ) -> None:
        """
        Initialize ImageGenClient.

        Args:
            config: Synthesizer configuration.
        """
        self.config = config or SynthesizerConfig()

        if not self.config.api_key:
            raise ValueError(
                "API key is required. Set FLOODSENSE_API_KEY environment variable "
                "or provide in config."
            )

        self.client = genai.Client(api_key=self.config.api_key)

        logger.info(f"Initialized ImageGenClient with model: {self.config.model}")

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
        if enhance_prompt:
            prompt = prompt + self.PROMPT_SUFFIX

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Generating image (attempt {attempt + 1}): {prompt}")

                response = self.client.models.generate_content(
                    model=self.config.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                    ),
                )

                # Extract image from response
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        image = part.as_image()
                        logger.debug("Image generated successfully")
                        return image

                logger.warning("No image data in response")

            except (APIError, ClientError, ServerError, ValueError, OSError) as e:
                logger.warning(
                    f"Generation failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2**attempt)
                    delay += random.uniform(0, 1)
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
                        filename = f"{prefix}_{idx:05d}.png"
                        filepath = output_dir / filename
                        image.save(filepath, quality=95)
                        generated_paths.append(filepath)
                        logger.info(f"Saved: {filepath}")
                    else:
                        logger.warning(f"Failed to generate image for prompt {idx}")

                except (APIError, ClientError, ServerError, ValueError, OSError) as e:
                    logger.exception(f"Error processing prompt {idx}: {e}")

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