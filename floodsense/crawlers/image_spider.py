"""
Generic image spider for scraping images from multiple sources.

Supports multi-threaded downloading with resume capability.
"""

import concurrent.futures
import re
from pathlib import Path
from typing import List, Optional, Set

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm

from floodsense.crawlers.base import BaseCrawler
from floodsense.utils.file_utils import CheckpointManager
from floodsense.validators.image_validator import ImageValidator


class ImageSpider(BaseCrawler):
    """
    Generic image spider for scraping images from web.

    Supports multiple image sources and concurrent downloads.
    """

    # Image URL patterns for different sources
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

    def __init__(self, *args, validator: Optional[ImageValidator] = None, **kwargs) -> None:
        """Initialize ImageSpider."""
        super().__init__(*args, **kwargs)
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.validator = validator

    def crawl(
        self,
        keywords: List[str],
        max_results: int = 100,
        enable_resume: bool = True,
    ) -> List[Path]:
        """
        Crawl images for given keywords.

        Args:
            keywords: List of search keywords.
            max_results: Maximum images to download per keyword.
            enable_resume: Whether to enable resume capability.

        Returns:
            List of paths to downloaded images.
        """
        if enable_resume:
            self.checkpoint_manager = CheckpointManager(
                self.config.data.checkpoint_dir, task_name="image_spider"
            )
        downloaded_paths: List[Path] = []

        for keyword in keywords:
            logger.info(f"Starting crawl for keyword: {keyword}")
            urls = self.get_image_urls(keyword, max_results=max_results)
            logger.info(f"Found {len(urls)} image URLs for '{keyword}'")

            keyword_dir = self.output_dir / self._sanitize_keyword(keyword)
            keyword_dir.mkdir(parents=True, exist_ok=True)

            paths = self._download_images(urls, keyword_dir, keyword, keywords)
            downloaded_paths.extend(paths)

        return downloaded_paths

    def get_image_urls(self, keyword: str, max_results: int = 100) -> List[str]:
        """
        Get image URLs using Google Images search.

        Args:
            keyword: Search keyword.
            max_results: Maximum number of URLs to return.

        Returns:
            List of image URLs.
        """
        urls: List[str] = []
        page = 0
        results_per_page = 100

        while len(urls) < max_results:
            search_url = self._build_search_url(keyword, page * results_per_page)
            response = self.request_with_retry(search_url)

            if response is None:
                logger.warning(f"Failed to fetch search results for page {page}")
                break

            page_urls = self._extract_image_urls(response.text)
            if not page_urls:
                logger.info(f"No more images found for '{keyword}' on page {page}")
                break

            urls.extend(page_urls)
            page += 1

            # Rate limiting
            if len(urls) < max_results:
                import time
                time.sleep(1)

        return urls[:max_results]

    def _build_search_url(self, keyword: str, start: int = 0) -> str:
        """
        Build Google Images search URL.

        Args:
            keyword: Search keyword.
            start: Starting index for pagination.

        Returns:
            Search URL.
        """
        base_url = "https://www.google.com/search"
        params = {
            "q": keyword,
            "tbm": "isch",
            "start": start,
            "ijn": "0",
        }
        param_str = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{base_url}?{param_str}"

    def _extract_image_urls(self, html: str) -> List[str]:
        """
        Extract image URLs from Google Images search results.

        Args:
            html: HTML content of search results page.

        Returns:
            List of image URLs.
        """
        urls: List[str] = []
        seen: Set[str] = set()

        # Extract URLs from data-src and src attributes
        src_pattern = r'\["(https://[^"]+)",\d+,\d+\]'
        matches = re.findall(src_pattern, html)

        for url in matches:
            # Clean up URL
            url = url.replace(r"\u003d", "=").replace(r"\u0026", "&")

            # Filter for image URLs
            if any(ext in url.lower() for ext in ["jpg", "jpeg", "png", "webp"]):
                if url not in seen:
                    seen.add(url)
                    urls.append(url)

        return urls

    def _download_images(
        self,
        urls: List[str],
        output_dir: Path,
        keyword: str,
        keywords: Optional[List[str]] = None,
    ) -> List[Path]:
        """
        Download images concurrently.

        Args:
            urls: List of image URLs.
            output_dir: Directory to save images.
            keyword: Keyword for checkpoint naming.
            keywords: Keywords for content validation.

        Returns:
            List of paths to downloaded images.
        """
        downloaded_paths: List[Path] = []

        # Get already downloaded URLs if resume is enabled
        completed_urls = set()
        if self.checkpoint_manager:
            completed = self.checkpoint_manager.get_completed_items()
            completed_urls = set(completed)

        # Filter out already downloaded URLs
        remaining_urls = [url for url in urls if url not in completed_urls]

        if not remaining_urls:
            logger.info(f"All images already downloaded for '{keyword}'")
            return downloaded_paths

        logger.info(f"Downloading {len(remaining_urls)} images for '{keyword}'")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    self._download_single_image,
                    url,
                    output_dir,
                    idx,
                    keywords,
                ): (url, idx)
                for idx, url in enumerate(remaining_urls)
            }

            with tqdm(
                total=len(remaining_urls),
                desc=f"Downloading {keyword}",
                unit="img",
            ) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    url, idx = futures[future]
                    try:
                        result = future.result()
                        if result:
                            downloaded_paths.append(result)
                            if self.checkpoint_manager:
                                self.checkpoint_manager.add_completed(url)
                    except Exception as e:
                        logger.error(f"Failed to download {url}: {e}")
                    pbar.update(1)

        return downloaded_paths

    def _download_single_image(
        self,
        url: str,
        output_dir: Path,
        idx: int,
        keywords: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """
        Download a single image.

        Args:
            url: Image URL.
            output_dir: Directory to save image.
            idx: Index for filename.
            keywords: Keywords for content validation.

        Returns:
            Path to downloaded file or None if failed.
        """
        try:
            response = self.request_with_retry(url, stream=True)
            if response is None:
                return None

            # Determine file extension
            content_type = response.headers.get("content-type", "")
            ext = self._get_extension_from_url(url) or self._get_extension_from_mime(
                content_type
            )

            if not ext:
                logger.warning(f"Could not determine extension for {url}")
                return None

            filename = f"image_{idx:05d}{ext}"
            filepath = output_dir / filename

            # Download and save
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Verify file is valid image
            try:
                from PIL import Image
                with Image.open(filepath) as img:
                    img.verify()
            except Exception as e:
                logger.warning(f"Invalid image file {filepath}: {e}")
                filepath.unlink()
                return None

            # Validate content if validator is enabled
            if self.validator and keywords:
                is_valid = self.validator.validate(filepath, keywords)
                if not is_valid:
                    logger.debug(f"Image failed content validation: {filepath}")
                    filepath.unlink()
                    return None

            return filepath

        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None

    @staticmethod
    def _sanitize_keyword(keyword: str) -> str:
        """
        Sanitize keyword for use as directory name.

        Args:
            keyword: Raw keyword string.

        Returns:
            Sanitized keyword.
        """
        # Remove special characters and replace spaces with underscores
        sanitized = re.sub(r'[<>:"/\\|?*]', "", keyword)
        sanitized = sanitized.strip().replace(" ", "_")
        return sanitized

    @staticmethod
    def _get_extension_from_url(url: str) -> Optional[str]:
        """
        Extract file extension from URL.

        Args:
            url: Image URL.

        Returns:
            File extension with dot or None.
        """
        path = url.split("?")[0]  # Remove query parameters
        ext = Path(path).suffix.lower()
        return ext if ext in ImageSpider.IMAGE_EXTENSIONS else None

    @staticmethod
    def _get_extension_from_mime(mime_type: str) -> Optional[str]:
        """
        Get file extension from MIME type.

        Args:
            mime_type: MIME type string.

        Returns:
            File extension with dot or None.
        """
        mime_to_ext = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
            "image/gif": ".gif",
        }
        return mime_to_ext.get(mime_type.lower())
