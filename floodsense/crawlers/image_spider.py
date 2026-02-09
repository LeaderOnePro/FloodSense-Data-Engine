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
from floodsense.utils.file_utils import CheckpointManager, FileUtils
from floodsense.validators.image_validator import ImageValidator

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
except ImportError:
    PlaywrightError = OSError
    PlaywrightTimeoutError = TimeoutError


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
            checkpoint_dir = Path("data/.checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir, task_name="image_spider"
            )
        downloaded_paths: List[Path] = []

        for keyword in keywords:
            logger.info(f"Starting crawl for keyword: {keyword}")
            urls = self.get_image_urls(keyword, max_results=max_results)
            logger.info(f"Found {len(urls)} image URLs for '{keyword}'")

            keyword_dir = self.output_dir / FileUtils.sanitize_keyword(keyword)
            keyword_dir.mkdir(parents=True, exist_ok=True)

            paths = self._download_images(urls, keyword_dir, keyword, keywords)
            downloaded_paths.extend(paths)

        return downloaded_paths

    def get_image_urls(self, keyword: str, max_results: int = 100) -> List[str]:
        """
        Get image URLs using Bing Images search with Playwright.

        Args:
            keyword: Search keyword.
            max_results: Maximum number of URLs to return.

        Returns:
            List of image URLs.
        """
        from playwright.sync_api import sync_playwright
        import time

        urls: List[str] = []
        seen: Set[str] = set()

        search_url = self._build_search_url(keyword, 0, engine="bing")
        logger.info(f"Opening browser for: {search_url}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=self.user_agents[0],
                viewport={"width": 1920, "height": 1080},
            )
            page = context.new_page()

            try:
                page.goto(search_url, wait_until="networkidle", timeout=30000)
                time.sleep(2)

                # Scroll to load more images
                scroll_count = 0
                max_scrolls = max(10, max_results // 15)

                while len(urls) < max_results and scroll_count < max_scrolls:
                    # Extract image URLs from current page
                    page_content = page.content()
                    page_urls = self._extract_image_urls(page_content)

                    for url in page_urls:
                        if url not in seen:
                            seen.add(url)
                            urls.append(url)

                    if len(urls) >= max_results:
                        break

                    # Scroll down
                    page.evaluate("window.scrollBy(0, window.innerHeight)")
                    time.sleep(0.5)
                    scroll_count += 1

                    # Click "See more images" button for Bing if present
                    try:
                        see_more = page.locator("a.btn_seemore")
                        if see_more.is_visible(timeout=500):
                            see_more.click()
                            time.sleep(1)
                    except (PlaywrightTimeoutError, PlaywrightError):
                        pass

                logger.info(f"Extracted {len(urls)} URLs after {scroll_count} scrolls")

            except (PlaywrightError, PlaywrightTimeoutError, OSError) as e:
                logger.exception(f"Playwright error: {e}")
            finally:
                browser.close()

        return urls[:max_results]

    def _build_search_url(self, keyword: str, start: int = 0, engine: str = "bing") -> str:
        """
        Build image search URL.

        Args:
            keyword: Search keyword.
            start: Starting index for pagination.
            engine: Search engine to use ("google" or "bing").

        Returns:
            Search URL.
        """
        if engine == "bing":
            from urllib.parse import quote
            base_url = "https://www.bing.com/images/search"
            return f"{base_url}?q={quote(keyword)}&first={start}&count=150&qft=+filterui:photo-photo"
        else:
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
        Extract image URLs from search results.

        Args:
            html: HTML content of search results page.

        Returns:
            List of image URLs.
        """
        from urllib.parse import unquote

        urls: List[str] = []
        seen: Set[str] = set()

        # Bing pattern: mediaurl parameter
        bing_pattern = r'mediaurl=([^&"]+)'
        bing_matches = re.findall(bing_pattern, html)
        for url in bing_matches:
            url = unquote(url)
            if any(ext in url.lower() for ext in ["jpg", "jpeg", "png", "webp"]):
                if url not in seen and url.startswith("http"):
                    seen.add(url)
                    urls.append(url)

        # Google pattern: data-src and src attributes
        google_pattern = r'\["(https://[^"]+)",\d+,\d+\]'
        google_matches = re.findall(google_pattern, html)
        for url in google_matches:
            url = url.replace(r"\u003d", "=").replace(r"\u0026", "&")
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
                    except (requests.exceptions.RequestException, OSError, ValueError) as e:
                        logger.exception(f"Failed to download {url}: {e}")
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
                from PIL import Image, UnidentifiedImageError
                with Image.open(filepath) as img:
                    img.verify()
            except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as e:
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

        except (requests.exceptions.RequestException, OSError) as e:
            logger.exception(f"Error downloading {url}: {e}")
            return None

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
