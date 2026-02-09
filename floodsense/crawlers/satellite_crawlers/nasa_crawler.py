"""
NASA Earth Observatory crawler for satellite imagery.

Website: https://earthobservatory.nasa.gov/
Rate Limit: None (but be respectful)
License: Public Domain (NASA imagery is not copyrighted)
"""

import re
from pathlib import Path
from typing import List, Optional, Set

from bs4 import BeautifulSoup
from loguru import logger

from floodsense.crawlers.base import BaseCrawler
from floodsense.utils.config import CrawlerConfig
from floodsense.utils.file_utils import FileUtils
from floodsense.utils.proxy import ProxyManager


@BaseCrawler.register("nasa")
class NASACrawler(BaseCrawler):
    """
    Crawler for NASA Earth Observatory imagery.

    Uses HTML scraping since NASA doesn't have a formal API for Earth Observatory.
    All NASA imagery is in the public domain.
    """

    BASE_URL = "https://earthobservatory.nasa.gov"
    SEARCH_URL = f"{BASE_URL}/search"

    # Categories relevant to floods
    FLOOD_CATEGORIES = [
        "floods",
        "hurricanes",
        "severe-storms",
        "water",
    ]

    def __init__(
        self,
        config: Optional[CrawlerConfig] = None,
        proxy_manager: Optional[ProxyManager] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize NASACrawler.

        Args:
            config: Crawler configuration.
            proxy_manager: Optional proxy manager.
            output_dir: Directory to save downloaded files.
        """
        super().__init__(
            config=config,
            proxy_manager=proxy_manager,
            output_dir=output_dir,
        )

    @property
    def source_name(self) -> str:
        """Return the source name."""
        return "NASA"

    def get_image_urls(self, keyword: str, max_results: int = 100) -> List[str]:
        """
        Get image URLs from NASA Earth Observatory.

        Args:
            keyword: Search keyword.
            max_results: Maximum number of URLs.

        Returns:
            List of image URLs.
        """
        urls: List[str] = []
        seen: Set[str] = set()
        page = 1

        logger.info(f"Searching NASA Earth Observatory for: {keyword}")

        while len(urls) < max_results:
            search_url = f"{self.SEARCH_URL}?q={keyword}&pg={page}"

            response = self.request_with_retry(search_url)
            if response is None:
                logger.warning("Failed to get NASA search results")
                break

            # Parse search results page
            page_urls = self._extract_article_image_urls(response.text)

            if not page_urls:
                logger.debug("No more results from NASA")
                break

            for url in page_urls:
                if url not in seen and len(urls) < max_results:
                    seen.add(url)
                    urls.append(url)

            # Check if we should continue to next page
            if "pg=" not in response.text or len(page_urls) < 10:
                break

            page += 1

        logger.info(f"Found {len(urls)} images from NASA Earth Observatory")
        return urls

    def _extract_article_image_urls(self, html: str) -> List[str]:
        """
        Extract image URLs from NASA search results page.

        Args:
            html: HTML content of search results.

        Returns:
            List of image URLs.
        """
        urls: List[str] = []
        soup = BeautifulSoup(html, "html.parser")

        # Find article links first
        articles = soup.find_all("a", class_="card-image")
        if not articles:
            # Try alternative selectors
            articles = soup.find_all("article")

        for article in articles:
            # Get article URL
            if article.name == "a":
                article_url = article.get("href", "")
            else:
                link = article.find("a")
                article_url = link.get("href", "") if link else ""

            if not article_url:
                continue

            if not article_url.startswith("http"):
                article_url = f"{self.BASE_URL}{article_url}"

            # Fetch article page to get full-resolution image
            article_response = self.request_with_retry(article_url)
            if article_response:
                image_url = self._extract_main_image(article_response.text)
                if image_url:
                    urls.append(image_url)

        # Also extract thumbnail images from search results
        thumbnails = soup.find_all("img")
        for img in thumbnails:
            src = img.get("src") or img.get("data-src", "")
            if src and any(
                ext in src.lower() for ext in [".jpg", ".jpeg", ".png"]
            ):
                if not src.startswith("http"):
                    src = f"{self.BASE_URL}{src}"
                # Try to get larger version by modifying URL
                large_url = self._get_large_image_url(src)
                if large_url and large_url not in urls:
                    urls.append(large_url)

        return urls

    def _extract_main_image(self, html: str) -> Optional[str]:
        """
        Extract main image URL from NASA article page.

        Args:
            html: HTML content of article page.

        Returns:
            Image URL or None.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Look for the main image in various locations
        # NASA often has a "lede-image" or similar class
        selectors = [
            ("figure", {"class": "lede-image"}),
            ("div", {"class": "main-image"}),
            ("figure", {"class": "image"}),
            ("div", {"class": "image-wrapper"}),
        ]

        for tag, attrs in selectors:
            container = soup.find(tag, attrs)
            if container:
                img = container.find("img")
                if img:
                    src = img.get("src") or img.get("data-src")
                    if src:
                        if not src.startswith("http"):
                            src = f"{self.BASE_URL}{src}"
                        return self._get_large_image_url(src)

        # Look for any large image in the content
        content = soup.find("div", class_="content") or soup.find("article")
        if content:
            images = content.find_all("img")
            for img in images:
                src = img.get("src") or img.get("data-src", "")
                # Skip icons and small images
                if src and not any(skip in src for skip in ["icon", "logo", "button"]):
                    if not src.startswith("http"):
                        src = f"{self.BASE_URL}{src}"
                    return self._get_large_image_url(src)

        # Look for meta image
        meta_image = soup.find("meta", property="og:image")
        if meta_image:
            return meta_image.get("content")

        return None

    def _get_large_image_url(self, url: str) -> str:
        """
        Convert thumbnail URL to full-size image URL.

        NASA uses patterns like _lrg, _full, etc.

        Args:
            url: Original image URL.

        Returns:
            Large image URL.
        """
        # Remove size suffixes and try to get largest version
        patterns = [
            (r"_th\.", "_lrg."),
            (r"_thumb\.", "_lrg."),
            (r"_med\.", "_lrg."),
            (r"_small\.", "_lrg."),
        ]

        for pattern, replacement in patterns:
            if re.search(pattern, url):
                return re.sub(pattern, replacement, url)

        return url

    def crawl(self, keywords: List[str], max_results: int = 100) -> List[Path]:
        """
        Crawl images for given keywords.

        Args:
            keywords: List of search keywords.
            max_results: Maximum images per keyword.

        Returns:
            List of paths to downloaded files.
        """
        downloaded_paths: List[Path] = []

        for keyword in keywords:
            logger.info(f"[NASA] Crawling for: {keyword}")
            urls = self.get_image_urls(keyword, max_results)

            if not urls:
                continue

            keyword_dir = self.output_dir / FileUtils.sanitize_keyword(keyword)
            keyword_dir.mkdir(parents=True, exist_ok=True)

            for idx, url in enumerate(urls):
                filepath = keyword_dir / f"nasa_{idx:05d}.jpg"
                if self.download_file(url, filepath):
                    downloaded_paths.append(filepath)

        return downloaded_paths

    def get_category_images(
        self, category: str = "floods", max_results: int = 100
    ) -> List[str]:
        """
        Get images from a specific NASA Earth Observatory category.

        Args:
            category: Category name (e.g., "floods", "hurricanes").
            max_results: Maximum number of URLs.

        Returns:
            List of image URLs.
        """
        urls: List[str] = []
        page = 1

        logger.info(f"Fetching NASA images from category: {category}")

        while len(urls) < max_results:
            category_url = f"{self.BASE_URL}/topic/{category}?pg={page}"

            response = self.request_with_retry(category_url)
            if response is None:
                break

            page_urls = self._extract_article_image_urls(response.text)

            if not page_urls:
                break

            urls.extend(page_urls)
            page += 1

        return urls[:max_results]
