"""
Unsplash API crawler for high-quality free images.

API Documentation: https://unsplash.com/documentation
Rate Limit: 50 requests per hour (free tier)
License: Unsplash License (free for commercial and non-commercial use)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from floodsense.crawlers.api_crawlers.base_api_crawler import BaseAPICrawler
from floodsense.crawlers.base import BaseCrawler
from floodsense.utils.config import CrawlerConfig
from floodsense.utils.proxy import ProxyManager


@BaseCrawler.register("unsplash")
class UnsplashCrawler(BaseAPICrawler):
    """
    Crawler for Unsplash API.

    Requires a free API key from https://unsplash.com/developers
    """

    RATE_LIMIT = 50  # requests per hour for free tier

    def __init__(
        self,
        config: Optional[CrawlerConfig] = None,
        proxy_manager: Optional[ProxyManager] = None,
        output_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize UnsplashCrawler.

        Args:
            config: Crawler configuration.
            proxy_manager: Optional proxy manager.
            output_dir: Directory to save downloaded files.
            api_key: Unsplash API access key.
        """
        super().__init__(
            config=config,
            proxy_manager=proxy_manager,
            output_dir=output_dir,
            api_key=api_key,
            rate_limit=self.RATE_LIMIT,
        )
        if not api_key:
            logger.warning(
                "Unsplash API key not provided. "
                "Set FLOODSENSE_UNSPLASH_API_KEY environment variable."
            )

    @property
    def api_base_url(self) -> str:
        """Return the Unsplash API base URL."""
        return "https://api.unsplash.com"

    @property
    def source_name(self) -> str:
        """Return the source name."""
        return "Unsplash"

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get Unsplash authentication headers.

        Returns:
            Dictionary with Authorization header.
        """
        if self.api_key:
            return {"Authorization": f"Client-ID {self.api_key}"}
        return {}

    def _build_search_params(
        self, keyword: str, page: int, per_page: int
    ) -> Dict[str, Any]:
        """
        Build Unsplash search parameters.

        Args:
            keyword: Search keyword.
            page: Page number.
            per_page: Results per page (max 30).

        Returns:
            Query parameters dictionary.
        """
        return {
            "query": keyword,
            "page": page,
            "per_page": min(per_page, 30),
            "orientation": "landscape",
            "content_filter": "high",
        }

    def _parse_search_response(
        self, response_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Parse Unsplash search response.

        Args:
            response_data: Raw API response.

        Returns:
            List of image dictionaries.
        """
        images = []
        results = response_data.get("results", [])

        for photo in results:
            # Get the regular size URL (good quality, not too large)
            urls = photo.get("urls", {})
            url = urls.get("regular") or urls.get("full") or urls.get("raw")

            if url:
                images.append({
                    "url": url,
                    "id": photo.get("id"),
                    "width": photo.get("width"),
                    "height": photo.get("height"),
                    "description": photo.get("description") or photo.get("alt_description"),
                    "author": photo.get("user", {}).get("name"),
                    "license": "Unsplash License",
                })

        return images

    def _get_total_pages(self, response_data: Dict[str, Any], per_page: int) -> int:
        """
        Get total pages from Unsplash response.

        Args:
            response_data: Raw API response.
            per_page: Results per page.

        Returns:
            Total number of pages.
        """
        total = response_data.get("total", 0)
        if total == 0:
            return 0
        return (total + per_page - 1) // per_page

    def get_image_urls(self, keyword: str, max_results: int = 100) -> List[str]:
        """
        Get image URLs from Unsplash.

        Overrides base method to use correct endpoint.

        Args:
            keyword: Search keyword.
            max_results: Maximum number of URLs.

        Returns:
            List of image URLs.
        """
        if not self.api_key:
            logger.error("Unsplash API key required but not provided")
            return []

        urls: List[str] = []
        per_page = min(30, max_results)
        page = 1
        total_pages = None

        logger.info(f"Searching Unsplash for: {keyword}")

        while len(urls) < max_results:
            params = self._build_search_params(keyword, page, per_page)
            response_data = self._make_api_request("/search/photos", params)

            if response_data is None:
                break

            images = self._parse_search_response(response_data)

            if not images:
                break

            for image in images:
                if len(urls) >= max_results:
                    break
                if image.get("url"):
                    urls.append(image["url"])

            if total_pages is None:
                total_pages = self._get_total_pages(response_data, per_page)

            if total_pages and page >= total_pages:
                break

            page += 1

        logger.info(f"Found {len(urls)} images from Unsplash")
        return urls
