"""
Pexels API crawler for high-quality free stock images.

API Documentation: https://www.pexels.com/api/documentation/
Rate Limit: 200 requests per hour
License: Pexels License (free for commercial use)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from floodsense.crawlers.api_crawlers.base_api_crawler import BaseAPICrawler
from floodsense.crawlers.base import BaseCrawler
from floodsense.utils.config import CrawlerConfig
from floodsense.utils.proxy import ProxyManager


@BaseCrawler.register("pexels")
class PexelsCrawler(BaseAPICrawler):
    """
    Crawler for Pexels API.

    Requires a free API key from https://www.pexels.com/api/
    """

    RATE_LIMIT = 200  # requests per hour

    def __init__(
        self,
        config: Optional[CrawlerConfig] = None,
        proxy_manager: Optional[ProxyManager] = None,
        output_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize PexelsCrawler.

        Args:
            config: Crawler configuration.
            proxy_manager: Optional proxy manager.
            output_dir: Directory to save downloaded files.
            api_key: Pexels API key.
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
                "Pexels API key not provided. "
                "Set FLOODSENSE_PEXELS_API_KEY environment variable."
            )

    @property
    def api_base_url(self) -> str:
        """Return the Pexels API base URL."""
        return "https://api.pexels.com/v1"

    @property
    def source_name(self) -> str:
        """Return the source name."""
        return "Pexels"

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get Pexels authentication headers.

        Returns:
            Dictionary with Authorization header.
        """
        if self.api_key:
            return {"Authorization": self.api_key}
        return {}

    def _build_search_params(
        self, keyword: str, page: int, per_page: int
    ) -> Dict[str, Any]:
        """
        Build Pexels search parameters.

        Args:
            keyword: Search keyword.
            page: Page number.
            per_page: Results per page (max 80).

        Returns:
            Query parameters dictionary.
        """
        return {
            "query": keyword,
            "page": page,
            "per_page": min(per_page, 80),
            "orientation": "landscape",
        }

    def _parse_search_response(
        self, response_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Parse Pexels search response.

        Args:
            response_data: Raw API response.

        Returns:
            List of image dictionaries.
        """
        images = []
        photos = response_data.get("photos", [])

        for photo in photos:
            # Get the large size URL (good balance of quality and size)
            src = photo.get("src", {})
            url = src.get("large2x") or src.get("large") or src.get("original")

            if url:
                images.append({
                    "url": url,
                    "id": photo.get("id"),
                    "width": photo.get("width"),
                    "height": photo.get("height"),
                    "description": photo.get("alt"),
                    "author": photo.get("photographer"),
                    "license": "Pexels License",
                })

        return images

    def _get_total_pages(self, response_data: Dict[str, Any], per_page: int) -> int:
        """
        Get total pages from Pexels response.

        Args:
            response_data: Raw API response.
            per_page: Results per page.

        Returns:
            Total number of pages.
        """
        total_results = response_data.get("total_results", 0)
        if total_results == 0:
            return 0
        return (total_results + per_page - 1) // per_page

    def get_image_urls(self, keyword: str, max_results: int = 100) -> List[str]:
        """
        Get image URLs from Pexels.

        Overrides base method to use correct endpoint.

        Args:
            keyword: Search keyword.
            max_results: Maximum number of URLs.

        Returns:
            List of image URLs.
        """
        if not self.api_key:
            logger.error("Pexels API key required but not provided")
            return []

        urls: List[str] = []
        per_page = min(80, max_results)
        page = 1
        total_pages = None

        logger.info(f"Searching Pexels for: {keyword}")

        while len(urls) < max_results:
            params = self._build_search_params(keyword, page, per_page)
            response_data = self._make_api_request("/search", params)

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

        logger.info(f"Found {len(urls)} images from Pexels")
        return urls
