"""
Wikimedia Commons API crawler for freely-licensed images.

API Documentation: https://www.mediawiki.org/wiki/API:Main_page
Rate Limit: None (but be reasonable)
License: Various CC licenses (CC-BY-SA, CC0, etc.)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from floodsense.crawlers.api_crawlers.base_api_crawler import BaseAPICrawler
from floodsense.crawlers.base import BaseCrawler
from floodsense.utils.config import CrawlerConfig
from floodsense.utils.proxy import ProxyManager


@BaseCrawler.register("wikimedia")
class WikimediaCrawler(BaseAPICrawler):
    """
    Crawler for Wikimedia Commons API.

    No API key required. Uses the MediaWiki API.
    """

    RATE_LIMIT = 500  # Self-imposed reasonable limit

    def __init__(
        self,
        config: Optional[CrawlerConfig] = None,
        proxy_manager: Optional[ProxyManager] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize WikimediaCrawler.

        Args:
            config: Crawler configuration.
            proxy_manager: Optional proxy manager.
            output_dir: Directory to save downloaded files.
        """
        super().__init__(
            config=config,
            proxy_manager=proxy_manager,
            output_dir=output_dir,
            api_key=None,  # No API key needed
            rate_limit=self.RATE_LIMIT,
        )

    @property
    def api_base_url(self) -> str:
        """Return the Wikimedia Commons API base URL."""
        return "https://commons.wikimedia.org/w/api.php"

    @property
    def source_name(self) -> str:
        """Return the source name."""
        return "Wikimedia"

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers (none required for Wikimedia).

        Returns:
            Empty dictionary.
        """
        return {}

    def _build_search_params(
        self, keyword: str, page: int, per_page: int
    ) -> Dict[str, Any]:
        """
        Build Wikimedia search parameters.

        Args:
            keyword: Search keyword.
            page: Page number (used for continue token).
            per_page: Results per page.

        Returns:
            Query parameters dictionary.
        """
        return {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {keyword}",
            "gsrnamespace": "6",  # File namespace
            "gsrlimit": min(per_page, 50),
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            "iiurlwidth": 1920,
        }

    def _parse_search_response(
        self, response_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Parse Wikimedia search response.

        Args:
            response_data: Raw API response.

        Returns:
            List of image dictionaries.
        """
        images = []
        query = response_data.get("query", {})
        pages = query.get("pages", {})

        for page_id, page_data in pages.items():
            imageinfo = page_data.get("imageinfo", [])
            if not imageinfo:
                continue

            info = imageinfo[0]
            url = info.get("thumburl") or info.get("url")

            if url:
                # Extract metadata
                metadata = info.get("extmetadata", {})
                description = metadata.get("ImageDescription", {}).get("value", "")
                author = metadata.get("Artist", {}).get("value", "")
                license_info = metadata.get("LicenseShortName", {}).get("value", "")

                images.append({
                    "url": url,
                    "id": page_id,
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "description": description[:200] if description else None,
                    "author": author[:100] if author else None,
                    "license": license_info or "Wikimedia Commons",
                })

        return images

    def _get_total_pages(self, response_data: Dict[str, Any], per_page: int) -> int:
        """
        Get total pages from Wikimedia response.

        Wikimedia uses continue tokens instead of page counts.

        Args:
            response_data: Raw API response.
            per_page: Results per page.

        Returns:
            Estimated number of pages (or large number if continue exists).
        """
        # Wikimedia uses continue tokens, so we check for presence
        if "continue" in response_data:
            return 999  # Indicates more results available
        return 1

    def get_image_urls(self, keyword: str, max_results: int = 100) -> List[str]:
        """
        Get image URLs from Wikimedia Commons.

        Overrides base method to handle continue tokens.

        Args:
            keyword: Search keyword.
            max_results: Maximum number of URLs.

        Returns:
            List of image URLs.
        """
        urls: List[str] = []
        per_page = min(50, max_results)
        continue_token = None

        logger.info(f"Searching Wikimedia Commons for: {keyword}")

        while len(urls) < max_results:
            params = self._build_search_params(keyword, 1, per_page)

            # Add continue token if we have one
            if continue_token:
                params.update(continue_token)

            # Make direct request (not using base method due to different URL structure)
            self.rate_limiter.wait_if_needed()

            try:
                response = self.session.get(
                    self.api_base_url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=self.config.timeout,
                    proxies=self._get_proxies(),
                )
                self.rate_limiter.record_request()
                response.raise_for_status()
                response_data = response.json()

            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                logger.exception(f"Wikimedia API request failed: {e}")
                break

            images = self._parse_search_response(response_data)

            if not images:
                break

            for image in images:
                if len(urls) >= max_results:
                    break
                if image.get("url"):
                    urls.append(image["url"])

            # Check for continue token
            if "continue" in response_data:
                continue_token = response_data["continue"]
            else:
                break

        logger.info(f"Found {len(urls)} images from Wikimedia Commons")
        return urls
