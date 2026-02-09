"""
Flickr API crawler for Creative Commons licensed images.

API Documentation: https://www.flickr.com/services/api/
Rate Limit: 3600 requests per hour
License: Various CC licenses
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from floodsense.crawlers.api_crawlers.base_api_crawler import BaseAPICrawler
from floodsense.utils.config import CrawlerConfig
from floodsense.utils.proxy import ProxyManager


class FlickrCrawler(BaseAPICrawler):
    """
    Crawler for Flickr API.

    Requires a free API key from https://www.flickr.com/services/apps/create/
    """

    RATE_LIMIT = 3600  # requests per hour

    # Flickr license IDs for Creative Commons licenses
    # 1=CC BY-NC-SA, 2=CC BY-NC, 3=CC BY-NC-ND, 4=CC BY, 5=CC BY-SA, 6=CC BY-ND
    # 7=No known copyright, 8=US Government Work, 9=CC0, 10=PDM
    CC_LICENSE_IDS = "1,2,3,4,5,6,7,8,9,10"

    LICENSE_NAMES = {
        "0": "All Rights Reserved",
        "1": "CC BY-NC-SA 2.0",
        "2": "CC BY-NC 2.0",
        "3": "CC BY-NC-ND 2.0",
        "4": "CC BY 2.0",
        "5": "CC BY-SA 2.0",
        "6": "CC BY-ND 2.0",
        "7": "No known copyright restrictions",
        "8": "United States Government Work",
        "9": "CC0 1.0",
        "10": "Public Domain Mark",
    }

    def __init__(
        self,
        config: Optional[CrawlerConfig] = None,
        proxy_manager: Optional[ProxyManager] = None,
        output_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize FlickrCrawler.

        Args:
            config: Crawler configuration.
            proxy_manager: Optional proxy manager.
            output_dir: Directory to save downloaded files.
            api_key: Flickr API key.
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
                "Flickr API key not provided. "
                "Set FLOODSENSE_FLICKR_API_KEY environment variable."
            )

    @property
    def api_base_url(self) -> str:
        """Return the Flickr API base URL."""
        return "https://api.flickr.com/services/rest"

    @property
    def source_name(self) -> str:
        """Return the source name."""
        return "Flickr"

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers (Flickr uses query params instead).

        Returns:
            Empty dictionary (API key is in params).
        """
        return {}

    def _build_search_params(
        self, keyword: str, page: int, per_page: int
    ) -> Dict[str, Any]:
        """
        Build Flickr search parameters.

        Args:
            keyword: Search keyword.
            page: Page number.
            per_page: Results per page (max 500).

        Returns:
            Query parameters dictionary.
        """
        return {
            "method": "flickr.photos.search",
            "api_key": self.api_key,
            "format": "json",
            "nojsoncallback": 1,
            "text": keyword,
            "page": page,
            "per_page": min(per_page, 500),
            "license": self.CC_LICENSE_IDS,
            "media": "photos",
            "content_type": 1,  # Photos only
            "sort": "relevance",
            "extras": "url_l,url_o,url_k,license,owner_name,description",
        }

    def _parse_search_response(
        self, response_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Parse Flickr search response.

        Args:
            response_data: Raw API response.

        Returns:
            List of image dictionaries.
        """
        images = []
        photos = response_data.get("photos", {})
        photo_list = photos.get("photo", [])

        for photo in photo_list:
            # Prefer larger sizes: url_k (2048px) > url_o (original) > url_l (1024px)
            url = photo.get("url_k") or photo.get("url_o") or photo.get("url_l")

            if not url:
                # Build URL from photo info if extras not available
                url = self._build_photo_url(photo)

            if url:
                license_id = str(photo.get("license", "0"))
                images.append({
                    "url": url,
                    "id": photo.get("id"),
                    "title": photo.get("title"),
                    "description": photo.get("description", {}).get("_content", ""),
                    "author": photo.get("ownername"),
                    "license": self.LICENSE_NAMES.get(license_id, "Unknown"),
                })

        return images

    def _build_photo_url(self, photo: Dict[str, Any]) -> Optional[str]:
        """
        Build photo URL from photo info.

        Args:
            photo: Photo data from API.

        Returns:
            Photo URL or None.
        """
        server = photo.get("server")
        photo_id = photo.get("id")
        secret = photo.get("secret")

        if all([server, photo_id, secret]):
            # Use 'b' size (1024px on longest side)
            return f"https://live.staticflickr.com/{server}/{photo_id}_{secret}_b.jpg"
        return None

    def _get_total_pages(self, response_data: Dict[str, Any], per_page: int) -> int:
        """
        Get total pages from Flickr response.

        Args:
            response_data: Raw API response.
            per_page: Results per page.

        Returns:
            Total number of pages.
        """
        photos = response_data.get("photos", {})
        return photos.get("pages", 0)

    def get_image_urls(self, keyword: str, max_results: int = 100) -> List[str]:
        """
        Get image URLs from Flickr.

        Overrides base method to use correct request structure.

        Args:
            keyword: Search keyword.
            max_results: Maximum number of URLs.

        Returns:
            List of image URLs.
        """
        if not self.api_key:
            logger.error("Flickr API key required but not provided")
            return []

        urls: List[str] = []
        per_page = min(100, max_results)
        page = 1
        total_pages = None

        logger.info(f"Searching Flickr for: {keyword}")

        while len(urls) < max_results:
            params = self._build_search_params(keyword, page, per_page)

            # Make direct request since Flickr uses query params for auth
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

                # Check for API errors
                if response_data.get("stat") != "ok":
                    logger.error(
                        f"Flickr API error: {response_data.get('message', 'Unknown error')}"
                    )
                    break

            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                logger.exception(f"Flickr API request failed: {e}")
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

        logger.info(f"Found {len(urls)} images from Flickr")
        return urls
